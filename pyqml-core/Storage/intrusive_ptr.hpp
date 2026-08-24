
#pragma once
#include <atomic>
#include <type_traits>
#include <stdexcept>
#include <concepts>

class refcount{
    mutable std::atomic<int32_t> ref_count{0};

    public:
    void incref() const {
        /*std::cout << "current.count: " << ref_count.load() << std::endl;
        std::cout << "new.count: " << ref_count.load() << std::endl;*/
        ref_count.fetch_add(1, std::memory_order_relaxed);
    }

    void decref() const{

        //std::cout << "current.count:" << ref_count.load() << std::endl;
        
        if (ref_count.fetch_sub(1, std::memory_order_acq_rel) == 1){
            /*
            std::cout << "new.count:" << ref_count.load() << std::endl;
            std::cout << "Deleting underlying storage at memor adress:" << this << std::endl;
            */
            delete this;
        }
    }

    virtual ~refcount() = default;

};

class Storage: public refcount{
    void* buffer; 

    void(*dtor)(void * p);

    size_t bytes_size;

    public:

    Storage(void* buf, void(*dtr)(void* p), size_t size): buffer(buf), dtor(dtr), bytes_size(size) {}

    template <typename U> 

    Storage(U* buff, size_t size): buffer(buff), bytes_size(size) {
        dtor = [](void * p){delete[] static_cast<U*>(p); };
    }

    void* get() const noexcept { return buffer; }

    size_t bytes() const noexcept { return bytes_size; }

    template <typename U> 

    U* get_typed() const noexcept { return static_cast<U*>(buffer); }

    template <typename U>
    U* data_ptr() const noexcept { return get_typed<U>(); }

    protected:

    ~Storage() override {
        dtor(buffer);
    }





};




template <typename Der>
concept IS_REF_COUNTED = std::is_base_of_v<refcount, Der>;

template <typename T>
class pyq_intrusive_ptr{

    T* storage;

    public:

        pyq_intrusive_ptr(T* stg)  {
            storage = stg;
            storage->incref();
        }

        pyq_intrusive_ptr(decltype(nullptr)): storage(nullptr) {}

        template<typename... Args>
        requires (!std::same_as<std::remove_cvref_t<Args>, pyq_intrusive_ptr> && ...) /*&& std::constructible_from<T, Args...>*/
        
        explicit pyq_intrusive_ptr(Args&& ... args){
            storage = new T(std::forward<Args>(args) ...);
            storage->incref();
        }

        template <typename... Args>

        pyq_intrusive_ptr<T> make_intrusive(Args&&... args){
            return pyq_intrusive_ptr(std::forward<Args>(args)...);
        }

        

        explicit operator bool() const noexcept {
         return storage != nullptr;
        }   

        T* operator->() const{
            return storage;
        }

        T& operator*() const{
            return *storage;
        }

        pyq_intrusive_ptr(): storage(nullptr) {}

        T* storage_ptr() const noexcept {return storage;}

        const T* storage_ptr() const noexcept {return storage;}

        pyq_intrusive_ptr(const pyq_intrusive_ptr&);

        pyq_intrusive_ptr(pyq_intrusive_ptr&&);

        pyq_intrusive_ptr& operator=(pyq_intrusive_ptr&&) noexcept;

        pyq_intrusive_ptr& operator=(const pyq_intrusive_ptr& ) noexcept;

        ~pyq_intrusive_ptr(); 
        
    
};


class StorageRef{
    pyq_intrusive_ptr<Storage> stg_;


    public:
        StorageRef() = default;

        StorageRef(pyq_intrusive_ptr<Storage> ptr): stg_(std::move(ptr)) {}

        void* data() const noexcept {
            return stg_ ? stg_->get() : nullptr;
        }

        void* get_void() const noexcept {
            return data();
        }

        template <typename T>
        T* get() const noexcept {
            return stg_ ? stg_->template get_typed<T>() : nullptr;
        }

        template <typename T>
        T* data_ptr() const noexcept {
            return get<T>();
        }

        template <typename... Args>
        StorageRef(Args&&... args): stg_(std::forward<Args>(args)...) {}

        size_t nbytes() const noexcept {
            return stg_ ? stg_->bytes() : 0;
        }

        const pyq_intrusive_ptr<Storage>& storage_ptr() const noexcept {
            return stg_;
        }

        pyq_intrusive_ptr<Storage> owner() const noexcept {
            return stg_;
        }

        operator bool() const noexcept{
            return static_cast<bool>(stg_);
        }

        Storage* operator->() const noexcept {
            return stg_.operator->();
        }

};





#include "intrusive_ptr.tpp"



