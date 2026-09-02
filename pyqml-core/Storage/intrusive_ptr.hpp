
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

    int32_t ref_count() const {
        return ref_count.load(std::memory_order_acq_rel);
    }

    protected:

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

    void* get(){return buffer;}

    size_t bytes(){return bytes_size;}

    template <typename U> 

    U* get_typed(){return static_cast<U*>(buffer);}

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

    template <typename U>
    friend class pyq_intrusive_ptr;

    public:

        pyq_intrusive_ptr(T* stg)  {
            storage = stg;
            retain();
        }

        pyq_intrusive_ptr(decltype(nullptr)): storage(nullptr) {}

        template <typename U>
        pyq_intrusive_ptr(const pyq_intrusive_ptr<U>& ptr) {
            storage = ptr.storage;
            retain();
        }


        template<typename... Args>
        requires (!std::same_as<std::remove_cvref_t<Args>, pyq_intrusive_ptr> && ...) /*&& std::constructible_from<T, Args...>*/
        
        explicit pyq_intrusive_ptr(Args&& ... args){
            storage = new T(std::forward<Args>(args) ...);
            retain();
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

        const T* const_storage_ptr() const noexcept {return storage;}

        void retain() noexcept {
            if (storage) storage->incref();
        }

        void reset_ref() noexcept {
            if (storage) storage->decref();
        }

        void reset() noexcept {
            reset_ref();
            storage = nullptr;
        }

        pyq_intrusive_ptr(const pyq_intrusive_ptr&);

        pyq_intrusive_ptr(pyq_intrusive_ptr&&);

        pyq_intrusive_ptr& operator=(pyq_intrusive_ptr&&) noexcept;

        pyq_intrusive_ptr& operator=(const pyq_intrusive_ptr& ) noexcept;

        template <typename U>
        pyq_intrusive_ptr& operator=(const pyq_intrusive_ptr<U>&) noexcept;

        

        ~pyq_intrusive_ptr(); 
        
    
};


class StorageRef{
    
    
    pyq_intrusive_ptr<Storage> stg_;


    public:
        StorageRef() = default;

        StorageRef(pyq_intrusive_ptr<Storage> ptr): stg_(std::move(ptr)) {}

        void* void_data() const noexcept {
            return stg_ ? stg_->get() : nullptr;
        }


        template <typename T>
        T* data_ptr() const noexcept {
             return stg_ ? stg_->template get_typed<T>() : nullptr;
        }

        template <typename... Args>
        StorageRef(Args&&... args): stg_(std::forward<Args>(args)...) {}

        size_t nbytes() const noexcept {
            return stg_ ? stg_->bytes() : 0;
        }

        operator bool() const noexcept{
            return static_cast<bool>(stg_);
        }

    

};





#include "intrusive_ptr.tpp"



