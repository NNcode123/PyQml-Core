
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
requires IS_REF_COUNTED<T>
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

        template <typename U> 
        U* get() const {
            return storage->template get_typed<U>();
        }



        void* get_void() const{
            if (std::is_same_v<Storage, T>){
                return storage->get();
            }
            else{
                throw std::runtime_error("Underlying buffer is not of type Storage");
            }
            
        }

        pyq_intrusive_ptr(): storage(nullptr) {}

        T* storage_ptr() const noexcept {return storage;}

        pyq_intrusive_ptr(const pyq_intrusive_ptr&);

        pyq_intrusive_ptr(pyq_intrusive_ptr&&);

        pyq_intrusive_ptr& operator=(pyq_intrusive_ptr&&) noexcept;

        pyq_intrusive_ptr& operator=(const pyq_intrusive_ptr& ) noexcept;

        ~pyq_intrusive_ptr(); 
        
    
};





#include "intrusive_ptr.tpp"



