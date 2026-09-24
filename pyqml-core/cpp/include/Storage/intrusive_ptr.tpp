#include "intrusive_ptr.hpp"


template <typename T>

pyq_intrusive_ptr<T>& pyq_intrusive_ptr<T>::operator=(pyq_intrusive_ptr<T>&& other) noexcept{
    if (this != &other){
        storage = other.storage;
        other.storage = nullptr;
        
    }
    return *this;

}

template <typename T>

pyq_intrusive_ptr<T>::pyq_intrusive_ptr(const pyq_intrusive_ptr<T>& other): storage(other.storage){
    retain();
}


template <typename T>

pyq_intrusive_ptr<T>& pyq_intrusive_ptr<T>::operator=(const pyq_intrusive_ptr<T>& other ) noexcept {

    return this->template operator=<T>(other);

}


template<typename T>
pyq_intrusive_ptr<T>::pyq_intrusive_ptr(pyq_intrusive_ptr<T>&& other): storage(other.storage){
    other.storage = nullptr;

}


template <typename T>
template <typename U>
pyq_intrusive_ptr<T>& pyq_intrusive_ptr<T>::operator=(const pyq_intrusive_ptr<U>& other) noexcept{

    if constexpr (std::is_same_v<T,U>){
        if (this == &other){
            return *this;
        }
    }
    
    auto temp = other; 
    reset_ref();
    storage = temp.storage;
    retain();
    
   
    return *this;
}



template <typename T>
pyq_intrusive_ptr<T>::~pyq_intrusive_ptr(){
    reset_ref();
}


template <typename T, typename... Args>
pyq_intrusive_ptr<T> make_intrusive(Args&&... args){
            return pyq_intrusive_ptr<T>(std::forward<Args>(args)...);
}


template <typename T>
pyq_intrusive_ptr<T> make_intrusive(void *p, void(*dtor)(void* u), size_t size){
    return pyq_intrusive_ptr<T>(new T(p,dtor, size));
}

template <typename T, typename buff_type>
pyq_intrusive_ptr<T> make_intrusive(buff_type* p, size_t size){
    return pyq_intrusive_ptr<T>(new T(p, size));
}

