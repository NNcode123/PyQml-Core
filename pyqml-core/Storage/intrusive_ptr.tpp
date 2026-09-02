#include "intrusive_ptr.hpp"


template <typename T>

pyq_intrusive_ptr<T>& pyq_intrusive_ptr<T>::operator=(pyq_intrusive_ptr<T>&& other) noexcept{
    if (this != &other){
        reset_ref();
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

    if (this != &other){
        other.retain();
        reset_ref();
        storage = other.storage;
        
    }
    return *this;

}




template<typename T>

pyq_intrusive_ptr<T>::pyq_intrusive_ptr(pyq_intrusive_ptr<T>&& other): storage(other.storage){
    other.storage = nullptr;

}


template <typename T>
template <typename U>
pyq_intrusive_ptr<T>& pyq_intrusive_ptr<T>::operator=(const pyq_intrusive_ptr<U>& other) noexcept{
    if (&other != this){
        other.retain();
        reset_ref();
    }
    storage = other.storage;
    return *this;
}



template <typename T>

pyq_intrusive_ptr<T>::~pyq_intrusive_ptr(){
    reset_ref();
}


template <typename T>

pyq_intrusive_ptr<T> make_intrusive(void *p, void(*dtor)(void* u), size_t size){
    return pyq_intrusive_ptr<T>(new T(p,dtor, size));
}

template <typename T, typename buff_type>

pyq_intrusive_ptr<T> make_intrusive(buff_type* p, size_t size){
    return pyq_intrusive_ptr<T>(new T(p, size));
}

