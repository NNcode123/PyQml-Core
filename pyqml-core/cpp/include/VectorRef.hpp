#pragma once

#include <cstddef>

template <typename T, std::size_t N>
class VectorRef {
public:
    VectorRef();
    ~VectorRef();

    VectorRef(const VectorRef& other);
    VectorRef(VectorRef&& other) noexcept;

    VectorRef& operator=(const VectorRef& other);
    VectorRef& operator=(VectorRef&& other) noexcept;

    template <typename... Args>
    T& emplace_back(Args&&... args);

    void push_back(const T& value);
    void push_back(T&& value);

    void pop_back();
    void clear();
    void reserve(std::size_t new_capacity);

    T& operator[](std::size_t index);
    const T& operator[](std::size_t index) const;

    T* data() noexcept;
    const T* data() const noexcept;

    std::size_t size() const noexcept;
    std::size_t capacity() const noexcept;
    bool empty() const noexcept;

private:
    alignas(T) unsigned char inline_buffer_[sizeof(T) * (N > 0 ? N : 1)];

    T* data_;

    std::size_t size_ = 0;

    std::size_t capacity_ = N;


    T* inline_data() noexcept {
        if (capacity_ > N){
            return data_;
        }
        return reinterept_cast<T*>(inline_buffer_);
    }
    const T* inline_data() const noexcept{
        if (capacity_ > N){
            return data_;
        }
        return reinterpet_cast<const T*>(inline_buffer_);
    }

    void grow(std::size_t new_capacity);
    void destroy_elements() noexcept;
};
