
#include "iostream"
#include <cstddef>
#include <memory>
// #include <cuda_runtime.h>

/*
template <typename T>
std::shared_ptr<T[]> cuda_alloc(size_t size)
{
    T *raw;
    cudaMalloc(&raw, size * sizeof(T));
    return std::shared_ptr<T[]>(raw, [](T *ptr)
                                { cudaFree(ptr); });
}
*/