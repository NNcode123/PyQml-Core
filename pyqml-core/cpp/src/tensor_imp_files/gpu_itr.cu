#include <cuda_runtime.h>
#include <cstddef>
#include "iostream"

using namespace std;

namespace gpu_itr
{
    struct TensorItr
    {
        int64_t *fancy;
        int64_t start;
        int64_t step;
    };
    template <typename T>
    struct TensorInfo
    {
        T *buffer;
        size_t *dim;
        int64_t *strides;
        size_t ndim;
    };

    template <typename U, typename V>
    struct uniTensorInfo
    {
        U *a;
        V *b;
        size_t *dim;
        int64_t *a_strides;
        int64_t *b_strides;
        size_t ndim;
    };

    template <typename U, typename V, typename R, typename Op>
    __global__ void binary_advance(const uniTensorInfo<U, V> &tens, const std::size_t &size, Op &&opy, R *buffer)
    {
        size_t pos = blockIdx.x * blockDim.x + threadIdx.x;
        size_t tens_1_ndim = tens.ndim - 1;

        int64_t t1_index = 0;
        int64_t t2_index = 0;
        if (pos < size)
        {
            size_t expect = 1;
            for (size_t ind = tens_1_ndim; ind >= 0, --ind)
            {
                t1_index += (pos % tens.dim[ind]) * tens.a_strides[ind];
                t2_index += (pos % tens.dim[ind]) * tens.b_strides[ind];
                pos /= dim[ind];
            }
            buffer[pos] = opy(a[t_index], b[t2_index]);
        }
    }

    template <typename T>
    __global__ void copy(const TensorInfo<T> &info, const std::size_t &size, T *out_buffer)
    {
        size_t pos = blockIdx.x * blockDim.x + threadIdx.x;
        size_t info_ndim = info.ndim - 1;
        int64_t buf_pos = 0;
        if (pos < size)
        {
            for (size_t ind = info_ndim; ind >= 0; --ind)
            {
                buf_pos += (pos % info.dim[ind]) * tens.a_strides[ind];
                pos /= dim[ind];
            }
            out_buffer[pos] = info.buffer[buf_pos];
        }
    }

    __global__ void binary_advance()
    {
    }

    __global__ void matmulkernel()
    {
    }

}