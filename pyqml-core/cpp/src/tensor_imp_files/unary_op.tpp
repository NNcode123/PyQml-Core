#include "../tensor.hpp"

template <typename T>
template <typename ElmOp>

tensor<T> tensor<T>::reduce_op(int a, ElmOp op)
{
    AxisIter itr[NDIM];
    size_t size_output = 1;
    size_t inner_size = dim_[a];
    size_t inner_ind = dim_.size() - 2;
    std::vector<size_t> new_dim = dim_;
    new_dim.erase(new_dim.begin() + a);

    for (size_t i = 0; i < dim_.size() - 1; ++i)
    {
        int ind = i;
        if (i >= a)
        {
            ++ind;
        }
        itr[i].advance = strides_[ind];
        itr[i].reset_val = (dim_[ind] - 1) * strides_[ind];
        itr[i].dim = dim_[ind];
        size_output *= dim_[ind];
    }

    std::shared_ptr<T[]> out(new T[size_output], std::default_delete<T[]>());
    T *__restrict out_data = out.get();
    const T *__restrict data__ = data_.get() + offset;

    int64_t inner_stride = strides_[a];
    int64_t reset_val = inner_stride * inner_size;
    T res_val;
    for (size_t j = 0; j < size_output; ++j)
    {
        res_val = *data__;
        data__ += inner_stride;
        for (size_t i = 1; i < inner_size; ++i)
        {
            res_val = op(res_val, *data__);

            data__ += inner_stride;
        }

        *out_data++ = res_val;
        data__ -= reset_val;
        getIndex(itr, 0, inner_ind, data__);
    }

    return tensor<T>(out, size_output, new_dim);
}

template <typename T>
tensor<T> tensor<T>::argmax(int u)
{
    return reduce_op(u, [](const T &a, const T &b)
                     { return std::max(a, b); });
}
