#include "../tensor.hpp"
#include <type_traits>
#include <cmath>

// This reduction helper collapses one axis of the tensor by applying the supplied operator
// repeatedly across the selected dimension and storing the result in a reduced output tensor.
template <typename T>
template <typename ElmOp>

tensor<T> tensor<T>::reduce_op(int a, ElmOp &&op) const
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

    pyq_intrusive_ptr<Storage> out(new T[size_output], size_output);
    T *__restrict out_data = out.template get<T>();
    const T *__restrict data__ = data_.template get<T>() + offset;

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
        advance(itr, 0, inner_ind, data__);
    }

    return tensor<T>(out, size_output, new_dim);
}

// This scalar-reduction helper walks the tensor and collapses all elements into a single
// value by repeatedly applying the supplied operator, which is used for reductions such as sum and mean.
template <typename T>
template <typename ElmOp>
T tensor<T>::reduce_op(ElmOp &&op)
{
    T res = 0;
    const T *__restrict data = data_.template get<T>() + offset;
    if (is_contiguous())
    {
        for (size_t i = 0; i < t_size; ++i)
        {
            res = op(res, *data++);
        }
    }
    else
    {
        AxisIter itr[tensor<T>::NDIM];
        size_t out_ind = dim_.size() - 1;
        for (size_t j = 0; j < dim_.size(); ++j)
        {
            itr[j].advance = strides_[j];
            itr[j].reset_val = strides_[j] * (dim_[j] - 1);
            itr[j].dim = dim_[j];
        }
        for (size_t i = 0; i < t_size; ++i)
        {
            res = op(res, *data);
            advance(itr, 0, out_ind, data);
        }
    }
    return res;
}

// This helper applies a unary function to every element in the tensor and writes the results
// into a new tensor with the same logical shape.
template <typename T>
template <typename ElmOp>
tensor<T> tensor<T>::apply_op(ElmOp &&op)
{
    T res = 0;
    std::vector<size_t> n_shp = dim_;
    size_t size = t_size;
    pyq_intrusive_ptr<Storage> out(new T[size], size);
    const T *__restrict out_data = out.template get<T>() + offset;
    const T *__restrict data = data_.template get<T>() + offset;
    if (is_contiguous())
    {
        for (size_t i = 0; i < t_size; ++i)
        {
            (*out_data++) = op(*data++);
        }
    }
    else
    {
        AxisIter itr[tensor<T>::NDIM];
        size_t out_ind = dim_.size() - 1;
        for (size_t j = 0; j < dim_.size(); ++j)
        {
            itr[j].advance = strides_[j];
            itr[j].reset_val = strides_[j] * (dim_[j] - 1);
            itr[j].dim = dim_[j];
        }
        for (size_t i = 0; i < t_size; ++i)
        {
            *out_data++ = op(*data);
            advance(itr, 0, out_ind, data);
        }
    }
    return tensor<T>(out, size, n_shp);
}

// This method applies the exponential function to each element and returns a new tensor
// with the same shape as the source tensor.
template <typename T>
tensor<T> tensor<T>::exp() const
{
    return apply_op([](const T &val)
                    { return exp(val); });
}

// This method reduces the tensor along the requested axis by taking the elementwise maximum
// across that dimension and returns the reduced tensor as a new object.
template <typename T>
tensor<T> tensor<T>::max(int u) const
{
    return reduce_op(u, [](const T &a, const T &b)
                     { return std::max(a, b); });
}

// This method reduces the tensor along the requested axis by taking the elementwise minimum
// across that dimension and returns the reduced tensor as a new object.
template <typename T>

tensor<T> tensor<T>::min(int u) const
{
    return reduce_op(u, [](const T &a, const T &b)
                     { return std::min(a, b); });
}

// This method reduces the tensor along the requested axis by summing the values across that
// dimension and returns the reduced tensor as a new object.
template <typename T>

tensor<T> tensor<T>::sum(int u) const
{
    return reduce_op(u, [](const T &a, const T &b)
                     { return a + b; });
}

// This method converts the tensor values into a new scalar type and stores them in a new
// tensor object, which is useful when callers need a different numeric representation.
template <typename T>

template <typename R>

tensor<R> tensor<T>::astype(bool copy) const
{
    /*
    if (std::is_same_v<std::decay_t<T>, std::decay_t<R>> && !copy)
    {
        return tensor<R>(data_, dim_, strides_, offset, t_size);
    }
        */
    AxisIter itr[NDIM];
    for (size_t j = 0; j < dim_.size(); ++j)
    {
        itr[j].advance = strides_[j];
        itr[j].reset_val = strides_[j] * (dim_[j] - 1);
        itr[j].dim = dim_[j];
    }

    pyq_intrusive_ptr<Storage> new_ptr(new R[t_size], t_size);
    R *__restrict raw_new = new_ptr.template get<R>();
    const T *__restrict cur_ptr = data_.template get<T>() + offset;
    size_t ind_dim = dim_.size() - 1;

    if (is_contiguous())
    {
        for (size_t k = 0; k < t_size; ++k)
        {
            *raw_new++ = static_cast<R>(*cur_ptr++);
        }
    }

    else
    {
        for (size_t k = 0; k < t_size; ++k)
        {

            *raw_new++ = static_cast<R>(*cur_ptr);

            advance(itr, 0, ind_dim, cur_ptr);
        }
    }
    return tensor<R>(new_ptr, t_size, dim_);
}

// This factory creates an uninitialized tensor with the requested shape, which is useful for
// allocating storage before filling it through later logic or host-side initialization.
template <typename T>

tensor<T> empty(const std::vector<size_t> &shape)
{
    size_t size = 1;
    for (auto &val : shape)
    {
        size *= val;
    }
    pyq_intrusive_ptr<Storage> data(new T[size], size);
    return tensor<T>(data, size, shape);
}
