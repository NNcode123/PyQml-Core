#include "../tensor.hpp"
#include "itr.hpp"

using namespace detail;

template <typename T, typename V, typename Func>

auto binary_ops(const tensor<T> &a, const tensor<V> &b, Func op)
{

    using R = decltype(op(std::declval<T>(), std::declval<V>()));
    size_t size_output = 1;
    std::vector<size_t> a_dim = a.dim();
    std::vector<size_t> b_dim = b.dim();
    std::vector<int64_t> a_strides = a.strides();
    std::vector<int64_t> b_strides = b.strides();
    size_t dim_size_min = std::min(a_dim.size(), b_dim.size());
    size_t dim_size_max = std::max(a_dim.size(), b_dim.size());
    std::vector<size_t> res_dim(dim_size_max);
    AxisIter a_itr[dim_size_max];
    AxisIter b_itr[dim_size_max];

    for (int j = 0; j < dim_size_max; ++j)
    {
        a_itr[j].dim = j < a_dim.size() ? a_dim[j] : 1;
        b_itr[j].dim = j < b_dim.size() ? b_dim[j] : 1;
        size_t max_dim = std::max(a_itr[j].dim, b_itr[j].dim);
        size_output *= max_dim;
        res_dim[j] = max_dim;
        if (b_itr[j].dim == a_itr[j].dim)
        {
            b_itr[j].advance = b_strides[j];
            a_itr[j].advance = a_strides[j];
            a_itr[j].reset_val = (a_itr[j].dim - 1) * b_itr[j].advance;
            b_itr[j].reset_val = (b_itr[j].dim - 1) * a_itr[j].advance;
        }
        else if (b_itr[j].dim > 1)
        {
            a_itr[j].dim = b_itr[j].dim;
            b_itr[j].advance = b_strides[j];
            b_itr[j].reset_val = (b_itr[j].dim - 1) * b_itr[j].advance;
        }
        else
        {
            b_itr[j].dim = a_itr[j].dim;
            a_itr[j].advance = a_strides[j];
            a_itr[j].reset_val = (a_itr[j].dim - 1) * a_itr[j].advance;
        }
    }

    std::shared_ptr<R[]> out(new R[size_output], std::default_delete<R[]>());
    R *__restrict out_data = out.get();

    const V *__restrict b_data = b.data();
    const T *__restrict a_data = a.data();
    size_t ind_max = dim_size_max - 1;

    for (size_t i = 0; i < size_output; ++i)
    {

        out_data[i] = op(static_cast<R>(*a_data), static_cast<R>(*b_data));

        getIndex(a_itr, b_itr, 0, ind_max, a_data, b_data);
    }
    // return tensor<T>(std::move(out), res_dim);
    return tensor<R>(out, size_output, res_dim);

    /*
    if (is_contiguous() && b.is_contiguous()){
        for (size_t j = 0; j < size_output; )
    }
        */
}

template <typename T>
template <typename Func>
tensor<T> tensor<T>::binary_op(const tensor<T> &a, const tensor<T> &b, Func op) const
{
    size_t size_output = 1;

    size_t dim_size_min = std::min(a.dim_.size(), b.dim_.size());
    size_t dim_size_max = std::max(a.dim_.size(), b.dim_.size());
    std::vector<size_t> res_dim(dim_size_max);
    AxisIter a_itr[NDIM];
    AxisIter b_itr[NDIM];

    for (int j = 0; j < dim_size_max; ++j)
    {
        a_itr[j].dim = j < a.dim_.size() ? a.dim_[j] : 1;
        b_itr[j].dim = j < b.dim_.size() ? b.dim_[j] : 1;
        
        size_t max_dim = std::max(a_itr[j].dim, b_itr[j].dim);
        if (b_itr[j].dim == a_itr[j].dim)
        {
            b_itr[j].advance = b.strides_[j];
            a_itr[j].advance = a.strides_[j];
            a_itr[j].reset_val = (a_itr[j].dim - 1) * a.strides_[j];
            b_itr[j].reset_val = (b_itr[j].dim - 1) * b.strides_[j];
        }
        else if (b_itr[j].dim > 1)
        {
            a_itr[j].dim = b_itr[j].dim;
            b_itr[j].advance = b.strides_[j];
            b_itr[j].reset_val = (b_itr[j].dim - 1) * b_itr[j].advance;
        }
        else
        {
            b_itr[j].dim = a_itr[j].dim;
            a_itr[j].advance = a.strides_[j];
            a_itr[j].reset_val = (a_itr[j].dim - 1) * a_itr[j].advance;
        }

        size_output *= max_dim;
        res_dim[j] = max_dim;
    }

    std::shared_ptr<T[]> out(new T[size_output], std::default_delete<T[]>());
    T *__restrict out_data = out.get();

    const T *__restrict b_data = b.data_.get() + b.offset;
    const T *__restrict a_data = a.data_.get() + a.offset;
    size_t ind_max = dim_size_max - 1;

    for (size_t i = 0; i < size_output; ++i)
    {

        out_data[i] = op(*a_data, *b_data);

        getIndex(a_itr, b_itr, 0, ind_max, a_data, b_data);
    }
    // return tensor<T>(std::move(out), res_dim);
    return tensor<T>(out, size_output, res_dim);

    /*
    if (is_contiguous() && b.is_contiguous()){
        for (size_t j = 0; j < size_output; )
    }
        */
}

template <typename T>
tensor<T> tensor<T>::operator+(const tensor<T> &other) const
{
    return binary_op(*this, other, [](const T &u, const T &v)
                     { return u + v; });
}

template <typename T>
tensor<T> tensor<T>::operator-(const tensor<T> &other) const
{
    return binary_op(*this, other, [](const T &u, const T &v)
                     { return u - v; });
}

template <typename T>
tensor<T> tensor<T>::operator*(const tensor<T> &other) const
{
    return binary_op(*this, other, [](const T &u, const T &v)
                     { return u * v; });
}

template <typename T>
tensor<T> tensor<T>::operator/(const tensor<T> &other) const
{
    return binary_op(*this, other, [](const T &u, const T &v)
                     { return u / v; });
}
