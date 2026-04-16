#include "../tensor.hpp"
#include "itr.hpp"
#include "set"
#include "unordered_map"
#include <functional>

using namespace detail;

template <typename T, typename V>
struct BroadCast
{
    static constexpr int size = 8;
    AxisIter a_itr[size];
    AxisIter b_itr[size];
    size_t size_output;
    std::vector<size_t> res_dim;
    BroadCast(const tensor<T> &a, const tensor<V> &b, size_t dim_size_max = 0)
    {
        std::vector<size_t> a_dim = a.dim();
        std::vector<size_t> b_dim = b.dim();
        std::vector<int64_t> a_strides = a.strides();
        std::vector<int64_t> b_strides = b.strides();
        if (dim_size_max == 0)
            dim_size_max = std::max(a_dim.size(), b_dim.size());

        res_dim.resize(dim_size_max);

        int a_ind = a_dim.size() - 1;
        int b_ind = b_dim.size() - 1;
        size_output = 1;

        for (int j = dim_size_max - 1; j >= 0; j--)
        {
            auto &cur_a_itr = a_itr[j];
            auto &cur_b_itr = b_itr[j];

            cur_a_itr.dim = (a_ind >= 0) ? a_dim[a_ind] : 1;
            cur_b_itr.dim = (b_ind >= 0) ? b_dim[b_ind] : 1;

            auto &a_d = cur_a_itr.dim;
            auto &b_d = cur_b_itr.dim;

            size_t max_dim = std::max(a_d, b_d);
            size_output *= max_dim;
            res_dim[j] = max_dim;

            if (a_d == b_d)
            {
                cur_a_itr.advance = a_strides[a_ind];
                cur_a_itr.reset_val = (a_d - 1) * cur_a_itr.advance;

                cur_b_itr.advance = b_strides[b_ind];
                cur_b_itr.reset_val = (b_d - 1) * cur_b_itr.advance;
            }
            else if (b_d > 1)
            {
                a_d = b_d;
                cur_b_itr.advance = b_strides[b_ind];
                cur_b_itr.reset_val = (b_d - 1) * cur_b_itr.advance;
            }
            else
            {
                b_d = a_d;
                cur_a_itr.advance = a_strides[a_ind];
                cur_a_itr.reset_val = (a_d - 1) * cur_a_itr.advance;
            }

            a_ind--;
            b_ind--;
        }
    }
};

template <typename T, typename V, typename Func>

auto binary_ops(const tensor<T> &a, const tensor<V> &b, Func op)
{

    using R = decltype(op(std::declval<T>(), std::declval<V>()));

    auto Itr_info = BroadCast<T, V>(a, b);
    auto &a_itr = Itr_info.a_itr;
    auto &b_itr = Itr_info.b_itr;

    auto res_dim = Itr_info.res_dim;
    std::size_t size_output = Itr_info.size_output;
    std::shared_ptr<R[]> out(new R[size_output], std::default_delete<R[]>());
    R *__restrict out_data = out.get();
    const V *__restrict b_data = b.data();
    const T *__restrict a_data = a.data();
    size_t ind_max = res_dim.size() - 1;

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
    auto Itr_info = BroadCast<T, T>(a, b);
    auto &a_itr = Itr_info.a_itr;
    auto &b_itr = Itr_info.b_itr;
    auto res_dim = std::move(Itr_info.res_dim);
    auto size_output = Itr_info.size_output;

    std::shared_ptr<T[]> out(new T[size_output], std::default_delete<T[]>());

    const T *__restrict b_data = b.data_.get() + b.offset;
    const T *__restrict a_data = a.data_.get() + a.offset;
    T *__restrict out_data = out.get();

    size_t ind_max = res_dim.size() - 1;

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

template <typename T>
[[nodiscard]] tensor<T> tensor<T>::tensor_prod(const tensor<T> &other) const
{
    std::vector<size_t> new_dim = dim_;
    new_dim.insert(new_dim.end(), other.dim_.begin(), other.dim_.end());

    const auto A = copy();
    const auto B = other.copy();

    const size_t this_size = size();
    const size_t other_size = other.size();
    const T *__restrict A_data = A.data_.get();
    const T *__restrict B_data = B.data_.get();

    std::shared_ptr<T[]> new_data(new T[this_size * other_size], std::default_delete<T[]>());
    const T *__restrict new_data_ = new_data.get();

    for (int64_t i = 0; i < this_size; ++i)
    {

        const T a_val = A_data[i];

        for (int64_t j = 0; j < other_size; ++j)
        {
            new_data_[j] = a_val * B_data[j];
        }
        new_data_ += other_size;
    }

    return tensor<T>(std::move(new_data), new_dim);
}

template <typename U, typename V>

auto matmul(const tensor<U> &tens_1, const tensor<V> &tens_2)
{
    return einsum(tens_1, tens_2, {tens_1.size() - 2, tens_1.size() - 1}, {tens_2.size() - 2, tens_2.size() - 1});
}

template <typename U, typename V>

auto einsum(const tensor<U> &tens_1, const tensor<V> &tens_2, std::vector<int> Axes_a, std::vector<int> Axes_b /*, std::vector<int> shared_axes*/)
{
    AxisIter a_cont[tensor<U>::NDIM];
    AxisIter b_cont[tensor<U>::NDIM];
    AxisIter a_free[tensor<U>::NDIM];
    AxisIter b_free[tensor<U>::NDIM];

    std::vector<bool> is_contract_a(tensor<U>::NDIM, false);
    std::vector<bool> is_contract_b(tensor<U>::NDIM, false);
    std::unordered_map<int, int> a_cont_pos;
    std::unordered_map<int, int> b_cont_pos;
    std::vector<size_t> res_dim;
    int index = 0;
    int index_1 = 0;
    for (auto x : Axes_a)
    {
        is_contract_a[x] = true;
        a_cont_pos[x] = index;
        ++index;
    }
    for (auto x : Axes_b)
    {
        is_contract_b[x] = true;
        b_cont_pos[x] = index_1;
        ++index_1;
    }

    auto a_old_dim = tens_1.dim();
    auto a_old_strides = tens_1.strides();
    auto b_old_dim = tens_2.dim();
    auto b_old_strides = tens_2.strides();

    size_t inner_size = 1;
    size_t size_output = 1;
    size_t a_size = Axes_a.size() - 1;
    size_t b_size = Axes_b.size() - 1;
    size_t a_free_size = a_old_dim.size() - Axes_a.size() - 1;
    size_t b_free_size = b_old_dim.size() - Axes_b.size() - 1;
    size_t a_tens_size = 1;
    size_t b_tens_size = 1;
    size_t a_free_index = 0;
    size_t b_free_index = 0;

    for (int i = 0; i < a_old_dim.size(); i++)
    {
        if (!is_contract_a[i])
        {
            auto &cur_a_free = a_free[a_free_index++];
            cur_a_free.dim = a_old_dim[i];
            cur_a_free.advance = a_old_strides[i];
            cur_a_free.reset_val = (cur_a_free.dim - 1) * cur_a_free.advance;
            size_output *= cur_a_free.dim;
            a_tens_size *= cur_a_free.dim;
            res_dim.push_back(cur_a_free.dim);
        }
        else
        {
            auto &cur_itr = a_cont[a_cont_pos[i]];
            cur_itr.dim = a_old_dim[i];
            cur_itr.advance = a_old_strides[i];
            cur_itr.reset_val = (cur_itr.dim - 1) * cur_itr.advance;
            inner_size *= cur_itr.dim;
        }
    }
    for (int i = 0; i < b_old_dim.size(); i++)
    {
        if (!is_contract_b[i])
        {
            auto &cur_b_free = b_free[b_free_index++];
            cur_b_free.dim = b_old_dim[i];
            cur_b_free.advance = b_old_strides[i];
            cur_b_free.reset_val = (cur_b_free.dim - 1) * cur_b_free.advance;
            size_output *= cur_b_free.dim;
            b_tens_size *= cur_b_free.dim;
            res_dim.push_back(cur_b_free.dim);
        }
        else
        {
            auto &cur_itr = b_cont[b_cont_pos[i]];
            cur_itr.dim = b_old_dim[i];
            cur_itr.advance = b_old_strides[i];
            cur_itr.reset_val = (cur_itr.dim - 1) * cur_itr.advance;
        }
    }

    size_t ind_max = res_dim.size() - 1;

    using T = decltype(std::multiplies<>()(std::declval<U>(), std::declval<V>()));

    std::shared_ptr<T[]> out(new T[size_output], std::default_delete<T[]>());

    const U *__restrict a_data = tens_1.data();

    const V *__restrict b_data = tens_2.data();
    T *__restrict out_data = out.get();

    for (size_t a_free_ind = 0; a_free_ind < a_tens_size; ++a_free_ind)
    {
        for (size_t b_free_ind = 0; b_free_ind < b_tens_size; ++b_free_ind)
        {
            T val = 0;
            for (size_t i_ind = 0; i_ind < inner_size; ++i_ind)
            {
                val += (*a_data) * (*b_data);
                getIndex(a_cont, 0, a_size, a_data);
                getIndex(b_cont, 0, b_size, b_data);
            }
            *out_data++ = val;
            getIndex(b_free, 0, b_free_size, b_data);
        }

        getIndex(a_free, 0, a_free_size, a_data);
    }

    /*
    tens_1.dim = a_old_dim;
    tens_1.strides = a_old_stirdes;
    tens_2.dim = b_old_strides;
    tens_2.strides = b_old_dim;
    */

    return tensor<T>(out, size_output, res_dim);
}