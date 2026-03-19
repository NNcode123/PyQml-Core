#pragma once
#include <vector>
#include <array>
// #include <complex.h>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <variant>
#include <cstring>
#include <memory>
#include "tensor_imp_files/itr.hpp"
using namespace detail;

using size_t = std::size_t;

struct Index
{
    int64_t val;
    inline Index(int64_t vals) : val(vals) {}
};

struct Range
{
    std::vector<int64_t> indices;
    std::vector<int64_t> differentials;
    Range(std::vector<int64_t> indc) : indices(indc)
    {
        differentials.resize(indices.size() - 1);
    }
};

struct Slice
{
    int64_t start;
    int64_t step;
    int64_t end;
    Slice(int64_t s, int64_t e, int64_t st = 1) : start(s), end(e), step(st) {}
};

struct SlicePlan
{
    std::vector<size_t> dim;
    std::vector<size_t> counts;
    std::vector<int64_t> strides;
    size_t start_index;
    size_t size;

    SlicePlan(std::vector<size_t> d, std::vector<size_t> cts, size_t s_index, size_t s) : dim(d), counts(cts), start_index(s_index), size(s) {}
    SlicePlan(std::vector<size_t> d, size_t s_idx, size_t s) : dim(d), start_index(s_idx), size(s) {}
    SlicePlan(std::vector<size_t> d, std::vector<int64_t> stride, size_t s_idx, size_t s) : dim(d), strides(stride), start_index(s_idx), size(s) {}
    SlicePlan(std::vector<size_t> &&d, std::vector<int64_t> &&s, size_t s_idx) : dim(std::move(d)), strides(std::move(s)), start_index(s_idx) {}
};

using AxisType = std::variant<Index, Range, Slice>;
using AxisView = std::variant<Index, Slice>;

template <typename T>
class tensor
{

    static constexpr size_t NDIM = 8;
    // size_t dim_stack[NDIM];
    // int64_t strides_stack[NDIM];
    std::shared_ptr<T[]> data_;
    size_t t_size;
    std::vector<size_t> dim_;
    std::vector<int64_t> strides_;
    std::size_t offset;

public:
    using Data_type = T;
    explicit tensor() : data_(nullptr), dim_({}) {}

    explicit tensor(std::vector<T> &&data, const std::vector<size_t> &dim) : t_size(data.size()), dim_(dim)

    {
        data_ = std::shared_ptr<T[]>(new T[t_size]);
        std::move(data.begin(), data.end(), data_.get());
        strides_.resize(dim_.size());
        offset = 0;
        for (size_t i = 0; i < strides_.size(); ++i)
        {
            int64_t cur_stride = 1;
            for (size_t j = i + 1; j < strides_.size(); ++j)
            {
                cur_stride *= dim_[j];
            }
            strides_[i] = cur_stride;
        }
    }

    explicit tensor(std::shared_ptr<T[]> data_s, size_t size_val, const std::vector<size_t> &dim) : data_(data_s), t_size(size_val), dim_(dim)
    {
        strides_.resize(dim_.size());
        offset = 0;
        for (size_t i = 0; i < strides_.size(); ++i)
        {
            int64_t cur_stride = 1;
            for (size_t j = i + 1; j < strides_.size(); ++j)
            {
                cur_stride *= dim_[j];
            }
            strides_[i] = cur_stride;
        }
    }

    [[nodiscard]] size_t size() const
    {
        size_t size = 1;
        for (const auto &val : dim_)
            size *= val;
        return size;
    }
    [[nodiscard]] std::vector<size_t> dim() const { return dim_; }
    [[nodiscard]] std::vector<int64_t> strides() const { return strides_; }
    [[nodiscard]] size_t ndim() { return dim_.size(); }
    [[nodiscard]] T *data() const { return data_.get(); }
    [[nodiscard]] std::vector<T> data_vector() const { return std::vector<T>(data_.get(), data_.get() + t_size); }
    [[nodiscard]] const std::vector<size_t> &shape() const { return dim_; }
    [[nodiscard]] size_t ofst() const { return offset; }
    [[nodiscard]] tensor<T> reshape(const std::vector<size_t> &newshape) const { return tensor<T>(data_, newshape); }

    [[nodiscard]] T at(const std::vector<size_t> &pos) const
    {
        size_t index, s_index;
        s_index = 0;
        index = offset;
        for (const auto &stride : strides_)
        {
            index += pos[s_index] * stride;
            s_index++;
        }
        return (data_)[index];
    }
    template <typename... Indices>
    [[nodiscard]] T operator()(Indices... indices) const
    {
        std::vector<size_t> pos = {indices...};
        return at(pos);
    }

    std::pair<SlicePlan, std::vector<AxisIter>> analyze_slices(const AxisType *inds, const size_t inds_size);
    SlicePlan analyze_slices(const AxisView *inds, size_t inds_size);
    template <typename... Slices>
    [[nodiscard]] tensor<T> slice(const Slices &...indices);
    template <typename... Slices>
    [[nodiscard]] tensor<T> slice_view(const Slices &...indices);
    std::pair<size_t, size_t> collapse_size() const;
    bool is_contiguous() const;
    [[nodiscard]] tensor<T> copy() const;
    template <typename Func>
    tensor<T> binary_op(const tensor<T> &a, const tensor<T> &b, Func op) const;
    tensor<T> operator+(const tensor<T> &b) const;
    tensor<T> operator-(const tensor<T> &b) const;
    tensor<T> operator*(const tensor<T> &b) const;
    tensor<T> operator/(const tensor<T> &b) const;
    template <typename V, typename R>
    tensor<R> operator+(const tensor<V> &other);
    template <typename ElmOp>
    tensor<T> reduce_op(int a, ElmOp op);
    template <typename ElmOp>
    tensor<T> reduce_op(ElmOp op);
    tensor<T> argmax(int u);
    tensor<T> sin() const;
    tensor<T> cos() const;
    tensor<T> exp() const;

    /*
    tensor<double> power(const double &a)
    {
        std::vector<T> new_data_ = *data_;
        for (auto &elm : new_data_)
        {
            elm = static_cast<T>(std::pow(elm, a));
        }
        return tensor<T>(new_data_, dim_);
    }
        */

    [[nodiscard]] tensor<T> tensor_prod(const tensor<T> &other) const
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

    tensor<T> &matrixPow(const size_t &val);
    tensor<T> &elemPow(const size_t &val);

    friend std::ostream &operator<<(std::ostream &out, const tensor<T> &tensor)
    {
        printTens(out, tensor, tensor.offset);
        return out;
    }
};

#include "tensor_imp_files/slice.tpp"
#include "tensor_imp_files/binary_op.tpp"
#include "tensor_imp_files/unary_op.tpp"
#include "tensor_imp_files/getter.tpp"