#pragma once
#include <vector>
#include <array>
#include <string>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <variant>
#include <cstring>
#include <memory>
#include "tensor_imp_files/itr.hpp"
#include "tensor_imp_files/tensor_metadata_util.hpp"
#include "thread/parallel.cpp"
#include "tensor_imp_files/cuda_alloc.cu"
using namespace detail;
using namespace parallel_sync;

using size_t = std::size_t;

struct Index
{
    int64_t val;

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
};

struct SlicePlan
{
    std::vector<size_t> dim;
    std::vector<int64_t> strides;
    size_t start_index;
    size_t size;
};

using AxisType = std::variant<Index, Range, Slice>;
using AxisView = std::variant<Index, Slice>;

template <typename T>
class tensor
{

    std::shared_ptr<T[]> data_;
    size_t t_size;
    std::vector<size_t> dim_;
    std::vector<int64_t> strides_;
    std::size_t offset;

public:
    using Data_type = T;
    static constexpr size_t NDIM = 8;
    // This default constructor creates an empty tensor shell that can be populated later
    // when a view or a concrete buffer is attached.
    explicit tensor() : data_(nullptr), dim_({}) {}

    // This constructor builds a tensor from a temporary vector and an explicit shape,
    // making it convenient to materialize a contiguous tensor from host-side data.
    tensor(std::vector<T> &&data, const std::vector<size_t> &dim) : t_size(data.size()), dim_(dim), offset(0)

    {
        data_ = std::shared_ptr<T[]>(new T[t_size]);
        // data_ = cuda_alloc<T>(data.size());
        std::move(data.begin(), data.end(), data_.get());
        fill_size_vec(dim, strides_);
    }

    // This constructor creates a tensor from a shared buffer and a known size, which is useful
    // for compact representations of already allocated storage and for reshape-like operations.
    tensor(std::shared_ptr<T[]> data_s, size_t size_val, const std::vector<size_t> &dim) : data_(data_s), t_size(size_val), dim_(dim), offset(0)
    {

        fill_size_vec(dim, strides_);
    }

    // This constructor wraps an existing shared buffer with explicit logical dimensions and
    // strides so the tensor can represent either contiguous data or a view over another buffer.
    static tensor<T> tensor_view(std::shared_ptr<T[]> buffer, const std::vector<size_t> &dims, const std::vector<int64_t> &strides, size_t offset, size_t t_size);

    [[nodiscard]] size_t size() const
    {
        return t_size;
    }
    [[nodiscard]] std::vector<size_t> dim() const { return dim_; }
    [[nodiscard]] std::vector<int64_t> strides() const { return strides_; }
    [[nodiscard]] size_t ndim() { return dim_.size(); }
    [[nodiscard]] T *data() const { return data_.get() + offset; }
    [[nodiscard]] std::vector<T> data_vector() const { return std::vector<T>(data_.get(), data_.get() + t_size); }
    [[nodiscard]] const std::vector<size_t> &shape() const { return dim_; }
    [[nodiscard]] size_t ofst() const { return offset; }
    [[nodiscard]] tensor<T> reshape(const std::vector<size_t> &newshape) const
    {
        if (is_contiguous())
        {
            return tensor<T>(data_, t_size, newshape);
        }
        tensor<T> cop_tens = copy();
        return tensor<T>(cop_tens.data_, cop_tens.t_size, newshape);
    }
    [[nodiscard]] std::shared_ptr<T[]> owner() const { return data_; }

    [[nodiscard]] T &at(const std::vector<int> &pos)
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
        std::vector<int> pos = {indices...};
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
    tensor<T> reduce_op(int a, ElmOp &&op) const;
    // unary ops
    template <typename ElmOp>
    T reduce_op(ElmOp &&op);

    template <typename ElmOp>
    tensor<T> apply_op(ElmOp &&op);

    tensor<T> max(int axis) const;
    tensor<T> min(int axis) const;
    tensor<T> sum(int axis) const;
    tensor<T> prod(int axis) const;
    tensor<T> mean(int axis) const;
    T sum() const;
    T mean() const;
    T max() const;
    tensor<T> sin() const;
    tensor<T> exp() const;
    tensor<T> cos() const;
    tensor<T> tan() const;
    tensor<T> cot() const;
    tensor<T> csc() const;
    tensor<T> sec() const;
    tensor<T> sinh() const;
    tensor<T> cosh() const;
    tensor<T> tanh() const;

    template <typename R>
    tensor<R> astype(bool copy = false) const;
    [[nodiscard]] tensor<T> tensor_prod(const tensor<T> &other) const;


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

    tensor<T> &matrixPow(const size_t &val);
    tensor<T> &elemPow(const size_t &val);

    friend std::ostream &operator<<(std::ostream &out, const tensor<T> &tensor)
    {
        printTens(out, tensor, 0);
        return out;
    }
};

#include "tensor_imp_files/slice.tpp"
#include "tensor_imp_files/binary_op.tpp"
#include "tensor_imp_files/unary_op.tpp"
#include "tensor_imp_files/getter.tpp"
#pragma message("getter.tpp included successfully")
