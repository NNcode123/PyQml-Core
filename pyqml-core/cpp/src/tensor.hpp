#include <vector>
#include <array>
#include <complex.h>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <cstring>
#include <memory>
#include <variant>

#pragma once

using size_t = std::size_t;
template <typename T>
struct typing
{
    // To be Defined Later....
};

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

struct AxisIter
{
    std::vector<int64_t> diffs;
    size_t count = 1;
    size_t dim = 0;
    int64_t advance = 0;
    int64_t reset_val = 0;

    int64_t next(size_t index)
    {
        if (!diffs.empty())
        {

            return diffs[index];
        }

        return advance;
    }
};

using AxisType = std::variant<Index, Range, Slice>;
using AxisView = std::variant<Index, Slice>;

template <typename T>
class tensor
{
    static constexpr size_t NDIM = 8;
    // size_t dim_stack[NDIM];
    // int64_t strides_stack[NDIM];
    // std::shared_ptr<std::vector<T>> data_;
    std::shared_ptr<T[]> data_;
    size_t t_size;
    std::vector<size_t> dim_;
    std::vector<int64_t> strides_;
    std::size_t offset;
    std::vector<int64_t> contiguous_strides() const
    {
        std::vector<int64_t> strideX(strides_.size());
        strideX.back() = 1;
        for (int i = strideX.size() - 2; i > 0; i--)
        {
            strideX[i] = dim_[i + 1] * strideX[i + 1];
        }
        return strideX;
    }
    inline int64_t getIndex(std::vector<size_t> &cur_counts, size_t start_dim_index,
                            size_t end_dim_index, int64_t cur_pos) const
    {

        while (dim_[end_dim_index] == cur_counts[end_dim_index])
        {

            cur_pos -= cur_counts[end_dim_index] * strides_[end_dim_index];

            cur_counts[end_dim_index] = 1;

            if (end_dim_index == start_dim_index)
            {
                return cur_pos;
            }

            end_dim_index--;
        }
        cur_counts[end_dim_index]++;
        cur_pos += strides_[end_dim_index];

        return cur_pos;
    }
    inline void getIndex(AxisIter *axis_info, size_t start_dim, size_t end_dim, const T *__restrict &src) const
    {
        while (axis_info[end_dim].count == axis_info[end_dim].dim)
        {
            src -= axis_info[end_dim].reset_val;
            axis_info[end_dim].count = 1;
            if (end_dim == start_dim)
            {
                return;
            }
            --end_dim;
        }
        src += axis_info[end_dim].advance;
        ++axis_info[end_dim].count;
    }
    inline int64_t getIndex(std::vector<AxisIter> &axis_info,
                            const std::vector<size_t> &new_dim,
                            size_t start_index, size_t end_index, int64_t cur_pos) const
    {

        while (axis_info[end_index].dim == axis_info[end_index].count)
        {

            cur_pos -= axis_info[end_index].reset_val;
            axis_info[end_index].count = 1;

            if (end_index == start_index)
            {
                return cur_pos;
            }
            end_index--;
        }

        cur_pos += axis_info[end_index].next(axis_info[end_index].count - 1);
        axis_info[end_index].count++;

        return cur_pos;
    }

    inline void getIndex(size_t *cur_counts, const int64_t *new_strides,
                         size_t start_index, size_t end_index, int64_t &cur_pos) const
    {

        size_t index = end_index;
        while (index >= start_index && cur_counts[index] + 1 == dim_[index])
        {
            cur_pos -= new_strides[index] * cur_counts[index];
            cur_counts[index] = 1;
            if (index == start_index)
                return;
            --index;
        }

        cur_counts[index]++;
        cur_pos += new_strides[index];

        // return cur_pos;
    }

    void getIndex(AxisIter *axes_a, AxisIter *axes_b, size_t start_index, size_t end_index, const T *__restrict &a_data, const T *__restrict &b_data) const
    {

        while (axes_a[end_index].count == axes_a[end_index].dim)
        {
            a_data -= axes_a[end_index].reset_val;
            b_data -= axes_b[end_index].reset_val;
            axes_a[end_index].count = 1;
            axes_b[end_index].count = 1;
            if (end_index == start_index)
            {
                return;
            }
            end_index--;
        }
        axes_a[end_index].count++;
        axes_b[end_index].count++;

        a_data += axes_a[end_index].advance;
        b_data += axes_b[end_index].advance;
    }

public:
    explicit tensor() : data_(nullptr), dim_({}) {}
    /*
    explicit tensor(std::vector<T> &&data, const std::vector<size_t> &dim) : data_(std::make_shared<std::vector<T>>(std::move(data))), dim_(dim)

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
        */
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
    [[nodiscard]] std::vector<size_t> dim() const
    {
        return dim_;
    }
    [[nodiscard]] std::vector<int64_t> strides() const
    {
        return strides_;
    }

    [[nodiscard]] T *data() const // std::vector<T>
    {
        return data_.get();
    }

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
        // return (*data_)[index];
        return (data_)[index];
    }
    template <typename... Indices>
    [[nodiscard]] T operator()(Indices... indices) const
    {
        std::vector<size_t> pos = {indices...};
        return at(pos);
    }

    std::pair<SlicePlan, std::vector<AxisIter>> analyze_slices(const AxisType *inds, const size_t inds_size)

    {
        const size_t ndim = dim_.size();

        std::vector<size_t> new_dim;

        std::vector<AxisIter> axis_iter;
        size_t new_data_size = 1;
        size_t cur_index = offset;

        for (size_t i = 0; i < ndim; ++i)
        {

            AxisIter axis_info;
            if (i < inds_size)
            {
                if (std::holds_alternative<Slice>(inds[i]))
                {
                    const Slice &s0 = std::get<Slice>(inds[i]);
                    Slice s = s0;
                    if (s.start < 0)
                        s.start += dim_[i];
                    if (s.end < 0 && !(s.step < 0 && s.end == -1))
                        s.end += dim_[i];

                    const int64_t span = std::abs(s.end - s.start);
                    const int64_t step = std::abs(s.step);

                    size_t len = span / step + 1;
                    if (span % step == 0)
                    {
                        --len;
                    }

                    new_dim.push_back(len);
                    axis_info.dim = len;
                    new_data_size *= new_dim.back();
                    cur_index += s.start * strides_[i];
                    axis_info.advance = s.step * strides_[i];
                    axis_info.reset_val = axis_info.advance * (len - 1);
                    axis_iter.emplace_back(axis_info);
                }
                else if (std::holds_alternative<Index>(inds[i]))
                {
                    const Index &idx_v = std::get<Index>(inds[i]);
                    Index idx = idx_v;
                    if (idx.val < 0)
                        idx.val += dim_[i];
                    cur_index += idx.val * strides_[i];
                }
                else if (std::holds_alternative<Range>(inds[i]))
                {
                    const Range &range_0 = std::get<Range>(inds[i]);
                    Range range = range_0;
                    for (size_t ind = 0; ind + 1 < range.indices.size(); ++ind)
                    {
                        range.differentials[ind] = (range.indices[ind + 1] - range.indices[ind]) * strides_[i];
                    }
                    axis_info.diffs = std::move(range.differentials);
                    axis_info.reset_val = (range.indices.back() - range.indices.front()) * strides_[i];
                    axis_info.dim = range.indices.size();
                    axis_iter.emplace_back(axis_info);
                    new_dim.push_back(range.indices.size());
                    cur_index += range.indices.front() * strides_[i];
                    new_data_size *= new_dim.back();
                }
            }
            else
            {

                axis_info.advance = strides_[i];
                axis_info.reset_val = (dim_[i] - 1) * strides_[i];
                axis_info.dim = dim_[i];
                axis_iter.emplace_back(axis_info);
                new_dim.push_back(dim_[i]);
                new_data_size *= dim_[i];
            }
        }

        return {SlicePlan(new_dim, cur_index, new_data_size), axis_iter};
    }

    SlicePlan analyze_slices(const AxisView *inds, size_t inds_size)
    {
        const size_t ndim = dim_.size();

        std::vector<size_t> new_dim;
        new_dim.reserve(ndim);
        std::vector<int64_t> new_strides;
        new_strides.reserve(ndim);
        size_t cur_index = offset;

        for (size_t i = 0; i < ndim; ++i)
        {
            if (i < inds_size)
            {
                if (std::holds_alternative<Slice>(inds[i]))
                {
                    const Slice &s0 = std::get<Slice>(inds[i]);
                    Slice s = s0;
                    if (s.start < 0)
                        s.start += dim_[i];
                    if (s.end < 0 && !(s.step < 0 && s.end == -1))
                        s.end += dim_[i];

                    const int64_t span = std::abs(s.end - s.start);
                    const int64_t step = std::abs(s.step);

                    size_t len = span / step + 1;
                    if (span % step == 0)
                    {
                        --len;
                    }

                    new_dim.push_back(len);
                    new_strides.push_back(s.step * strides_[i]);
                    cur_index += s.start * strides_[i];
                }
                else if (std::holds_alternative<Index>(inds[i]))
                {
                    const Index &idx_v = std::get<Index>(inds[i]);
                    Index idx = idx_v;
                    if (idx.val < 0)
                        idx.val += dim_[i];
                    cur_index += idx.val * strides_[i];
                }
            }
            else
            {
                new_dim.push_back(dim_[i]);
                new_strides.push_back(strides_[i]);
            }
        }

        return SlicePlan(std::move(new_dim), std::move(new_strides), cur_index);
    }
    template <typename... Slices>
    [[nodiscard]] tensor<T> slice(const Slices &...indices)
    {
        std::array<AxisType, sizeof...(indices)> inds = {indices...};
        auto [plan, Axis_Iter] = analyze_slices(inds.data(), inds.size());
        std::vector<size_t> new_dim = std::move(plan.dim);
        int64_t cur_index = std::move(plan.start_index);
        size_t new_data_size = std::move(plan.size);
        // std::vector<T> new_data(new_data_size);
        // const T *__restrict data_ptr = data_->data();
        std::shared_ptr<T[]> new_data(new T[new_data_size], std::default_delete<T[]>());
        const T *__restrict data_ptr = new_data.get();

        for (size_t i = 0; i < new_data_size; ++i)
        {
            new_data[i] = data_ptr[cur_index];

            cur_index = getIndex(Axis_Iter, new_dim, 0, new_dim.size() - 1, cur_index);
        }

        // return tensor<T>(std::move(new_data), new_dim);
        return tensor<T>(new_data, new_data_size, new_dim);
    }
    template <typename... Slices>

    [[nodiscard]] tensor<T> slice_view(const Slices &...indices)
    {
        tensor<T> out;
        std::array<AxisView, sizeof...(indices)> inds = {indices...};
        SlicePlan plan = analyze_slices(inds.data(), inds.size());
        out.data_ = data_;
        out.dim_ = std::move(plan.dim);
        out.offset = plan.start_index;
        out.strides_ = std::move(plan.strides);

        return out;
    }

    std::pair<size_t, size_t> collapse_size() const
    {
        if (strides_.back() != 1)
        {
            return {1, 0};
        }
        size_t cur_size = dim_.back();
        size_t offset = dim_.size() - 2;
        for (int i = dim_.size() - 1; i > 0; --i)
        {
            if (strides_[i - 1] == dim_[i] * strides_[i])
            {
                cur_size *= dim_[i - 1];
                offset = i - 1;
            }
            else
            {
                break;
            }
        }
        return {cur_size, offset};
    }
    bool is_contiguous() const
    {
        size_t expect = 1;
        for (int i = dim_.size() - 1; i >= 0; --i)
        {
            if (strides_[i] != expect)
                return false;
            expect *= dim_[i];
        }
        return true;
    }
    [[nodiscard]] tensor<T> copy() const
    {
        if (is_contiguous())
        {
            // return tensor<T>(std::move(*data_), dim_);
            return *this;
        }

        // std::vector<T> new_data;
        // new_data.reserve(size());
        // T *__restrict data = new_data.data();
        // const T *__restrict src = data_->data();
        const T *__restrict src = data_.get();
        std::shared_ptr<T[]> new_data(new T[size()], std::default_delete<T[]>());
        T *__restrict data = new_data.get();

        src += offset;
        std::vector<size_t> counts(dim_.size(), 1);
        const auto [cont_size, cont_offset] = collapse_size();
        size_t radix_size = size() / cont_size;
        int64_t radix_dim_pos = cont_offset;
        int64_t cur_pos = 0;
        int64_t dest_pos = 0;
        AxisIter iter[NDIM];
        for (int i = 0; i < dim_.size(); ++i)
        {
            iter[i].advance = strides_[i];
            iter[i].count = 1;
            iter[i].reset_val = (dim_[i] - 1) * strides_[i];
            iter[i].dim = dim_[i];
        }

        /*

        if (cont_offset == 0)
        {
            if (cont_size > 1)
            {

                const int64_t dstride = strides_[0];
                for (size_t i = 0; i < radix_size; ++i)
                {
                    std::memcpy(data,
                                src,
                                cont_size * sizeof(T));
                    data += cont_size;
                    src += dstride;
                }
            }
            else if (dim_.size() == 1)
            {
                const int64_t dstride = strides_[0];
                for (size_t ind = 0; ind < radix_size; ++ind)
                {
                    *data++ = *src;
                    src += dstride;
                }
            }

            else if (dim_.size() == 2)
            {
                const ssize_t dim0_len = (ssize_t)dim_[0];
                const ssize_t dim1_len = (ssize_t)dim_[1];

                const ssize_t stride0 = (ssize_t)strides_[0];
                const ssize_t stride1 = (ssize_t)strides_[1];

                for (ssize_t ind = 0; ind < dim0_len; ++ind)
                {

                    const T *__restrict src_i = src + ind * stride0;

                    for (ssize_t j = 0; j < dim1_len; ++j)
                    {
                        *data++ = *src_i;
                        src_i += stride1;
                    }
                }
            }

            else if (dim_.size() == 3)
            {
                const ssize_t dim0_len = (ssize_t)dim_[0];
                const ssize_t dim1_len = (ssize_t)dim_[1];
                const ssize_t dim2_len = (ssize_t)dim_[2];

                const ssize_t stride0 = (ssize_t)strides_[0];
                const ssize_t stride1 = (ssize_t)strides_[1];
                const ssize_t stride2 = (ssize_t)strides_[2];

                for (ssize_t ind = 0; ind < dim0_len; ++ind)
                {
                    const T *__restrict src_ind = src + ind * stride0;

                    for (ssize_t j = 0; j < dim1_len; ++j)
                    {
                        const T *__restrict src_ind_j = src_ind + j * stride1;

                        for (ssize_t k = 0; k < dim2_len; ++k)
                        {
                            *data++ = *src_ind_j;
                            src_ind_j += stride2;
                        }
                    }
                }
            }
        }

        else if (cont_offset == 1)
        {
            const ssize_t dim0_len = (ssize_t)dim_[0];
            const ssize_t dim1_len = (ssize_t)dim_[1];

            const ssize_t stride0 = (ssize_t)strides_[0];
            const ssize_t stride1 = (ssize_t)strides_[1];

            for (ssize_t ind = 0; ind < dim0_len; ++ind)
            {
                const T *__restrict src_ind = src + ind * stride0;

                for (ssize_t j = 0; j < dim1_len; ++j)
                {
                    std::memcpy(data,
                                src_ind,
                                cont_size * sizeof(T));
                    data += cont_size;
                    src_ind += stride1;
                }
            }
        }

        else if (cont_offset == 2)
        {
            const ssize_t dim0_len = (ssize_t)dim_[0];
            const ssize_t dim1_len = (ssize_t)dim_[1];
            const ssize_t dim2_len = (ssize_t)dim_[2];

            const ssize_t stride0 = (ssize_t)strides_[0];
            const ssize_t stride1 = (ssize_t)strides_[1];
            const ssize_t stride2 = (ssize_t)strides_[2];

            for (ssize_t ind = 0; ind < dim0_len; ++ind)
            {
                const T *__restrict src_ind = src + ind * stride0;

                for (ssize_t j = 0; j < dim1_len; ++j)
                {
                    const T *__restrict src_ind_j = src_ind_j + j * stride1;
                    for (ssize_t k = 0; k < dim2_len; ++k)
                    {
                        std::memcpy(data, src_ind_j, cont_size * sizeof(T));
                        data += cont_size;
                        src_ind_j += stride2;
                    }
                }
            }
        }
            */

        // else
        //{
        const size_t copy_size = cont_size * sizeof(T);
        for (size_t i = 0; i < radix_size; ++i)
        {

            std::memcpy(data,
                        src,
                        copy_size);

            getIndex(iter, 0, radix_dim_pos, src);
            data += cont_size;
        }
        // }

        // return tensor<T>(std::move(new_data), dim_);
        return tensor<T>(new_data, size(), dim_);
    }
    [[nodiscard]] const std::vector<size_t> &shape() const
    {
        return dim_;
    }
    //[[nodiscard]] const std::vector<int64_t> &strides() const { return strides_; }
    [[nodsicard]] tensor<T> reshape(const std::vector<size_t> &newshape) const
    {
        return tensor<T>(data_, newshape);
    }

    /*
    std::pair<std::vector<int64_t>, std::vector<size_t>> repermuteDims(const tensor<T> &a, const tensor<T> &b)
    {
    }
    */

    tensor<T> operator+(const tensor<T> &b) const
    {
        size_t size_output = 1;

        size_t dim_size_min = std::min(dim_.size(), b.dim_.size());
        size_t dim_size_max = std::max(dim_.size(), b.dim_.size());
        std::vector<size_t> res_dim(dim_size_max);
        AxisIter a_itr[NDIM];
        AxisIter b_itr[NDIM];

        for (int j = 0; j < dim_size_max; ++j)
        {
            a_itr[j].dim = j < dim_.size() ? dim_[j] : 1;
            b_itr[j].dim = j < b.dim_.size() ? b.dim_[j] : 1;

            if (a_itr[j].dim == 1 && b_itr[j].dim > 1)
            {
                a_itr[j].dim = b_itr[j].dim;
                size_output *= b_itr[j].dim;
                res_dim[j] = b_itr[j].dim;
                b_itr[j].advance = b.strides_[j];
                b_itr[j].reset_val = (b_itr[j].dim - 1) * b_itr[j].advance;
            }
            else if (b_itr[j].dim == 1 && a_itr[j].dim > 1)
            {
                b_itr[j].dim = a_itr[j].dim;
                size_output *= a_itr[j].dim;
                res_dim[j] = a_itr[j].dim;
                a_itr[j].advance = strides_[j];
                a_itr[j].reset_val = (a_itr[j].dim - 1) * a_itr[j].advance;
            }
            else if (b_itr[j].dim == a_itr[j].dim)
            {
                b_itr[j].advance = b.strides_[j];
                a_itr[j].advance = strides_[j];
                a_itr[j].reset_val = (a_itr[j].dim - 1) * strides_[j];
                b_itr[j].reset_val = (b_itr[j].dim - 1) * b.strides_[j];
                size_output *= a_itr[j].dim;
                res_dim[j] = a_itr[j].dim;
            }
        }
        // std::vector<T> out(size_output);

        // T *out_data = out.data();
        std::shared_ptr<T[]> out(new T[size_output], std::default_delete<T[]>());
        T *__restrict out_data = out.get();

        const T *__restrict b_data = b.data_.get() + b.offset;
        const T *__restrict a_data = data_.get() + offset;
        int64_t a_index = 0;
        int64_t b_index = 0;
        size_t ind_max = dim_size_max - 1;

        for (size_t i = 0; i < size_output; ++i)
        {

            out_data[i] = *a_data + *b_data;

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
    tensor<T> operator-(const tensor<T> &b) const
    {
        size_t size_output = 1;
        size_t dim_size_min = std::min(dim_.size(), b.dim_.size());
        size_t dim_size_max = std::max(dim_.size(), b.dim_.size());
        std::vector<size_t> res_dim(dim_size_max);
        AxisIter a_itr[NDIM];
        AxisIter b_itr[NDIM];

        for (int j = 0; j < dim_size_max; ++j)
        {
            a_itr[j].dim = j < dim_.size() ? dim_[j] : 1;
            b_itr[j].dim = j < b.dim_.size() ? b.dim_[j] : 1;

            if (a_itr[j].dim == 1 && b_itr[j].dim > 1)
            {
                a_itr[j].dim = b_itr[j].dim;
                size_output *= b_itr[j].dim;
                res_dim[j] = b_itr[j].dim;
                b_itr[j].advance = b.strides_[j];
                b_itr[j].reset_val = (b_itr[j].dim - 1) * b_itr[j].advance;
            }
            else if (b_itr[j].dim == 1 && a_itr[j].dim > 1)
            {
                b_itr[j].dim = a_itr[j].dim;
                size_output *= a_itr[j].dim;
                res_dim[j] = a_itr[j].dim;
                a_itr[j].advance = strides_[j];
                a_itr[j].reset_val = (a_itr[j].dim - 1) * a_itr[j].advance;
            }
            else if (b_itr[j].dim == a_itr[j].dim)
            {
                b_itr[j].advance = b.strides_[j];
                a_itr[j].advance = strides_[j];
                a_itr[j].reset_val = (a_itr[j].dim - 1) * strides_[j];
                b_itr[j].reset_val = (b_itr[j].dim - 1) * b.strides_[j];
                size_output *= a_itr[j].dim;
                res_dim[j] = a_itr[j].dim;
            }
        }

        /*
        int a_ind = dim_.size();
        int b_ind = b.dim_.size();

        for (int i = 1; i <= dim_size_max; i++)
        {
            a_ind--;
            b_ind--;
            if (a_ind < 0)
            {

                size_output *= b.dim_[b_ind];
                res_dim[dim_size_max - i] = b.dim_[b_ind];
                continue;
            }
            else if (b_ind < 0)
            {
                size_output *= dim_[a_ind];
                res_dim[dim_size_max - i] = dim_[a_ind];
                continue;
            }
            if (dim_[a_ind] == 1 && b.dim_[b_ind] > 1)
            {
                new_a_strides[a_ind] = 0;
                size_output *= b.dim_[b_ind];
                res_dim[dim_size_max - i] = b.dim_[b_ind];
            }
            else if (b.dim_[b_ind] == 1 && dim_[a_ind] > 1)
            {
                new_b_strides[b_ind] = 0;
                size_output *= dim_[a_ind];
                res_dim[dim_size_max - i] = dim_[a_ind];
            }
            else if (dim_[a_ind] != b.dim_[b_ind])
            {
                // throw std::invalid_arguement("Axes are mismatched. Check your code");
            }
            else
            {
                size_output *= dim_[a_ind];
                res_dim[dim_size_max - i] = dim_[a_ind];
            }
        }

        int64_t a_index = offset;
        int64_t b_index = b.offset;
        const T *__restrict b_data = b.data_->data();
        const T *__restrict a_data = data_->data();
        std::vector<size_t> a_counts(dim_.size(), 0);
        std::vector<size_t> b_counts(b.dim_.size(), 0);
        std::vector<T> out(size_output);
        T *out_data = out.data();
        size_t counts_stack[NDIM];
        size_t bcounts_stack[NDIM];
        int64_t astrides_stack[NDIM];
        int64_t bstrides_stack[NDIM];

        for (size_t i = 0; i < a_counts.size(); i++)
            counts_stack[i] = a_counts[i];

        for (size_t i = 0; i < b_counts.size(); i++)
            bcounts_stack[i] = b_counts[i];

        for (size_t i = 0; i < new_a_strides.size(); i++)
            astrides_stack[i] = new_a_strides[i];

        for (size_t i = 0; i < new_b_strides.size(); i++)
            bstrides_stack[i] = new_b_strides[i];

        size_t *counts_data = counts_stack;
        size_t *bcounts_data = bcounts_stack;
        int64_t *astrides_data = astrides_stack;
        int64_t *bstrides_data = bstrides_stack;

        const size_t max_a_dim = a_counts.size() - 1;
        const size_t max_b_dim = b_counts.size() - 1;


        if (is_contiguous() && b.is_contiguous()){
            for (size_t j = 0; j < size_output; )
        }
        */
        // std::vector<T> out(size_output);

        // T *out_data = out.data();
        std::shared_ptr<T[]> out(new T[size_output], std::default_delete<T[]>());
        T *__restrict out_data = out.get();

        const T *__restrict b_data = b.data_.get() + b.offset;
        const T *__restrict a_data = data_.get() + offset;
        int64_t a_index = 0;
        int64_t b_index = 0;
        size_t ind_max = dim_size_max - 1;

        for (size_t i = 0; i < size_output; ++i)
        {

            out_data[i] = *a_data - *b_data;

            getIndex(a_itr, b_itr, 0, ind_max, a_data, b_data);
        }
        // return tensor<T>(std::move(out), res_dim);
        return tensor<T>(out, size_output, res_dim);
    }
    tensor<T> operator*(const tensor<T> &b) const
    {
        size_t size_output = 1;
        size_t dim_size_min = std::min(dim_.size(), b.dim_.size());
        size_t dim_size_max = std::max(dim_.size(), b.dim_.size());
        std::vector<size_t> res_dim(dim_size_max);
        AxisIter a_itr[NDIM];
        AxisIter b_itr[NDIM];

        for (int j = 0; j < dim_size_max; ++j)
        {
            a_itr[j].dim = j < dim_.size() ? dim_[j] : 1;
            b_itr[j].dim = j < b.dim_.size() ? b.dim_[j] : 1;

            if (a_itr[j].dim == 1 && b_itr[j].dim > 1)
            {
                a_itr[j].dim = b_itr[j].dim;
                size_output *= b_itr[j].dim;
                res_dim[j] = b_itr[j].dim;
                b_itr[j].advance = b.strides_[j];
                b_itr[j].reset_val = (b_itr[j].dim - 1) * b_itr[j].advance;
            }
            else if (b_itr[j].dim == 1 && a_itr[j].dim > 1)
            {
                b_itr[j].dim = a_itr[j].dim;
                size_output *= a_itr[j].dim;
                res_dim[j] = a_itr[j].dim;
                a_itr[j].advance = strides_[j];
                a_itr[j].reset_val = (a_itr[j].dim - 1) * a_itr[j].advance;
            }
            else if (b_itr[j].dim == a_itr[j].dim)
            {
                b_itr[j].advance = b.strides_[j];
                a_itr[j].advance = strides_[j];
                a_itr[j].reset_val = (a_itr[j].dim - 1) * strides_[j];
                b_itr[j].reset_val = (b_itr[j].dim - 1) * b.strides_[j];
                size_output *= a_itr[j].dim;
                res_dim[j] = a_itr[j].dim;
            }
        }
        // std::vector<T> out(size_output);

        // T *out_data = out.data();
        std::shared_ptr<T[]> out(new T[size_output], std::default_delete<T[]>());
        T *__restrict out_data = out.get();

        const T *__restrict b_data = b.data_.get() + b.offset;
        const T *__restrict a_data = data_.get() + offset;
        int64_t a_index = 0;
        int64_t b_index = 0;
        size_t ind_max = dim_size_max - 1;

        for (size_t i = 0; i < size_output; ++i)
        {

            out_data[i] = (*a_data) * (*b_data);

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
    tensor<T> operator/(const tensor<T> &b) const
    {
        size_t size_output = 1;
        size_t dim_size_min = std::min(dim_.size(), b.dim_.size());
        size_t dim_size_max = std::max(dim_.size(), b.dim_.size());
        std::vector<size_t> res_dim(dim_size_max);
        AxisIter a_itr[NDIM];
        AxisIter b_itr[NDIM];

        for (int j = 0; j < dim_size_max; ++j)
        {
            a_itr[j].dim = j < dim_.size() ? dim_[j] : 1;
            b_itr[j].dim = j < b.dim_.size() ? b.dim_[j] : 1;

            if (a_itr[j].dim == 1 && b_itr[j].dim > 1)
            {
                a_itr[j].dim = b_itr[j].dim;
                size_output *= b_itr[j].dim;
                res_dim[j] = b_itr[j].dim;
                b_itr[j].advance = b.strides_[j];
                b_itr[j].reset_val = (b_itr[j].dim - 1) * b_itr[j].advance;
            }
            else if (b_itr[j].dim == 1 && a_itr[j].dim > 1)
            {
                b_itr[j].dim = a_itr[j].dim;
                size_output *= a_itr[j].dim;
                res_dim[j] = a_itr[j].dim;
                a_itr[j].advance = strides_[j];
                a_itr[j].reset_val = (a_itr[j].dim - 1) * a_itr[j].advance;
            }
            else if (b_itr[j].dim == a_itr[j].dim)
            {
                b_itr[j].advance = b.strides_[j];
                a_itr[j].advance = strides_[j];
                a_itr[j].reset_val = (a_itr[j].dim - 1) * strides_[j];
                b_itr[j].reset_val = (b_itr[j].dim - 1) * b.strides_[j];
                size_output *= a_itr[j].dim;
                res_dim[j] = a_itr[j].dim;
            }
        }

        // std::vector<T> out(size_output);

        // T *out_data = out.data();
        std::shared_ptr<T[]> out(new T[size_output], std::default_delete<T[]>());
        T *__restrict out_data = out.get();

        const T *__restrict b_data = b.data_.get() + b.offset;
        const T *__restrict a_data = data_.get() + offset;
        int64_t a_index = 0;
        int64_t b_index = 0;
        size_t ind_max = dim_size_max - 1;

        for (size_t i = 0; i < size_output; ++i)
        {

            out_data[i] = (*a_data) / (*b_data);

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
    template <typename ElmOp>
    tensor<T> reduce_op(int a, ElmOp op)
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
            itr[ind].advance = strides_[ind];
            itr[ind].reset_val = (dim_[ind] - 1) * strides_[ind];
            itr[ind].dim = dim_[ind];
            size_output *= dim_[ind];
        }
        /*
        std::vector<T> out(size_output);
        T *__restrict out_data = out.data();
        const T *__restrict data__ = data_->data() + offset;
        */
        // std::vector<T> out(size_output);

        // T *out_data = out.data();
        std::shared_ptr<T[]> out(new T[size_output], std::default_delete<T[]>());
        const T *__restrict out_data = out.get();
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

        // return tensor<T>(std::move(out), new_dim);
        return tensor<T>(out, size_output, new_dim);
    }
    /*
    tensor<T> argmax(int axis = -1){
        int64_t intial_offset =
    } tensor<T> argmin(int axis = -1)
    {
    }
    */
    tensor<double> power(const double &a)
    {
        std::vector<T> new_data_ = *data_;
        for (auto &elm : new_data_)
        {
            elm = static_cast<T>(std::pow(elm, a));
        }
        return tensor<T>(new_data_, dim_);
    }

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

        /*
        std::vector<T> new_data;
        new_data.resize(this_size * other_size); // avoid zero-fill if not needed
        T *__restrict new_data_ = new_data.data();
        */

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

    // General tensor contraction over arbitrary axes
    // axes_this and axes_other must be same length
    /*
    tensor<T> operator+(const tensor<T>& b) const
{
    if (size() != b.size())
        throw std::runtime_error("Size mismatch");

    std::vector<T> out(size());

    for (size_t i = 0; i < size(); ++i)
        out[i] = data_[i] + b.data_[i];

    return tensor<T>(out, dim_);
}
    */
    tensor<T> &matrixPow(const size_t &val);
    tensor<T> &elemPow(const size_t &val);

    friend std::ostream &operator<<(std::ostream &out, const tensor<T> &tensor)
    {
        printTens(out, tensor, tensor.offset);
        return out;
    }
};

template <typename T>
void printTens(std::ostream &out, tensor<T> tensor, int64_t start, size_t depth = 1)
{

    std::vector<int64_t> stride = tensor.strides();
    std::vector<size_t> dim = tensor.dim();
    const T *__restrict data = tensor.data();

    int width = 0;
    if (width == 0)
    {
        /*
        for (const auto &val : data)
        {
            width = std::max(width, (int)std::to_string(val).size());
        }
            */
        for (size_t ind = 0; ind < tensor.size(); ++ind)
        {
            width = std::max(width, (int)std::to_string(data[ind]).size());
        }
    }

    if (depth == dim.size())
    {
        int dir = 1;
        int64_t cur_stride = stride[depth - 1];
        int64_t last_dim = dim[depth - 1];
        out << "[";
        for (size_t times = 0; times < last_dim; times++)
        {

            out << std::setw(width) << data[start] << " ";
            start += cur_stride;
        }
        out << "]";
        return;
    }

    int64_t cur_stride = stride[depth - 1];
    size_t num_sub_tensors = dim[depth - 1];
    out << "[";
    for (int64_t i = 0; i < num_sub_tensors; i++)
    {
        if (i > 0)
        {
            for (size_t pad = 0; pad < depth; pad++)
            {
                out << " ";
            }
        }

        printTens(out, tensor, start + i * cur_stride, depth + 1);
        if (i != num_sub_tensors - 1)
            out << "\n\n";
    }
    out << "]";
}
