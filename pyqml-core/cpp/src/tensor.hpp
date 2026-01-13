#include <vector>
#include <array>
#include <complex.h>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <cstring>
#include <memory>

#pragma once

using size_t = std::size_t;
template <typename T>
struct typing
{
    // To be Defined Later....
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
    size_t start_index;
    size_t size;
    SlicePlan(std::vector<size_t> d, std::vector<size_t> cts, size_t s_index, size_t s) : dim(d), counts(cts), start_index(s_index), size(s) {}
    SlicePlan(std::vector<size_t> d, size_t s_idx, size_t s) : dim(d), start_index(s_idx), size(s) {}
};

template <typename T>
class tensor
{
    std::shared_ptr<std::vector<T>> data_;
    std::vector<size_t> dim_;
    std::vector<int64_t> strides_;
    bool owns_data;
    std::size_t offset;
    explicit tensor(const std::vector<T> &data, const std::vector<size_t> &dim,
                    const std::vector<int64_t> &strides, size_t start) : data_(data), dim_(dim), strides_(strides), offset(start)
    {
        owns_data = false;
    }

    int64_t getIndex(std::vector<size_t> &cur_counts, size_t start_dim_index,
                     size_t end_dim_index, int64_t cur_pos) const
    {
        int index = end_dim_index;
        while (index > start_dim_index && dim_[index] == cur_counts[index] + 1)
        {

            cur_pos -= cur_counts[index] * strides_[index];

            cur_counts[index] = 0;

            index--;
        }
        cur_counts[index]++;
        cur_pos += strides_[index];

        return cur_pos;
    }

public:
    explicit tensor(const std::vector<T> &data, const std::vector<size_t> &dim) : data_(std::make_shared<std::vector<T>>(data)), dim_(dim)

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
    [[nodiscard]] std::vector<T> data() const { return *data_; }
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
        return (*data_)[index];
    }
    template <typename... Indices>
    [[nodiscard]] T operator()(Indices... indices) const
    {
        std::vector<size_t> pos = {indices...};
        return at(pos);
    }

    SlicePlan analyze_slices(std::vector<Slice> &inds)

    {
        for (size_t i = inds.size(); i < dim_.size(); i++)
        {
            inds.emplace_back(Slice(0, dim_[i]));
        }
        std::vector<size_t> new_dim(dim_.size());
        std::vector<size_t> cur_counts(dim_.size(), 0);
        size_t new_data_size = 1;
        size_t cur_index = offset;
        for (size_t i = 0; i < inds.size(); ++i)
        {
            if (inds[i].end < 0 && !(inds[i].step < 0 && inds[i].end == -1))
            {
                inds[i].end += dim_[i];
            }
            if (inds[i].start < 0)
            {
                inds[i].start += dim_[i];
            }
            const int64_t span = std::abs(inds[i].end - inds[i].start);
            const int64_t step = std::abs(inds[i].step);
            size_t len = span / step + 1;
            if (span % step == 0)
            {
                --len;
            }
            new_dim[i] = len;
            new_data_size *= len;
            cur_index += inds[i].start * strides_[i];
        }

        return SlicePlan(new_dim, cur_counts, cur_index, new_data_size);
    }
    SlicePlan analyze_slices(const Slice *inds, size_t inds_size)
    {
        const size_t ndim = dim_.size();

        std::vector<size_t> new_dim(ndim);
        size_t new_data_size = 1;
        size_t cur_index = offset;

        for (size_t i = 0; i < ndim; ++i)
        {

            Slice s = (i < inds_size) ? inds[i] : Slice(0, dim_[i], 1);

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

            new_dim[i] = len;
            new_data_size *= len;
            cur_index += s.start * strides_[i];
        }

        return SlicePlan(new_dim, cur_index, new_data_size);
    }
    template <typename... Slices>
    [[nodiscard]] tensor<T> slice(const Slices &...indices)
    {

        std::vector<Slice> inds = {indices...};
        SlicePlan plan = analyze_slices(inds);
        std::vector<size_t> cur_counts = plan.counts;
        std::vector<size_t> new_dim = plan.dim;
        int64_t cur_index = plan.start_index;
        size_t new_data_size = plan.size;

        std::vector<T> new_data(new_data_size, 0);
        for (size_t i = 0; i < new_data.size(); ++i)
        {
            new_data[i] = (*data_)[cur_index];

            if (new_dim.back() == cur_counts.back() + 1)
            {

                int cur_pos = new_dim.size() - 1;
                while (cur_pos >= 0 && new_dim[cur_pos] == cur_counts[cur_pos] + 1)
                {
                    cur_index -= inds[cur_pos].step * cur_counts[cur_pos] * strides_[cur_pos];

                    cur_counts[cur_pos] = 0;

                    cur_pos--;
                }
                if (cur_pos >= 0)
                {
                    cur_counts[cur_pos]++;
                    cur_index += strides_[cur_pos] * inds[cur_pos].step;
                }
            }
            else
            {
                cur_counts.back()++;
                cur_index += strides_.back() * inds.back().step;
            }
        }

        return tensor<T>(std::move(new_data), new_dim);
    }
    template <typename... Slices>

    [[nodiscard]] tensor<T> slice_view(const Slices &...indices)
    {

        tensor<T> out = *this;
        std::array<Slice, sizeof...(indices)> inds = {indices...};
        SlicePlan plan = analyze_slices(inds.data(), inds.size());
        out.dim_ = plan.dim;
        out.offset = plan.start_index;
        for (int i = 0; i < inds.size(); ++i)
        {
            if (inds[i].step != 1)
                out.strides_[i] *= inds[i].step;
        }

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
            return tensor<T>(*data_, dim_);
        }

        std::vector<T> new_data;
        new_data.resize(size());
        T *__restrict data = new_data.data();
        const T *__restrict src = data_->data();
        std::vector<size_t> counts(dim_.size(), 0);
        const auto [cont_size, cont_offset] = collapse_size();
        size_t radix_size = size() / cont_size;
        int64_t radix_dim_pos = cont_offset;
        int64_t cur_pos = offset;
        int64_t dest_pos = 0;

        if (cont_offset == 0)
        {
            if (cont_size > 1)
            {

                const int64_t dstride = strides_[0];
                int64_t c_pos = offset;
                int64_t d_pos = 0;
                for (size_t i = 0; i < radix_size; ++i)
                {
                    std::memcpy(data + d_pos,
                                src + c_pos,
                                cont_size * sizeof(T));
                    c_pos += dstride;
                    d_pos += cont_size;
                }
            }
            else if (dim_.size() == 1)
            {
                const int64_t dstride = strides_[0];
                for (size_t ind = 0; ind < radix_size; ++ind)
                {
                    data[++dest_pos] = src[cur_pos];
                    cur_pos += dstride;
                }
            }
            else if (dim_.size() == 2)
            {

                const size_t dim0_len = dim_[0];
                const size_t dim1_len = dim_[1];
                const int64_t stride0 = strides_[0];
                const int64_t stride1 = strides_[1];

                for (size_t ind = 0; ind < dim0_len; ++ind)
                {
                    const int64_t affine_walk_0 = offset + ind * stride0;
                    for (size_t j = 0; j < dim1_len; ++j)
                    {
                        const int64_t cur_pos = affine_walk_0 + j * stride1;
                        data[dest_pos++] = src[cur_pos];
                    }
                }
            }
            else if (dim_.size() == 3)
            {
                const size_t dim0_len = dim_[0];
                const size_t dim1_len = dim_[1];
                const size_t dim2_len = dim_[2];
                const int64_t stride0 = strides_[0];
                const int64_t stride1 = strides_[1];
                const int64_t stride2 = strides_[2];
                int64_t start = 0;

                for (size_t ind = 0; ind < dim0_len; ++ind)
                {
                    const int64_t affine_walk_0 = offset + stride0 * ind;
                    for (size_t j = 0; j < dim1_len; ++j)
                    {
                        const int64_t affine_walk_1 = affine_walk_0 + stride1 * j;
                        for (size_t k = 0; k < dim2_len; ++k)
                        {
                            const int64_t pos = affine_walk_1 + stride2 * k;
                            data[start++] = src[pos];
                        }
                    }
                }
            }
        }
        else if (cont_offset == 1)
        {

            const size_t dim0_len = dim_[0];
            const size_t dim1_len = dim_[1];
            const int64_t stride0 = strides_[0];
            const int64_t stride1 = strides_[1];

            size_t d_pos = 0;

            for (size_t ind = 0; ind < dim0_len; ++ind)
            {

                const int64_t base0 = offset + ind * stride0;
                for (size_t j = 0; j < dim1_len; ++j)
                {
                    std::memcpy(data + d_pos,
                                src + (base0 + j * stride1),
                                cont_size * sizeof(T));

                    d_pos += cont_size;
                }
            }
        }
        else if (cont_offset == 2)
        {

            const size_t dim0_len = dim_[0];
            const size_t dim1_len = dim_[1];
            const size_t dim2_len = dim_[2];
            const int64_t stride0 = strides_[0];
            const int64_t stride1 = strides_[1];
            const int64_t stride2 = strides_[2];
            int64_t d_pos = 0;

            for (size_t ind = 0; ind < dim0_len; ++ind)
            {
                const int64_t affine_walk_0 = offset + stride0 * ind;
                for (size_t j = 0; j < dim1_len; ++j)
                {
                    const int64_t affine_walk_1 = affine_walk_0 + stride1 * j;
                    for (size_t k = 0; k < dim2_len; ++k)
                    {
                        const int64_t pos = affine_walk_1 + stride2 * k;
                        std::memcpy(data + d_pos,
                                    src + pos,
                                    cont_size * sizeof(T));
                        d_pos += cont_size;
                    }
                }
            }
        }
        else
        {
            for (size_t i = 0; i < radix_size; ++i)
            {

                std::memcpy(data + dest_pos + cont_size * i,
                            src + cur_pos,
                            cont_size * sizeof(T));

                cur_pos = getIndex(counts, 0, radix_dim_pos, cur_pos);
            }
        }

        /*
         for (size_t i = 0; i < radix_size; ++i)
         {

             std::memcpy(new_data.data() + dest_pos + cont_size * i,
                         data_.data() + cur_pos,
                         cont_size * sizeof(T));

             cur_pos = getIndex(counts, 0, radix_dim_pos, cur_pos);
         }
         */

        return tensor<T>(new_data, dim_);
    }
    [[nodiscard]] const std::vector<size_t> &shape() const { return dim_; }
    [[nodsicard]] tensor<T> reshape(const std::vector<size_t> &newshape) const
    {
        return tensor<T>(data_, newshape);
    }
    [[nodiscard]]
    tensor<T> tensor_prod(const tensor<T> &other) const
    {

        std::vector<size_t> new_dim = dim_;
        new_dim.insert(new_dim.end(), other.dim_.begin(), other.dim_.end());

        const auto A = copy();
        const auto B = other.copy();

        const size_t this_size = size();
        const size_t other_size = other.size();
        const T *__restrict A_data = A.data_->data();
        const T *__restrict B_data = B.data_->data();

        std::vector<T>
            new_data;
        new_data.resize(this_size * other_size); // avoid zero-fill if not needed
        T *__restrict new_data_ = new_data.data();

        for (size_t i = 0; i < this_size; ++i)
        {

            const T a_val = A_data[i];
            T *__restrict out = new_data_ + i * other_size;

            for (size_t j = 0; j < other_size; ++j)
            {
                out[j] = a_val * B_data[j];
            }
        }

        return tensor<T>(new_data, new_dim);
    }

    // General tensor contraction over arbitrary axes
    // axes_this and axes_other must be same length

    tensor<T> &matrixPow(const size_t &val);
    tensor<T> &elemPow(const size_t &val);
    tensor<T> &operator+(tensor<T> &tens)
    {
        return tensor<T>();
    }

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
    std::vector<T> data = tensor.data();
    int width = 0;
    if (width == 0)
    {
        for (const auto &val : data)
        {
            width = std::max(width, (int)std::to_string(val).size());
        }
    }

    if (depth == dim.size())
    {
        int dir = 1;
        int64_t cur_stride = stride[depth - 1];
        if (cur_stride < 0)
        {
            dir *= -1;
        }

        for (size_t times = 0; times < dim[depth - 1]; times++)
        {

            out << std::setw(width) << data[start] << " ";
            start += dir;
        }
        return;
    }

    int64_t cur_stride = stride[depth - 1];
    size_t num_sub_tensors = dim[depth - 1];

    if (depth == 1)
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
        out << "[";
        printTens(out, tensor, start + i * cur_stride, depth + 1);
        if (i != num_sub_tensors - 1)
        {
            out << "]\n";
            if (depth >= 1 && depth < dim.size())
                out << "\n";
        }
        else
        {
            out << "]";
        }
    }
    if (depth == 1)
        out << "]";
}