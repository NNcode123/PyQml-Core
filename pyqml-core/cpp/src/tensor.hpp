#include <vector>
#include <complex.h>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <iomanip>
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
};

template <typename T>
class tensor
{
    std::vector<T> data_;
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
                     size_t end_dim_index, int64_t cur_pos, size_t step = 1) const
    {
        if (dim_[end_dim_index] == cur_counts[end_dim_index] + 1)
        {
            int index = end_dim_index;
            while (index > start_dim_index && dim_[index] == cur_counts[index] + 1)
            {

                cur_pos -= step * cur_counts[index] * strides_[index];

                cur_counts[index] = 0;

                index--;
            }
            if (index >= start_dim_index)
            {
                cur_counts[index]++;
                cur_pos += strides_[index] * step;
            }
        }
        else
        {
            cur_counts[end_dim_index]++;
            cur_pos += strides_[end_dim_index] * step;
        }
        return cur_pos;
    }

public:
    explicit tensor(const std::vector<T> &data, const std::vector<size_t> &dim) : data_(data), dim_(dim)

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
    [[nodiscard]] std::vector<T> data() const { return data_; }
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
        return data_[index];
    }
    template <typename... Indices>
    [[nodiscard]] T operator()(Indices... indices) const
    {
        std::vector<size_t> pos = {indices...};
        return at(pos);
    }

    SlicePlan analyze_slices(std::vector<Slice> &inds)
    {
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
            new_dim[i] = std::abs(inds[i].end - inds[i].start) / (std::abs(inds[i].step)) + 1;
            if (std::abs(inds[i].end - inds[i].start) % std::abs(inds[i].step) == 0)
            {
                new_dim[i]--;
            }

            new_data_size *= new_dim[i];
            cur_index += inds[i].start * strides_[i];
        }
        for (size_t i = inds.size(); i < dim_.size(); i++)
        {
            new_dim[i] = dim_[i];
            new_data_size *= new_dim[i];
            inds.push_back(Slice(0, new_dim[i]));
        }
        return SlicePlan(new_dim, cur_counts, cur_index, new_data_size);
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
        for (size_t i = 0; i < new_data.size(); i++)
        {
            new_data[i] = data_[cur_index];

            if (new_dim.back() == cur_counts.back() + 1)
            {

                int cur_pos = new_dim.size() - 1;
                while (cur_pos >= 0 && new_dim[cur_pos] == cur_counts[cur_pos] + 1)
                {
                    /*
                        if (inds[cur_pos].step > 0)
                        {
                            int64_t last_tensor_index = inds[cur_pos].start + inds[cur_pos].step * cur_counts[cur_pos];
                            cur_index -= last_tensor_index * strides_[cur_pos];
                        }
                        else
                        {

                            cur_index -= inds[cur_pos].step * cur_counts[cur_pos] * strides_[cur_pos];
                        }
                        */
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

        return tensor<T>(new_data, new_dim);
    }
    template <typename... Slices>

    [[nodiscard]] tensor<T> slice_view(const Slices &...indices)
    {

        std::vector<Slice> inds = {indices...};
        SlicePlan plan = analyze_slices(inds);
        std::vector<size_t> new_dim = plan.dim;
        size_t cur_index = plan.start_index;
        std::vector<int64_t> new_strides = strides_;
        for (int i = 0; i < inds.size(); ++i)
        {
            new_strides[i] *= inds[i].step;
        }

        return tensor<T>(data_, new_dim, new_strides, cur_index);
    }

    std::pair<std::vector<size_t>, std::vector<int64_t>>
    collapse(const std::vector<size_t> &dim,
             const std::vector<int64_t> &strides)
    {

        std::vector<size_t> cdim = dim;
        std::vector<int64_t> cstrides = strides;

        if (cdim.size() < 2 || cstrides.back() != 1)
        {
            return {cdim, cstrides};
        }

        for (int i = cdim.size() - 1; i > 0; --i)
        {
            if (cstrides[i - 1] == static_cast<int64_t>(cdim[i]) * cstrides[i])
            {

                size_t last_dim = cdim.back();

                cdim.pop_back();
                cdim.back() *= last_dim;

                cstrides.pop_back();
            }
            else
            {
                break;
            }
        }

        return {cdim, cstrides};
    }

    std::pair<size_t, size_t> collapse_size() const
    {
        if (strides_.back() != 1)
        {
            return {1, 0};
        }
        size_t cur_size = dim_.back();
        size_t offset = 0;
        for (int i = dim_.size() - 1; i > 0; --i)
        {
            if (strides_[i - 1] == dim_[i] * strides_[i])
            {
                cur_size *= dim_[i - 1];
                offset++;
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
            return *this;
        }
        std::vector<T> new_data(size(), 0);
        std::vector<size_t> counts(dim_.size(), 0);
        int64_t cur_index = offset;
        for (size_t i = 0; i < new_data.size(); i++)
        {
            new_data[i] = data_[cur_index];
            cur_index = getIndex(counts, 0, dim_.size() - 1, cur_index);
        }
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

        auto A = copy();
        auto B = other.copy();

        const size_t this_size = size();
        const size_t other_size = other.size();
        const T *__restrict A_data = A.data_.data();
        const T *__restrict B_data = B.data_.data();

        std::vector<T>
            new_data;
        new_data.resize(this_size * other_size); // avoid zero-fill if not needed
        T *__restrict new_data_ = new_data.data();

        for (size_t i = 0; i < this_size; ++i)
        {

            T a_val = A_data[i];
            T *out = new_data_ + i * other_size;

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
        }
        else
        {
            out << "]";
        }
    }
    if (depth == 1)
        out << "]";
}