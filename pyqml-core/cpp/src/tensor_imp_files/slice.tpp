#include <iostream>
#include <cstdint>
#include "../tensor.hpp"
#include <type_traits>

template <typename T>
std::pair<SlicePlan, std::vector<AxisIter>> tensor<T>::analyze_slices(const AxisType *inds, const size_t inds_size)

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

template <typename T>
SlicePlan tensor<T>::analyze_slices(const AxisView *inds, size_t inds_size)
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

template <typename T>
template <typename... Slices>
[[nodiscard]] tensor<T> tensor<T>::slice(const Slices &...indices)
{
    std::array<AxisType, sizeof...(indices)> inds = {indices...};
    auto [plan, Axis_Iter] = analyze_slices(inds.data(), inds.size());
    std::vector<size_t> new_dim = std::move(plan.dim);
    int64_t cur_index = std::move(plan.start_index);
    size_t new_data_size = std::move(plan.size);
    std::shared_ptr<T[]> new_data(new T[new_data_size], std::default_delete<T[]>());
    const T *__restrict data_ptr = new_data.get();

    for (size_t i = 0; i < new_data_size; ++i)
    {
        new_data[i] = data_ptr[cur_index];

        cur_index = getIndex(Axis_Iter, new_dim, 0, new_dim.size() - 1, cur_index);
    }

    return tensor<T>(new_data, new_data_size, new_dim);
}

template <typename T>
template <typename... Slices>
[[nodiscard]] tensor<T> tensor<T>::slice_view(const Slices &...indices)
{
    tensor<T> out;
    std::array<AxisView, sizeof...(indices)> inds = {indices...};
    SlicePlan plan = analyze_slices(inds.data(), inds.size());
    out.data_ = data_;
    out.dim_ = std::move(plan.dim);
    out.offset = plan.start_index;
    out.strides_ = std::move(plan.strides);
    size_t in_size = 1;
    for (const auto &elm : out.dim_)
    {
        in_size *= elm;
    }
    out.t_size = in_size;

    return out;
}

template <typename T>
std::pair<size_t, size_t> tensor<T>::collapse_size() const
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
template <typename T>
bool tensor<T>::is_contiguous() const
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

template <typename T>
[[nodiscard]] tensor<T> tensor<T>::copy() const
{
    if (is_contiguous())
    {
        return *this;
    }

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

    return tensor<T>(new_data, size(), dim_);
}