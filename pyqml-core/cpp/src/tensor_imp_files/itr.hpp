#pragma once
#include <vector>
#include <cstdint>

namespace detail
{
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

    template <typename T>
    inline void getIndex(AxisIter *axis_info, size_t start_dim, size_t end_dim, const T *__restrict &src)
    {
        while (axis_info[end_dim].count >= axis_info[end_dim].dim)
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
                            size_t start_index, size_t end_index, int64_t cur_pos)
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

    /*template <typename T>
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
    }
        */

    template <typename T, typename V>

    inline void getIndex(AxisIter *axes_a, AxisIter *axes_b, size_t start_index, size_t end_index, const T *__restrict &a_data, const V *__restrict &b_data)
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

}


namespace gpu_detail{


    
}