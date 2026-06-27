#include <thread>
#include "../tensor_imp_files/itr.hpp"
#include "threads.hpp"

using namespace detail;

namespace parallel_sync
{

    static threads pool;
    // This helper snapshots the iterator state for a given position so parallel work can be
    // partitioned without losing the current logical coordinates of the broadcasted axes.
    void init_itr_state(size_t pos, size_t ndim, AxisIter *a_itr, AxisIter *b_itr, AxisIter *a_out, AxisIter *b_out)
    {
        for (size_t i = 0; i < ndim; ++i)
        {
            a_out[i] = a_itr[i];
            b_out[i] = b_itr[i];
        }
        for (int j = ndim - 1; j >= 0; --j)
        {
            a_out[j].count = (pos % (a_itr[j].dim)) + 1;
            b_out[j].count = (pos % (b_itr[j].dim)) + 1;
            pos = pos / a_itr[j].dim;
        }
    }

}