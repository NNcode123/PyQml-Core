#include "../Tensor.hpp"

Tensor einsum_(const Tensor &a, const Tensor &b, const std::vector<int> &axes_a, const std::vector<int> &axes_b)
{
    return Tensor::dispatchOp(a, b, [&](auto &t_1, auto &t_2)
                              { return einsum(t_1, t_2, axes_a, axes_b); });
}
