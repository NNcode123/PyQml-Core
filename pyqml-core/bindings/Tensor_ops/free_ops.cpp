#include "../Tensor.hpp"

// This wrapper forwards an einsum-style contraction to the core tensor implementation
// while preserving the Tensor-level interface used by the Python bindings.
Tensor einsum_(const Tensor &a, const Tensor &b, const std::vector<int> &axes_a, const std::vector<int> &axes_b)
{
    return Tensor::dispatchOp(a, b, [&](auto &t_1, auto &t_2)
                              { return einsum(t_1, t_2, axes_a, axes_b); });
}

/*

Tensor concat_(const Tensor& a, const Tensor&b, size_t axis){
    return Tensor::dispatchOp(a, b, [&](auto &t_1, auto &t_2)
                            {
                                return concat(t_1, t_2); });
}


Tensor hstack_(const Tensor& a, const Tensor&b){
    return Tensor::dispatchOp(a, b, [&](auto &t_1, auto &t_2)
                            {
                                return hstack(t_1, t_2); });
}


Tensor vstack_(const Tensor& a, const Tensor&b, size_t axis){
    return Tensor::dispatchOp(a, b, [&](auto &t_1, auto &t_2)
                            {
                                return vstack(t_1, t_2); });
}

*/