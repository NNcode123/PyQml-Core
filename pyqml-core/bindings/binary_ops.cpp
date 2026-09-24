#include "Tensor.hpp"
#include "Autograd/attach_grad_binary.hpp"

Tensor Tensor::astype(DType new_type, [[maybe_unused]] bool h) const
{
    if (dtype == new_type)
    {
        return *this;
    }

    TYPE_CAST_DISPATCH((*this), new_type)
}

Tensor Tensor::operator+(const Tensor &other) const
{
    return bin_op(*this, other, [&](auto &t_1, auto &t_2)
                  { return binary_ops(t_1, t_2, std::plus<>()); });
}

Tensor Tensor::operator-(const Tensor &other) const
{
    auto result = [&]()
    {
        BINARY_OP_DISPATCH((*this), other, [&](auto &t_1, auto &t_2)
                           { return binary_ops(t_1, t_2, std::minus<>()); });
    }();

    if (requires_grad() || other.requires_grad())
    {
        Attach_Grad(result, Sub, (*this), other)
    }

    return result;
}

Tensor Tensor::operator*(const Tensor &other) const
{
    BINARY_OP_DISPATCH((*this), other, [&](auto &t_1, auto &t_2)
                       { return binary_ops(t_1, t_2, std::multiplies<>()); });
}

Tensor Tensor::operator/(const Tensor &other) const
{
    BINARY_OP_DISPATCH((*this), other, [&](auto &t_1, auto &t_2)
                       { return binary_ops(t_1, t_2, std::divides<>()); });
}

Tensor &Tensor::operator+=(const Tensor &other)
{
    if (!data)
    {
        *this = other;
        return *this;
    }

    *this = *this + other;
    return *this;
}

Tensor einsum_(const Tensor &a, const Tensor &b,
               const std::vector<int> &axes_a, const std::vector<int> &axes_b)
{
    return bin_op(a, b, [&](auto &t_1, auto &t_2)
                  { return einsum(t_1, t_2, axes_a, axes_b); });
}
