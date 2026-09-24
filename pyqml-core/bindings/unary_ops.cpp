#include "Tensor.hpp"

std::string Tensor::print_val()
{
    return getProp([](auto &t)
                   { return t.get_str(); });
}

Tensor Tensor::max(const std::vector<int> &axis) const
{
    return getTens([&](auto &t)
                   { return t.max(axis); });
}

Tensor Tensor::min(const std::vector<int> &axis) const
{
    return getTens([&](auto &t)
                   { return t.min(axis); });
}

Tensor Tensor::sum(const std::vector<int> &axis, bool keepdim) const
{
    return getTens([&](auto &t)
                   { return t.sum(axis, keepdim); });
}

Tensor Tensor::ones(const std::vector<size_t> &shape, DType type)
{
    return Tensor::fill(shape, 1, type);
}

Tensor Tensor::zeroes(const std::vector<size_t> &shape, DType type)
{
    return Tensor::fill(shape, 0, type);
}

Tensor Tensor::reshape(const std::vector<size_t> &shape)
{
    return getTens([&](auto &t)
                   { return t.reshape(shape); });
}

Tensor Tensor::unbroadcast(const Tensor &in, const std::vector<size_t> &shape)
{
    const auto &original_shape = in.shape_;
    std::vector<int> broadcast_axes;

    for (size_t axis = 0; axis < original_shape.size(); ++axis)
    {
        if (original_shape[axis] != 1 && shape[axis] == 1)
        {
            broadcast_axes.push_back(static_cast<int>(axis));
        }
    }

    return in.sum(broadcast_axes);
}
