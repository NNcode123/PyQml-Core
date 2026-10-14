#include <pyqml-core/bindings/Tensor.hpp>
#include <pyqml-core/bindings/Autograd/SumNode.hpp>

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
    Tensor result = getTens([&](auto &t)
                            { return t.sum(axis, keepdim); });

    if (requires_grad())
    {
        std::vector<Edge> edges{{grad_fn(), 0}};
        std::vector<InputMetadata> metadata{
            InputMetadata{.shape = shape_, .type = dtype}};
        auto node = make_intrusive<SumNode>(std::move(edges), std::move(metadata));
        result.info = pyq_intrusive_ptr<grad_meta>(
            new grad_meta{Tensor(), true, false, false, node});
    }

    return result;
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
    
    if (original_shape == shape)
    {
        return in;
    }

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
