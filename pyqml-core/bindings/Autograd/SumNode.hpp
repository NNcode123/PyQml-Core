#pragma once

#include <pyqml-core/bindings/Autograd/Node.hpp>
#include <pyqml-core/bindings/Tensor.hpp>

struct SumNode : Node
{
    SumNode(std::vector<Edge> &&edges, std::vector<InputMetadata> &&info)
        : Node(std::move(edges), std::move(info))
    {
    }

    std::vector<Tensor> backward(std::vector<Tensor> &&grads) override
    {
        Tensor input_gradient = Tensor::ones(info[0].shape, info[0].type) * grads[0];
        return {input_gradient.astype(info[0].type)};
    }
};





