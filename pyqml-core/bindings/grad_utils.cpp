#include "Tensor.hpp"

Tensor& Tensor::get_grad() {
    return info->grad;
}

const Tensor& Tensor::const_get_grad() const {
    return info->grad;
}

const pyq_intrusive_ptr<Node>& Tensor::grad_fn() const {
    return info->node;
}

bool Tensor::is_leaf() const {
    return info ? info->is_leaf : false;
}

bool Tensor::requires_grad() const {
    return info ? info->requires_grad : false;
}

void Tensor::retain_grad() {
    if (info) {
        info->retain_grad = true;
    }
}
