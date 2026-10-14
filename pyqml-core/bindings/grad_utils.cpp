#include <pyqml-core/bindings/Tensor.hpp>
#include <pyqml-core/bindings/Autograd/Engine.hpp>


Tensor& Tensor::get_grad() {
    return get_info()->grad;
}

const Tensor& Tensor::const_get_grad() const {
    return info->grad;
}

pyq_intrusive_ptr<Node> Tensor::grad_fn() const {
    return info ? info->node : pyq_intrusive_ptr<Node>{};
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


void Tensor::backward() const {
    if (!requires_grad()){return;}

    Engine eng{};

    Node* start_fn = grad_fn().storage_ptr();

    eng.backward(start_fn);

}

pyq_intrusive_ptr<grad_meta>& Tensor::get_info() {
    return info;
}


/*
void Tensor::set_info(pyq_intrusive_ptr<grad_meta> new_info) noexcept {
    info = std::move(new_info);
}
*/