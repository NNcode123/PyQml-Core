#include "Node.hpp"
#include "../grad_meta.hpp"

struct AccumulateGradNode: public Node{

    Tensor tens;

    AccumulateGradNode(std::vector<Edge>&& func, std::vector<InputMetadata>&& info, const Tensor& a): Node(std::move(func), std::move(info)), tens(a) {}

    std::vector<Tensor> backward(std::vector<Tensor>&& args){
        auto grad_m = *grad_info;
        grad_m.grad += args[0];
        return {};
    }

};