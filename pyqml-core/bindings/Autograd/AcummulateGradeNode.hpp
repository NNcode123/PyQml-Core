#include "Node.hpp"
#include "../grad_meta.hpp"

struct AccumulateGradNode: Node{

    AccumulateGradNode(std::vector<Edge>&& func): Node(std::move(func)) {}

    std::vector<Tensor> backward(std::vector<Tensor>&& args){
        auto grad_m = *grad_info;
        grad_m.grad += args[0];
        return {};
    }

};