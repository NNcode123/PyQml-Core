#include "Node.hpp"
#include "../Tensor.hpp"


struct AccumulateGradNode: public Node{

    Tensor var;

    AccumulateGradNode(std::vector<Edge>&& func, std::vector<InputMetadata>&& info, const Tensor& a): Node(std::move(func), std::move(info)), var(a) {}

    std::vector<Tensor> backward(std::vector<Tensor>&& args){
        auto& grad_m = var.get_grad();
        grad_m += args[0];
        return {};
    }

    /*

    private: 
    
    ~AccumulateGradNode() override{

    }
    */

};  