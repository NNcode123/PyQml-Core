#include "Tensor.hpp"
#include "Node.hpp"



struct DivNode: public Node {

    Tensor lhs;

    Tensor rhs;

    DivNode(std::vector<Edge>&& funcs, std::vector<InputMetadata>&& info, const Tensor & a, const Tensor & b): Node(std::move(funcs), std::move(info)), lhs(a), rhs(b) {}


    std::vector<Tensor> backward(std::vector<Tensor>&& args){

    
        Tensor lhs_rw = Tensor::unbroadcast(1/rhs * args[0], info[0].shape);
        Tensor rhs_rw = Tensor::unbroadcast(lhs/(rhs*rhs) * args[0], info[1].shape);

        return {lhs_rw.astype(info[0].type), rhs_rw.astype(info[1].type)};



    }




};