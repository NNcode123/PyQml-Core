#include "Tensor.hpp"
#include "Node.hpp"

struct MulNode: public Node {

    Tensor lhs;

    Tensor rhs;

    MulNode(std::vector<Edge>&& funcs, const Tensor & a, const Tensor & b): Node(std::move(funcs)), lhs(a), rhs(b) {}


    std::vector<Tensor> backward(std::vector<Tensor>&& args){

    
        Tensor lhs_rw = Tensor::unbroadcast(rhs * args[0], info[0].shape);
        Tensor rhs_rw = Tensor::unbroadcast(lhs * args[0], info[1].shape);

        return {lhs_rw.astype(info[0].type), rhs_rw.astype(info[1].type)};



    }




};