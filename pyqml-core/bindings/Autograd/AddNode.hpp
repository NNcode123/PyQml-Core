#include "Node.hpp"
#include "Tensor.hpp"

struct AddNode: Node {
   

    AddNode(std::vector<Edge>&& func): Node(std::move(func)) {}


    std::vector<Tensor> backward(std::vector<Tensor> && grads) const noexcept override{

        
        /*

        given Node(a,b) = c

        This function will recieve dLeaf/dc in vector of tensor Then compute local dLeaf/da = J_a^T @ dLeaf/dc , dLeaf/db = J_b^T @ dLeaf/da 
        where J_b = dc/db, J_a = dc/da

        Then func will return {dLeaf/da, dLeaf/db}

        */

        /*
        

        Tensor t;
        Tensor v; 



        return {a.grad, b.grad};

        */

        return {Tensor::unbroadcast(grads[0], info[0].shape ), Tensor::unbroadcast(grads[0], info[0].shape)};


        
        
   
    }
};

