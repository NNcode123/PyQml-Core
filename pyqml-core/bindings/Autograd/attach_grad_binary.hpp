
#include "Node.hpp"
#include "SubNode.hpp"
#include "AddNode.hpp"
#include "MulNode.hpp"
#include "DivNode.hpp"


#define Attach_Grad(Tensor, name, LHS, RHS)\
    auto lhs_info = InputMetadata{.shape = LHS.shape(), .type = LHS.type()};\
    auto rhs_info = InputMetadata{.shape = RHS.shape(), .type = RHS.type()};\
    std::vector<InputMetadata> node_info =  {lhs_info, rhs_info};\
    const auto& edges = std::vector<Edge>{\
        {.node_fn = LHS.node_fn()},\
        {.node_fn = RHS.node_fn()}\
    };\
    const auto& node_l = make_intrusive<name##Node>(edges, node_info ,LHS, RHS  );\
    Tensor.info = make_intrusive<grad_meta>(new grad_meta{.grad = nullptr,\
        .requires_grad = true, .is_leaf = false, .retain_grad = false\
       ,.node = node_l});\
       



