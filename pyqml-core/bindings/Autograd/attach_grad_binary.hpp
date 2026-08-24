#include "../Tensor.hpp"
#include "Node.hpp"

#define Attach_Grad(Tensor, name, LHS, RHS)\
    auto& lhs_info = InputMetadata{.shape = LHS.shape(), .type = RHS.type()};\
    auto& rhs_info = InputMetadata{.shape = RHS.shape(), .type = RHS.type()};\
    auto& node_info = {lhs_info, rhs_info};\
    auto& edges = std::vector<Edge>{\
        {.node_fn = LHS.node_fn()},\
        {.node_fn = RHS.node_fn()}\
    };\
    auto node_l = make_intrusive<name##Node>(edges, node_info ,LHS, RHS  );\
    Tensor.info = make_intrusive<grad_meta>(new grad_meta{.grad = nullptr,\
        .requires_grad = true, .is_leaf = false, .retain_grad = LHS.requires_grad() | \
        RHS.requires_grad(),.node = node_l});\
       



