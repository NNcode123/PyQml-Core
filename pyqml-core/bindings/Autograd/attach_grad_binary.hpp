
#include "Node.hpp"
#include "SubNode.hpp"
#include "AddNode.hpp"
#include "MulNode.hpp"
#include "DivNode.hpp"




#define Attach_Grad(Tensor, name, LHS, RHS)\
    auto lhs_info = InputMetadata{.shape = LHS.shape(), .type = LHS.type()};\
    auto rhs_info = InputMetadata{.shape = RHS.shape(), .type = RHS.type()};\
    std::vector<InputMetadata> node_info =  {lhs_info, rhs_info};\
    auto edges = std::vector<Edge>{\
        {.node_fn = LHS.grad_fn()},\
        {.node_fn = RHS.grad_fn()}\
    };\
    pyq_intrusive_ptr<Node> node_l;\
    using NodeType = name##Node;  \
    if constexpr (std::is_same_v<NodeType, SubNode> || std::is_same_v<NodeType, AddNode>){\
        node_l = make_intrusive<NodeType>(std::move(edges), std::move(node_info));\
    }\
    else{\
        node_l = make_intrusive<NodeType>(std::move(edges), std::move(node_info), LHS, RHS);\
    }\
    Tensor.info = pyq_intrusive_ptr<grad_meta>(new grad_meta{ /*.grad =*/  nullptr,\
        /*.requires_grad =*/ true, /*.is_leaf = */ false, /*.retain_grad = */ false\
       , /*.node = */ node_l});\







