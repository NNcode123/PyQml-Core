
#include <pyqml-core/bindings/Autograd/Node.hpp>
#include <pyqml-core/bindings/Autograd/SubNode.hpp>
#include <pyqml-core/bindings/Autograd/AddNode.hpp>
#include <pyqml-core/bindings/Autograd/MulNode.hpp>
#include <pyqml-core/bindings/Autograd/DivNode.hpp>




#define Attach_Grad(Tensor, name, LHS, RHS)\
    auto lhs_info = InputMetadata{.shape = LHS.shape(), .type = LHS.type()};\
    auto rhs_info = InputMetadata{.shape = RHS.shape(), .type = RHS.type()};\
    std::vector<InputMetadata> node_info =  {lhs_info, rhs_info};\
    std::vector<Edge> edges;\
    if (LHS.grad_fn())\
        edges.push_back({LHS.grad_fn(), 0});\
    if (RHS.grad_fn())\
        edges.push_back({RHS.grad_fn(), 0});\
    pyq_intrusive_ptr<Node> node_l;\
    using NodeType = name##Node;\
    if constexpr (std::is_same_v<NodeType, SubNode> || std::is_same_v<NodeType, AddNode>){\
        node_l = make_intrusive<NodeType>(std::move(edges), std::move(node_info));\
    }\
    else{\
        node_l = make_intrusive<NodeType>(std::move(edges), std::move(node_info), LHS, RHS);\
    }\
    Tensor.info = pyq_intrusive_ptr<grad_meta>(new grad_meta{ /*.grad =*/  nullptr,\
        /*.requires_grad =*/ true, /*.is_leaf = */ false, /*.retain_grad = */ false\
       , /*.node = */ node_l});\







