// This header defines the base autograd node abstraction that higher-level gradient
// graph classes can build on when representing tensor operations.
#include <vector>

#include <unordered_map>
#include "Edge.hpp"
#include "../dtype.hpp"
#include <unordered_set>
#include <queue>


class Tensor;
class Edge;
class grad_meta;
class FunctionPreHook;
class FunctionPostHook;


struct InputMetadata{

    std::vector<size_t> shape;

    DType type;

};



struct Node: public refcount
{

    // std::vector<Hook> pre_hooks;

    // edges represents the Nodes of the parent Tensors that produced this child Tensor's Nodes 
    std::vector<Edge> edges;

    std::vector<InputMetadata> info;

    grad_meta* grad_info = nullptr;

    Node(std::vector<Edge>&& edge_val, std::vector<InputMetadata>&& info): edges(std::move(edge_val)), info(std::move(info)) {}

    virtual std::vector<Tensor> backward(std::vector<Tensor>&& tensor_input) const = 0;

    // std::vector<Hook> post_hooks;



};







