// This header defines the base autograd node abstraction that higher-level gradient
// graph classes can build on when representing tensor operations.
#include <vector>
#include "../../Storage/intrusive_ptr.hpp"
#include <unordered_map>
#include <queue>
#include <unique_ptr>




//Hash out the specifics later mkay great let's roll.

struct Edge{

    pyq_intrusive_ptr<Node> node_fn;

    size_t int_nr = 0; 

};



class Tensor;

struct Node: public refcount
{

    // edges represents the Nodes of the parent Tensors that produced this child Tensor's Nodes 
    std::vector<Edge> edges;

    Node(std::vector<Edge>&& edge_val): edges(std::move(edge_val)) {}

    virtual std::vector<Tensor> backward(std::vector<Tensor>&& tensor_input) const = 0;
};

struct grad_meta{

    std::unique_ptr<Tensor> grad;
    bool requires_grad = false;
    bool is_lef = false;
    bool retain_grad = false;

};






struct Engine {

    std::unordered_map<Node*, std::vector<Tensor>> buffer;

    std::unordered_map<Node*, size_t> dependencies; 

    std::queue<Node*> tasks;
    
};



*/