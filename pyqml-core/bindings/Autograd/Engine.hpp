#include "Tensor.hpp"
#include "Node.hpp"
#include "../grad_meta.hpp"

struct Engine {

    std::unordered_map<Node*, std::vector<Tensor>> buffer;

    std::unordered_map<Node*, size_t> dependencies; 
    
    std::unordered_set<Node*> seen;
    
    std::vector<Node*> graph_nodes;

    std::queue<Node*> tasks;

    void build_topo(Node* start){

        if (!start) 
            return;

        auto& parents = start->edges;

        

        seen.insert(start);

        for (auto && val: parents){
            auto parent = val.node_fn.storage_ptr();
            if (!seen.count(parent)){
                build_topo(parent);
            }
            
        }

        graph_nodes.push_back(start);

    }

    void compute_dependencies(Node* node)
{
    if (!node || seen.count(node))
        return;

    seen.insert(node);

    for(auto& edge : node->edges)
    {
        Node* parent = edge.node_fn.storage_ptr();

        dependencies[parent]++;

        compute_dependencies(parent);
    }
}

    void backward(Node* start){
        compute_dependencies(start);

        for (auto& [node, incoming_grad]: buffer){
            incoming_grad.resize(node->info.size());
        }

        tasks.push(start);



        std::vector<size_t> shape = {3,4,5};

        buffer[start] = {Tensor::ones(shape,DType::Int32 )};

        while (!tasks.empty()){

            Node* cur = tasks.front();
            tasks.pop();

            const auto& parents = cur->edges;

            
            
            if (cur->grad_info->retain_grad){
                Tensor& grad = cur->grad_info->grad;
                auto& gradients = buffer[cur];
                for (auto& tens: gradients){
                    grad += tens;
                }
            }
            

            auto grads = cur->backward(std::move(buffer[cur]));

            for (size_t i = 0; i < parents.size(); ++i ){
                auto edge = parents[i];
                
                auto Node = edge.node_fn.storage_ptr();
                
                auto& parent_grads = buffer[Node];
                
                parent_grads[edge.input_nr] += grads[i];
                
                if (--dependencies[Node] == 0) {
                    tasks.push(Node);
                }



            }


        }


    }
    
};

