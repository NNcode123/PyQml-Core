#include <pyqml-core/bindings/Tensor.hpp>
#include <pyqml-core/bindings/Autograd/Node.hpp>

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



        std::vector<size_t> shape = start->info[0].shape;
        auto Type = start->info[0].type;

        buffer[start] = {Tensor::ones(shape, Type )};

        while (!tasks.empty()){

            Node* cur = tasks.front();
            tasks.pop();

            const auto& parents = cur->edges;

            
            
            /*
            if (cur->grad_info->retain_grad){
                Tensor& grad = cur->grad_info->grad;
                auto& gradients = buffer[cur];
                for (auto& tens: gradients){
                    grad += tens;
                }
            }
            */
            
            std::cout << "buffer_cur size is: " << buffer[cur].size() << "\n";

            auto grads = cur->backward(std::move(buffer[cur]));

            for (size_t i = 0; i < parents.size(); ++i ){
                auto edge = parents[i];
                
                auto node = edge.node_fn.storage_ptr();

                if (buffer.find(node) == buffer.end()){
                    buffer[node] = std::vector<Tensor>(node->info.size());
                }
                
                auto& parent_grads = buffer[node];

                std::cout << "Accumulating gradient from child " << i << " back to the parent, where child is the " << edge.input_nr +1 << " child of parent " << "\n";
                
                parent_grads[edge.input_nr] += grads[i];
                
                if (--dependencies[node] == 0 ) {
                    tasks.push(node);
                }



            }


        }


    }
    
};
