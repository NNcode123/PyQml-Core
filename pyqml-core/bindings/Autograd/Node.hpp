// This header defines the base autograd node abstraction that higher-level gradient
// graph classes can build on when representing tensor operations.
#pragma once
#include <vector>
#include <unordered_map>
#include <pyqml-core/bindings/Autograd/Edge.hpp>
#include <pyqml-core/bindings/dtype.hpp>
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

    

    // edges represents the Nodes of the parent Tensors that produced this child Tensor's Nodes 
   

    Node(const Node&) = delete;

    Node& operator=(const Node&) = delete;

    Node(Node&&) = delete;

    Node& operator=(Node&&) = delete;
    
    Node(std::vector<Edge>&& edge_val, std::vector<InputMetadata>&& info): edges(std::move(edge_val)), info(std::move(info)) {}

    virtual std::vector<Tensor> backward(std::vector<Tensor>&& tensor_input) = 0;

    void release_node_and_neighbors(Node* func){

        if (!func) {return;}

        for (auto& edge: func->edges){

            auto& ptr = edge.node_fn;

            release_node_and_neighbors(ptr.storage_ptr());

            ptr.reset();    




        }

    }

    /*
    const std::vector<Edge>& const_next_edge() const;
    std::vector<Edge>& next_edge() const;
    const std::vector<InputMetadata>& input_metadata_() const;
    std::vector<InputMetadata>& input_metadata_() const;
    */

    ~Node() override {


        std::cout << "Destroying node at address" << this << "with ref_count" << strong_ref_cnt(); 


        
        release_node_and_neighbors(this);
        

        release_resources();
        

    }

    void release_resources(){
        
        info.clear();
    }

    


    //protected: 

    std::vector<Edge> edges;

    std::vector<InputMetadata> info;

    

    //std::vector<std::unique_ptr<FunctionPostHook>> post_hooks;

    //std::vector<std::unique_ptr<FunctionPreHook>> pre_hooks; 

    




};





