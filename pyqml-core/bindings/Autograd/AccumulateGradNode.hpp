#include <pyqml-core/bindings/Autograd/Node.hpp>
#include <pyqml-core/bindings/Tensor.hpp>
#include <iostream>


struct AccumulateGradNode: public Node{

    weak_intrusive_ptr<grad_meta> meta_ref;

    AccumulateGradNode(std::vector<Edge>&& func, std::vector<InputMetadata>&& info,  const pyq_intrusive_ptr<grad_meta>& meta): Node(std::move(func), std::move(info)), meta_ref(meta) {}

    void set_meta_ref(const pyq_intrusive_ptr<grad_meta>& meta) noexcept
    {
        meta_ref = weak_intrusive_ptr<grad_meta>(meta);
    }

    std::vector<Tensor> backward(std::vector<Tensor>&& args){
        std::cout << "args_size" << args.size() << "\n";
        
        auto meta = meta_ref.lock();
        
        auto& grad_m = meta->grad;
        //std::cout << grad_m.print_val() << std::endl;
        
        std::cout << "args: " << args[0].print_val() << "\n";
        
        grad_m += args[0];

        
        std::cout << grad_m.print_val() << "\n";

        return {};
    }

    /*

    private: 
    
    ~AccumulateGradNode() override{

    }
    */

};  