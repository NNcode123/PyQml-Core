#include "AccumulateGradNode.hpp"
#include "../../cpp/include/Storage/intrusive_ptr.hpp"
#include "../Tensor.hpp"


struct grad_helper{

void attach_grad(Tensor& t) const noexcept{


    pyq_intrusive_ptr<Node> node_l = make_intrusive<AccumulateGradNode>(std::vector<Tensor>{}, 
        std::vector<InputMetadata>{InputMetadata{.shape = t.shape(), .type = t.type()}}, t);
    t.get_info() = pyq_intrusive_ptr<grad_meta>(new grad_meta{
        nullptr, true, true, false, node_l
    });
 /* implementation */
 /*......*/
}

void requires_grad_(Tensor& t) const noexcept{
    if (t.grad_fn()){
        throw std::runtime_error("Cannot keep gradient of non-leaf Tensor through this method. Use retain_grad_ instead.");
    }
    attach_grad(t);
}

void retain_grad_(Tensor& t) const noexcept{
/**/
}



};
