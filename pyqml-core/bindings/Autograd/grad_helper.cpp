#include <pyqml-core/bindings/Tensor.hpp>
#include <pyqml-core/bindings/Autograd/AccumulateGradNode.hpp>
#include <pyqml-core/bindings/Autograd/grad_helper.hpp>

void grad_helper::attach_grad(Tensor& t) noexcept
{

    
    pyq_intrusive_ptr<AccumulateGradNode> node_l = make_intrusive<AccumulateGradNode>(std::vector<Edge>{},
        std::vector<InputMetadata>{InputMetadata{.shape = t.shape(), .type = t.type()}}, nullptr);

    t.get_info() = pyq_intrusive_ptr<grad_meta>(new grad_meta{
        Tensor(), true, true, false, node_l
    });

    node_l->set_meta_ref(t.get_info());

 /* implementation */
 /*......*/
}


/*
void grad_helper::requires_grad_(Tensor& t) noexcept
{
    if (t.grad_fn()){
        throw std::runtime_error("Cannot keep gradient of non-leaf Tensor through this method. Use retain_grad_() instead.");
    }
    attach_grad(t);
}
*/

/*
void grad_helper::retain_grad_(Tensor& t) noexcept
{
}
*/