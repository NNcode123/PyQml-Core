#include "Storage/intrusive_ptr.hpp"
#include "Tensor.hpp"

struct grad_meta{

    Tensor grad;

    bool requires_grad = false;

    bool is_leaf = false;

    bool retain_grad = false;

    pyq_intrusive_ptr<Node> node;

};

