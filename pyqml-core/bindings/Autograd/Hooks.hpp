#include <vector>
#include <functional>
class Tensor;

struct FunctionPreHooks{

    public:

    virtual void operator()(std::vector<Tensor>& grad, size_t output_nr = 0)  = 0;


};


struct FunctionPostHooks{

    public:

    virtual void operator()(std::vector<Tensor>& grad, const std::vector<Tensor>& incoming_grad)  = 0;

};

struct CppFunctionPreHook: FunctionPreHooks {

    using hook = std::function<Tensor(std::vector<Tensor>&)>;

    hooks = std::unqiue_



}