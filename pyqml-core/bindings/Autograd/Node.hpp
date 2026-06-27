// This header defines the base autograd node abstraction that higher-level gradient
// graph classes can build on when representing tensor operations.
#include "../Tensor.hpp"

struct Node
{

    std::vector<Tensor *> parents;

    virtual void backward() const = 0;
};
