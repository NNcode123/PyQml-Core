
#include "bindings.hpp"

namespace py = pybind11;

PYBIND11_MODULE(pyqmlcore, m)
{

    bind_tensor(m);
    bind_dtype(m);

    
}