
#include "bindings.hpp"

namespace py = pybind11;

// This file is the module entry point for the pybind11 extension and exposes the
// native tensor and dtype bindings to Python as part of the pyqmlcore package.
PYBIND11_MODULE(pyqmlcore, m)
{

    bind_tensor(m);
    bind_dtype(m);
}
