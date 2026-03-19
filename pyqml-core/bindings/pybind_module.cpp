#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "../cpp/src/tensor.hpp"
#include "bindings.hpp"

namespace py = pybind11;

PYBIND11_MODULE(pyqmlcore, m)
{

    bind_tensor(m);
    bind_dtype(m);

    /*

    py::class_<tensor<double>>(m, "TensorDouble")

        .def(py::init<std::vector<double>, std::vector<size_t>>())
        .def("data", &tensor<double>::data)
        .def("__add__", [](const tensor<double> &a, const tensor<double> &b)
             { return a + b; });

    /*
py::class_<tensor<int8_t>>(m, "TensorFloat16")
 .def(py::init<std::vector<int8_t>, std::vector<size_t>>());
 */
}