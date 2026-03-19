

#include "dtype.hpp"
#include "bindings.hpp"

namespace py = pybind11;

void bind_tensor(py::module &m)
{

    py::class_<Tensor>(m, "Tensor")
        .def(py::init([](py::list data,
                         py::list dim,
                         py::object type = py::none())
                      {

        DType dtype;
        if (type.is_none()) {
            dtype = DType::Float32;
        } else {
            dtype = type.cast<DType>();
        }
        

        switch (dtype) {
            case DType::Int8:
                return Tensor(
                    tensor<int8_t>(data.cast<std::vector<int8_t>>(), dim.cast<std::vector<size_t>>()), dtype
                );

            case DType::Int16:
                return Tensor(
                    tensor<int16_t>(data.cast<std::vector<int16_t>>(), dim.cast<std::vector<size_t>>()),
                    dtype                );

            case DType::Int32:
                return Tensor(
                    tensor<int32_t>(data.cast<std::vector<int32_t>>(), dim.cast<std::vector<size_t>>()),
                    dtype
                );

            case DType::Int64:
                return Tensor(
                    tensor<int64_t>(data.cast<std::vector<int64_t>>(), dim.cast<std::vector<size_t>>()),
                    dtype
                );

                /*

            case DType::Int:
                return Tensor(
                    tensor<int>(data.cast<std::vector<int>>(), dim.cast<std::vector<size_t>>().cast<std::vector<size_t>>()),
                    dtype
                );
                */

            case DType::Float32:
                return Tensor(
                    tensor<float>(data.cast<std::vector<float>>(), dim.cast<std::vector<size_t>>()),
                    dtype
                );

            case DType::Float64:
                return Tensor(
                    tensor<double>(data.cast<std::vector<double>>(), dim.cast<std::vector<size_t>>()),
                    dtype
                );

            default:
                throw std::runtime_error("Unsupported dtype");
        } }),
             py::arg("data"),
             py::arg("dim"),
             py::arg("type") = py::none())

        .def("__add__", [](const Tensor &a, const Tensor &b)
             { return Tensor(a) + Tensor(b); })
        .def("__sub__", [](const Tensor &a, const Tensor &b)
             { return Tensor(a) - Tensor(b); })

                .def("__repr__", [](const Tensor &a)

             {  std::stringstream out;
                 return Tensor(a).print_val(out); });
    //.def("dtype", )
}
