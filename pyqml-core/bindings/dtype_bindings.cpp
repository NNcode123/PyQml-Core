#include "bindings.hpp"
#include "dtype.hpp"

namespace py = pybind11;

void bind_dtype(py::module_ &m)
{
    py::enum_<DType>(m, "dtype")
        .value("int8", DType::Int8)
        .value("int16", DType::Int16)
        .value("int32", DType::Int32)
        .value("int64", DType::Int64)
        //.value("int", DType::Int)
        .value("float32", DType::Float32)
        .value("float64", DType::Float64)
        .export_values()
        .def("__repr__", [](DType d)
             {
        switch (d) {
            case DType::Int8: return "int8";
            case DType::Int16: return "int16";
            case DType::Int32: return "int32";
            case DType::Int64: return "int64";
            case DType::Float32: return "float32";
            case DType::Float64: return "float64";
        } });
}