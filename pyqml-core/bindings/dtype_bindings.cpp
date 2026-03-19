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
        .export_values();

    m.attr("int8") = DType::Int8;
    m.attr("int16") = DType::Int16;
    m.attr("int32") = DType::Int32;
    m.attr("int64") = DType::Int64;
    // m.attr("int") = DType::Int;
    m.attr("float32") = DType::Float32;
    m.attr("float64") = DType::Float64;
}