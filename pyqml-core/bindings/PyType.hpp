#pragma once
#include "bindings.hpp"
template <typename... Ts>

DType inf_dtyp_python(py::dtype dt)
{
    DType result;
    bool found = false;
    ([&]()
     { if (!found && dt.is(py::dtype::of<Ts>())){
            result = typeDType<Ts>::type;
            found = true;
        } }(), ...);

    return result;
}

inline DType infer_types(py::object obj, bool obj_instance)
{
    if (obj_instance)
    {
        if (py::isinstance<py::int_>(obj))
        {
            return DType::Int64;
        }
        if (py::isinstance<py::float_>(obj))
        {
            return DType::Float64;
        }
    }
    else
    {
        if (py::isinstance<py::type>(obj))
        {
            py::object bins = py::module_::import("builtins");
            py::type Ty = py::cast<py::type>(obj);
            if (obj.is(bins.attr("int")))
            {
                return DType::Int64;
            }
            if (obj.is(bins.attr("float")))
            {
                return DType::Float64;
            }
        }
    }
    try
    {
        return py::cast<DType>(obj);
    }
    catch (const py::cast_error &e)
    {
        throw std::runtime_error("Unsupported object type");
    }
    throw std::runtime_error("Unsupported object type");
}