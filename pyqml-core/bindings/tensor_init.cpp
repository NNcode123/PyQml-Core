
#pragma once
#include "Tensor.hpp"
#include "PyType.hpp"

void bind_tensor(py::module &m)
{

    py::class_<Tensor>(m, "Tensor")
        .def(py::init([](py::list data,
                         py::list dim,
                         py::object type = py::none())
                      {

        DType dtype = type.is_none() ? DType::NoneType : type.cast<DType>();
    
        py::array vals = py::array(data);
        auto dt = vals.dtype();

        DType implied_type = inf_dtyp_python<int8_t,int16_t,int32_t,int64_t,float,double>(dt);

        DType res_type = type.is_none() ? implied_type: dtype;

        Tensor res =  Tensor::dispatch(res_type, [&](auto inst_t){
            using R = std::decay_t<decltype(inst_t)>;
           py::array_t<R, py::array::c_style | py::array::forcecast> new_val =
        vals.cast<py::array_t<R, py::array::c_style | py::array::forcecast>>();
            return Tensor(new_val, dim.cast<std::vector<size_t>>(), res_type );
        });
      
        return res; }),
             py::arg("data"),
             py::arg("dim"),
             py::arg("type") = py::none())

        .def("__add__", &Tensor::operator+)

        .def("__sub__", &Tensor::operator-)

        .def("__mul__", &Tensor::operator*)

        .def("__truediv__", &Tensor::operator/)

        .def("__repr__", &Tensor::print_val)

        .def("astype", [&](const Tensor &a, py::object q, py::bool_ u)
             {
            DType n_type = infer_types(q, false);
           bool u_n = u;
            return a.astype(n_type,u_n); }, py::arg("dtype"), py::arg("copy") = false)

        .def_property_readonly("dtype", &Tensor::type)

        .def_property_readonly("shape", &Tensor::shape);

    m.def("arange", [&](py::object start, py::object step, py::object end)
          {
            DType st_d = infer_types(start,true);
            DType st_s = infer_types(step,true);
            DType st_e = infer_types(end,true);
            DType max_type = std::max({st_d,st_s,st_e});
            Tensor res =  Tensor::dispatch(max_type, [&](auto val){
                using T = std::decay_t<decltype(val)>;
                T n_st = start.cast<T>();
                T n_ste = step.cast<T>();
                T n_end = end.cast<T>();
                return Tensor::arange(n_st, n_ste, n_end, max_type);
            });
            return res; });
    m.def("to_numpy", &Tensor::to_numpy);
    m.def("einsum", [&](const Tensor &a, const Tensor &b, py::object axes_a, py::object axes_b)
          { return einsum_(a, b, axes_a.cast<std::vector<int>>(), axes_b.cast<std::vector<int>>()); });
}
