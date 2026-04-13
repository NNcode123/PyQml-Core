
#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/complex.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "dtype.hpp"

namespace py = pybind11;

void bind_dtype(py::module_ &m);
void bind_tensor(py::module_ &m);
