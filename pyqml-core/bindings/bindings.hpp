
#pragma once
// This header centralizes the pybind11 declarations used across the binding layer so the
// different extension modules can share a consistent interface and namespace setup.
#include <pybind11/pybind11.h>
#include <pybind11/complex.h>
#include <pybind11/stl.h>
#include <pybind11/pytypes.h>
#include <pybind11/numpy.h>
#include "dtype.hpp"

namespace py = pybind11;

void bind_dtype(py::module_ &m);
void bind_tensor(py::module_ &m);
void bind_constants(py::module_ &m);
