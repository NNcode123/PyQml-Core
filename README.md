# PyQml-Core
PyQml is a C++ backed Python module that supports efficient, scalable linear algebra operators for Quantum and Machine Learning(QML) focused projects.

## Architecture
```text
PyQml-Core/
├── README.md
└── pyqml-core/
    ├── bindings/
    │   ├── Autograd/
    │   ├── Tensor_ops/
    │   ├── bindings.hpp
    │   ├── dtype.hpp
    │   ├── dtype_bindings.cpp
    │   ├── pybind_module.cpp
    │   ├── PyType.hpp
    │   ├── tensor_init.cpp
    │   └── Tensor.hpp
    ├── cpp/
    │   └── src/
    │       ├── tensor.hpp
    │       ├── tensor_imp_files/
    │       ├── test.cpp
    │       └── thread/
    └── tests/
        ├── func_test.py
        ├── libe.py
        └── test.py
```

The package is organized into three layers:
- bindings/: the pybind11 bridge that exposes the native tensor engine to Python.
- cpp/src/: the core tensor implementation, including slicing, broadcasting, reductions, and thread helpers.
- tests/: small validation and benchmarking scripts used to exercise the module from Python.


### Memory Layout 

The tensor uses **row-major (C-style)** memory layout, meaning indices are mapped linearly as:

$$
\text{index(i,j,k)} = i \cdot (d_2 d_3) + j \cdot d_3 + k
$$

---

### General Interpretation

- Earlier indices have **larger strides**
- The **last dimension is contiguous**
- This matches:
  - NumPy default layout
  - C / C++ arrays
  - Most ML frameworks

---

### Why Row-Major?

- Cache-friendly
- Interoperable with NumPy
- Simple stride computation
- Natural fit for variadic indexing operators

---

### Benchmarking 

Basic operations, including the standard elementwise operations and contraction/einsum routines, were benchmarked against canonical Numpy implementations. 

Results demonstrated that this implementation is in the same performance neighborhood as Numpy for basic routines. While Numpy remains highly optimized for more complex pipelines, my current design supports a strong baseline for performance and correctness.

### Benchmark Results 

---

Performance visualizations can be found in:

```text
tests/results.png
```


This file contains plots comparing runtime across different tensor sizes and operations.
