# PyQml-Core
PyQml is a C++ backed Python module that supports efficient, scalable linear algebra operators for Quantum and Machine Learning(QML) focused projects.4

## Architecture 
```text
pyqml-core/
├── cpp/
│   ├── include/pyqml/
│   │   ├── matrix.hpp
│   │   ├── tensor.hpp
│   │   └── ml/
│   ├── src/
│   │   ├── matrix.cpp
│   │   ├── tensor.cpp
│   │   └── ml/
│
├── bindings/
│   └── pybind_module.cpp
│
├── python/pyqml/
│   │── __init__.py
│
├── tests/
└── README.md
```
is:

$$
\text{index} = i \cdot (d_2 d_3) + j \cdot d_3 + k
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

### Notes for Implementation

- Strides can be precomputed as:
  $$
  \text{stride}_k = \prod_{j=k+1}^{n} d_j
  $$
- Bounds checking should occur before computing the linear index
- Reshape operations must preserve:
  $$
  \prod d_{\text{old}} = \prod d_{\text{new}}
  $$

---