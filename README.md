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

## Building from Source

These instructions assume you are building from the repository root and that the source tree matches the repository layout shown above.

### Prerequisites
- CMake version 3.16 or newer.
- A C++20 compatible compiler.
- A C17 compatible C compiler.
- Python 3.x with development headers.
- `pybind11` installed.
- NVIDIA CUDA Toolkit with the `nvcc` compiler installed.
- OpenMP is optional. The project can build without OpenMP if it is unavailable.

### Configure CMake
From the repository root, create or enter the build folder under `cpp/src/build`:

```bash
cd cpp/src/build
```

Configure the project with CMake:

```bash
cmake -S . -B .
```

- `-S .` tells CMake to use the current directory as the source tree.
- `-B .` tells CMake to write build files into the current directory.

### Build the project
Build the project in Release mode:

```bash
cmake --build . --config Release
```

- `cmake --build .` builds the generated project files using the selected generator.
- `--config Release` requests the Release configuration for multi-configuration generators such as Visual Studio.

### Python installation
After a successful build, the generated `pyqmlcore` extension module is automatically copied into the detected Python environment's `site-packages` directory by the CMake build script.

On Windows this will typically be a `.pyd` file; on Linux or macOS it will be the platform-specific shared library file.

### Verify the installation
Open Python and run:

```python
import pyqmlcore
```

If the import succeeds, the build and installation completed successfully.

### Troubleshooting
- `pybind11` not found
  - Install `pybind11` and ensure CMake can locate it.
  - You can override the path with `-Dpybind11_DIR=/path/to/pybind11/share/cmake/pybind11`.
- Wrong Python interpreter
  - Use `-DPython_EXECUTABLE=/path/to/python` to select the desired interpreter.
- CUDA not found
  - Install the CUDA Toolkit and make sure `nvcc` is available on your `PATH`.
  - If detection fails, add `-DCUDA_TOOLKIT_ROOT_DIR=/path/to/cuda`.
- Compiler does not support C++20
  - Use a newer compiler or set `-DCMAKE_CXX_COMPILER=/path/to/clang++` or the appropriate compiler path.
- OpenMP not found
  - OpenMP support is optional; the project can still build without it.
- Stale CMake cache
  - Delete `CMakeCache.txt` inside `cpp/src/build` or remove the build folder and rerun configuration.

### CMake variables to override
If automatic detection fails, pass one or more of these variables to CMake:

```bash
cmake -S . -B . \
  -DPython_EXECUTABLE=/path/to/python \
  -Dpybind11_DIR=/path/to/pybind11/share/cmake/pybind11 \
  -DCUDA_TOOLKIT_ROOT_DIR=/path/to/cuda \
  -DCMAKE_CXX_COMPILER=/path/to/clang++ \
  -DCMAKE_C_COMPILER=/path/to/gcc
```

### Notes
- The project currently targets C++20.
- CUDA (NVCC) must be installed for CUDA-enabled builds.
- This guide avoids hard-coded personal paths and is written for generic repository layouts.

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
