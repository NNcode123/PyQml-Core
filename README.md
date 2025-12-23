# PyQml-Core
PyQml is a C++ backended Python module that supports efficient, scalable linear algebra operators for Quantum and Machine Learning(QML) focused projects.4

### Test Modification
This is a *test* to see if the seperate branches work

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
