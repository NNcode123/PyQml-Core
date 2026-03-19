cmake_minimum_required(VERSION 3.16)
project(PyQmlCore LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -mtune=native -DNDEBUG")

find_package(OpenMP REQUIRED)

# ✅ FIXED Python + pybind
set(PYBIND11_FINDPYTHON ON)
find_package(Python REQUIRED COMPONENTS Interpreter Development)

set(CMAKE_PREFIX_PATH "C:/Users/nitro/AppData/Local/Programs/Python/Python313/Lib/site-packages/pybind11/share/cmake/pybind11")
find_package(pybind11 REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/cpp/include)

add_executable(test
    src/test.cpp
)


target_link_libraries(test PRIVATE OpenMP::OpenMP_CXX)


pybind11_add_module(pyqmlcore
     ${CMAKE_CURRENT_SOURCE_DIR}/../bindings/pybind_module.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/../bindings/tensor_init.cpp
       ${CMAKE_CURRENT_SOURCE_DIR}/../bindings/dtype_bindings.cpp
     
)

target_link_libraries(pyqmlcore PRIVATE OpenMP::OpenMP_CXX)
