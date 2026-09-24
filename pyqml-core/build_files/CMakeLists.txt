cmake_minimum_required(VERSION 3.16)

project(PyQmlCore LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS_RELEASE "/O2 /GL /LTCG /arch:AVX2 /fp:fast /openmp /DNDEBUG /MP")

find_package(OpenMP)

# ---------------- PYTHON ----------------
set(PYBIND11_FINDPYTHON ON)
find_package(Python REQUIRED COMPONENTS Interpreter Development)

execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import site; print(site.getsitepackages()[0])"
    OUTPUT_VARIABLE PYTHON_SITE
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

message(STATUS "Python site-packages: ${PYTHON_SITE}")

# ---------------- PYBIND11 ----------------
set(CMAKE_PREFIX_PATH
    "C:/Users/nitro/AppData/Local/Programs/Python/Python313/Lib/site-packages/pybind11/share/cmake/pybind11"
)
find_package(pybind11 REQUIRED)

set(PYQML_CORE_DIR ${PROJECT_SOURCE_DIR}/..)
set(BINDINGS_DIR ${PYQML_CORE_DIR}/bindings)
set(CPP_DIR ${PYQML_CORE_DIR}/cpp)

include_directories(${CPP_DIR})

# ---------------- TEST ----------------
add_executable(test ${CPP_DIR}/src/test.cpp)
add_executable(test_2 ${BINDINGS_DIR}/tens_test.cpp)
target_link_libraries(test_2 PRIVATE pybind11::module)

if(OpenMP_CXX_FOUND)
    target_link_libraries(test PRIVATE OpenMP::OpenMP_CXX)
endif()

# ---------------- PYTHON MODULE ----------------
pybind11_add_module(pyqmlcore
    ${BINDINGS_DIR}/pybind_module.cpp
    ${BINDINGS_DIR}/tensor_init.cpp
    ${BINDINGS_DIR}/binary_ops.cpp
    ${BINDINGS_DIR}/unary_ops.cpp
    ${BINDINGS_DIR}/dtype_bindings.cpp
    ${BINDINGS_DIR}/grad_utils.cpp
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(pyqmlcore PRIVATE OpenMP::OpenMP_CXX)
endif()

# ---------------- CLEAN OLD FILES ----------------
add_custom_command(TARGET pyqmlcore PRE_BUILD
    COMMAND ${CMAKE_COMMAND} -E rm -f
        "${PYTHON_SITE}/pyqmlcore.pyd"
        "${PYTHON_SITE}/pyqmlcore.lib"
        "${PYTHON_SITE}/pyqmlcore.exp"
        "${PYTHON_SITE}/pyqmlcore.pdb"
        "${PYTHON_SITE}/pyqmlcore.ilk"
)

# ---------------- COPY AFTER BUILD ----------------
add_custom_command(TARGET pyqmlcore POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
        $<TARGET_FILE:pyqmlcore>
        "${PYTHON_SITE}"

    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_PDB_FILE:pyqmlcore>
        "${PYTHON_SITE}"
)
