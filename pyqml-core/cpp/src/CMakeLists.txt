cmake_minimum_required(VERSION 3.16)

project(PyQmlCore LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -mtune=native -DNDEBUG")

find_package(OpenMP REQUIRED)

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

include_directories(${PROJECT_SOURCE_DIR}/cpp/include)

# ---------------- TEST ----------------
add_executable(test test.cpp)
target_link_libraries(test PRIVATE OpenMP::OpenMP_CXX)

# ---------------- PYTHON MODULE ----------------
pybind11_add_module(pyqmlcore
    ${CMAKE_CURRENT_SOURCE_DIR}/../../bindings/pybind_module.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../../bindings/tensor_init.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../../bindings/dtype_bindings.cpp
)

target_link_libraries(pyqmlcore PRIVATE OpenMP::OpenMP_CXX)

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
