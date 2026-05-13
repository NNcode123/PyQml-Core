
cmake_minimum_required(VERSION 3.16)

project(PyQmlCore LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS_RELEASE "/O2 /GL /DNDEBUG")

# 🔥 make OpenMP optional (fixes your error)
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

find_package(OpenBLAS REQUIRED)



# Allow Easy Includes for Files
include_directories(${PROJECT_SOURCE_DIR}/cpp)


# ---------------- TEST ----------------
add_executable(test test.cpp)

# only link OpenMP if it exists
if(OpenMP_CXX_FOUND)
    target_link_libraries(test PRIVATE OpenMP::OpenMP_CXX)
endif()

set(BUILD_LIB_DIR ${PROJECT_SOURCE_DIR}/bindings)

# ---------------- PYTHON MODULE ----------------
pybind11_add_module(pyqmlcore
    ${BUILD_LIB_DIR}/*.cpp
)

# only link OpenMP if available (IMPORTANT)
if(OpenMP_CXX_FOUND)
    target_link_libraries(pyqmlcore PRIVATE OpenMP::OpenMP_CXX)
endif()

target_link_libraries(pyqmlcore PRIVATE OpenBLAS::OpenBLAS)

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


