project(tensor)
cmake_minimum_required(VERSION 2.8)

option(BUILD_PYTHON_EXTENSION "Build the Python module" OFF)
option(BUILD_TESTS "Build the tests" ON)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR})

add_subdirectory(src)

if(BUILD_PYTHON_EXTENSION)
    message(STATUS "Building Python module")
    add_subdirectory(py_tensor)
endif(BUILD_PYTHON_EXTENSION)

if(BUILD_TESTS)
    message(STATUS "Building test program")
    add_subdirectory(tests)
endif(BUILD_TESTS)

