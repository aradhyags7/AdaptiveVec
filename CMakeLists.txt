cmake_minimum_required(VERSION 3.16)
project(AdaptiveVec VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(ENABLE_AVX2 "Enable AVX2/FMA hardware SIMD acceleration" ON)
option(ENABLE_OPENMP "Enable OpenMP multi-threading" ON)
option(BUILD_TESTS "Build native C++ test suite" ON)
option(BUILD_BENCHMARKS "Build native benchmark runner" ON)
option(BUILD_PYTHON_BINDINGS "Build Python pybind11 extension module" ON)

# Optimization and compiler warnings
if(MSVC)
    add_compile_options(/O2 /W4 /bigobj)
    if(ENABLE_AVX2)
        add_compile_options(/arch:AVX2)
    endif()
    if(ENABLE_OPENMP)
        find_package(OpenMP)
        if(OpenMP_CXX_FOUND)
            add_compile_options(${OpenMP_CXX_FLAGS})
        endif()
    endif()
else()
    add_compile_options(-O3 -Wall -Wextra)
    if(ENABLE_AVX2)
        add_compile_options(-mavx2 -mfma)
    endif()
    if(ENABLE_OPENMP)
        find_package(OpenMP)
        if(OpenMP_CXX_FOUND)
            add_compile_options(${OpenMP_CXX_FLAGS})
        endif()
    endif()
endif()

# Core interface library
add_library(adaptivevec_cpp INTERFACE)
target_include_directories(adaptivevec_cpp INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/hnswlib
)
if(OpenMP_CXX_FOUND)
    target_link_libraries(adaptivevec_cpp INTERFACE OpenMP::OpenMP_CXX)
endif()

# Benchmark Target
if(BUILD_BENCHMARKS)
    add_executable(benchmark_runner cpp/benchmark_main.cpp)
    target_link_libraries(benchmark_runner PRIVATE adaptivevec_cpp)
    set_target_properties(benchmark_runner PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/cpp
    )
endif()

# Test Suite Targets
if(BUILD_TESTS)
    enable_testing()
    add_executable(native_tests cpp/tests/test_main.cpp)
    target_link_libraries(native_tests PRIVATE adaptivevec_cpp)
    set_target_properties(native_tests PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/cpp/tests
    )
    add_test(NAME NativeTestSuite COMMAND native_tests WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

    set(ADDITIONAL_CPP_TESTS
        test_auto_calibrator:cpp/tests/test_auto_calibrator.cpp
        test_numerical_correctness:cpp/tests/test_numerical_correctness.cpp
        test_runtime_safety:cpp/tests/test_runtime_safety.cpp
        test_concurrent_mutation_preservation:cpp/test_concurrent_mutation_preservation.cpp
        test_serving_manager_concurrency:cpp/test_serving_manager_concurrency.cpp
        test_sq4_correctness:cpp/test_sq4_correctness.cpp
    )
    foreach(entry ${ADDITIONAL_CPP_TESTS})
        string(REPLACE ":" ";" parts "${entry}")
        list(GET parts 0 tgt_name)
        list(GET parts 1 src_path)
        add_executable(${tgt_name} ${src_path})
        target_link_libraries(${tgt_name} PRIVATE adaptivevec_cpp)
        if(WIN32)
            target_link_libraries(${tgt_name} PRIVATE psapi)
        endif()
        set_target_properties(${tgt_name} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests
        )
        add_test(NAME ${tgt_name} COMMAND ${tgt_name} WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
    endforeach()
endif()

# pybind11 Python Module
if(BUILD_PYTHON_BINDINGS)
    find_package(pybind11 QUIET)
    if(pybind11_FOUND)
        pybind11_add_module(_adaptivevec_cpp cpp/bindings.cpp)
        target_link_libraries(_adaptivevec_cpp PRIVATE adaptivevec_cpp)
        set_target_properties(_adaptivevec_cpp PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/adaptivevec
        )
    else()
        message(STATUS "pybind11 not found via CMake find_package; python bindings can be compiled directly via g++ or setup.py")
    endif()
endif()
