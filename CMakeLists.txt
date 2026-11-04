cmake_minimum_required(VERSION 4.4)

project(llama2 LANGUAGES CXX)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()

option(LLAMA2_BUILD_TESTS
    "Build the core tests"
    OFF
)
option(LLAMA2_ENABLE_CUDA
    "Enable CUDA backend"
    OFF
)
if(LLAMA2_ENABLE_CUDA)
    message(FATAL_ERROR
        "CUDA backend is not implemented yet."
    )
endif()

find_package(OpenMP REQUIRED COMPONENTS CXX)

add_library(llama2_cpu_core STATIC
    # CPU Kernels
    src/kernel/cpu/matvec.cpp
    src/kernel/cpu/rmsnorm.cpp
    src/kernel/cpu/rope.cpp
    src/kernel/cpu/softmax.cpp
    src/kernel/cpu/swiglu.cpp
    src/kernel/cpu/attention.cpp
    # Operators
    src/ops/matvec.cpp
    src/ops/rmsnorm.cpp
    src/ops/softmax.cpp
    src/ops/rope.cpp
    src/ops/swiglu.cpp
    src/ops/attention.cpp
)

add_library(llama2::cpu_core ALIAS llama2_cpu_core)
target_include_directories(llama2_cpu_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_compile_features(llama2_cpu_core PUBLIC cxx_std_26)
target_link_libraries(llama2_cpu_core PRIVATE
    OpenMP::OpenMP_CXX
)

if(MSVC)
    target_compile_options(llama2_cpu_core PRIVATE
        /W4
    )
else()
    target_compile_options(llama2_cpu_core PRIVATE
        -Wall
        -Wextra
        -Wpedantic
    )
endif()

if(LLAMA2_BUILD_TESTS)
    enable_testing()
endif()