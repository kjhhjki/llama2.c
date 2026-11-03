cmake_minimum_required(VERSION 4.4)

project(llama2 LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 23)

option(LLAMA2_BUILD_TESTS "Build the core smoke test" OFF)
option(LLAMA2_ENABLE_CUDA "Enable the CUDA conditional compilation path" OFF)

find_package(OpenMP REQUIRED COMPONENTS CXX)

add_library(llama2_cpu_core STATIC
    src/kernel/cpu/matvec.cpp
    src/kernel/cpu/rmsnorm.cpp
    src/kernel/cpu/rope.cpp
    src/kernel/cpu/softmax.cpp
    src/ops/matvec.cpp
    src/ops/rmsnorm.cpp
    src/ops/softmax.cpp
)
target_compile_features(llama2_cpu_core PUBLIC cxx_std_23)
target_include_directories(llama2_cpu_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_definitions(llama2_cpu_core PUBLIC CPU)
target_link_libraries(llama2_cpu_core PUBLIC OpenMP::OpenMP_CXX)

if (LLAMA2_ENABLE_CUDA)
    target_compile_definitions(llama2_cpu_core PUBLIC CUDA)
endif()

if (MSVC)
    target_compile_options(llama2_cpu_core PRIVATE /W4)
else()
    target_compile_options(llama2_cpu_core PRIVATE -Wall -Wextra -Wpedantic)
endif()
