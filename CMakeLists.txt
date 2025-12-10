cmake_minimum_required(VERSION 3.18)
project(cuda-kernel-essentials LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Build each kernel as a separate executable
add_executable(vector_add src/vector_add.cu)
add_executable(elementwise src/elementwise.cu)
add_executable(reduction src/reduction.cu)
add_executable(prefix_sum src/prefix_sum.cu)
add_executable(matmul src/matmul.cu)

# Set CUDA architectures (A10 is SM 86)
set_target_properties(vector_add elementwise reduction prefix_sum matmul
    PROPERTIES CUDA_ARCHITECTURES "86")
