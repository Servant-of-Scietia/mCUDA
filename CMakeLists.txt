cmake_minimum_required(VERSION 3.28)
project(untitled LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 11)

find_package(CUDAToolkit REQUIRED)

# Add executable
add_executable(add add.cpp)
add_executable(matmul matmul.cpp)

# Link CUDA runtime libraries
target_link_libraries(add cudnn CUDA::cudart CUDA::nvrtc)
target_link_libraries(matmul cudnn CUDA::cudart CUDA::nvrtc)

target_precompile_headers(add PRIVATE pch.h)
target_precompile_headers(matmul PRIVATE pch.h)

