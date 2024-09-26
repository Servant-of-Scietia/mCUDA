cmake_minimum_required(VERSION 3.28)
project(untitled CUDA)
find_package(CUDA REQUIRED)

set(CMAKE_CUDA_STANDARD 20)

add_executable(untitled add.cu
        test.cu
        test.cu
        add.cu)

set_target_properties(untitled PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
