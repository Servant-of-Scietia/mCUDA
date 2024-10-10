cmake_minimum_required(VERSION 3.28)
project(untitled LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 11)

# Specify the path to CUDA
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)

# Specify the path to cuDNN
set(CUDNN_LIBRARY /usr/include)

# Add include directories
include_directories(${CUDA_TOOLKIT_ROOT_DIR}/include)
include_directories(${CUDNN_LIBRARY})

# include cudnn-frontend
add_subdirectory(cudnn-frontend)

# Add executable
add_executable(main main.cpp)
add_executable(main2 main2.cpp)

# Link CUDA runtime libraries
target_link_libraries(main ${CUDA_LIBRARIES} cudnn_frontend cudnn)
target_link_libraries(main2 ${CUDA_LIBRARIES} cudnn_frontend cudnn)

target_precompile_headers(main PRIVATE pch.h)
target_precompile_headers(main2 PRIVATE pch.h)

