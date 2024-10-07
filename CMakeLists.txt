cmake_minimum_required(VERSION 3.28)
project(untitled LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 11)

# Specify the path to CUDA if necessary
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)
set(CUDNN_FRONTEND_LIBRARY /usr/local/cudnn-frontend)
set(CUDNN_LIBRARY /usr/include)

# Add include directories
include_directories(${CUDA_TOOLKIT_ROOT_DIR}/include)
include_directories(${CUDNN_LIBRARY})
include_directories(${CUDNN_FRONTEND_LIBRARY}/include)

# Add executable
add_executable(untitled main.cpp)

# Link CUDA runtime libraries
target_link_libraries(untitled ${CUDA_LIBRARIES} ${CUDNN_FRONTEND_LIBRARY} cudnn)
