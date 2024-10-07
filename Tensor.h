//
// Created by servant-of-scietia on 10/6/24.
//

#ifndef TENSOR_H
#define TENSOR_H

#include <cuda_runtime.h>
#include <cstdint>
#include <iostream>

#define checkCudaErrors(status)                                                          \
    {                                                                                    \
        if (status != cudaSuccess)                                                       \
        {                                                                                \
            std::cerr << "CUDA error: " << cudaGetErrorString(status) << std::endl;      \
            exit(1);                                                                     \
        }                                                                                \
    }

template <typename Type>
struct Tensor
{
    Type* devPtr;
    Type* hostPtr;
    std::int64_t size;

  protected:
    explicit Tensor(){};

  public:
    explicit Tensor(std::int64_t size) : size(size)
    {
        checkCudaErrors(cudaMalloc((void**)&devPtr, (std::size_t)(size * sizeof(Type))));
        hostPtr = new Type[size];
    }

    ~Tensor()
    {
        if (devPtr != nullptr)
        {
            cudaFree(devPtr);
            devPtr = nullptr;
        }
        if (hostPtr != nullptr)
        {
            delete[] hostPtr;
            hostPtr = nullptr;
        }
    }
};



#endif //TENSOR_H
