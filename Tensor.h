//
// Created by servant-of-scietia on 10/6/24.
//
// Tensor.h

#ifndef TENSOR_H
#define TENSOR_H

#include "pch.h"

#define checkCudaErrors(status)                                                      \
{                                                                                    \
    if (status != cudaSuccess)                                                       \
    {                                                                                \
        std::cerr << "CUDA error: " << cudaGetErrorString(status) << std::endl;      \
        exit(1);                                                                     \
    }                                                                                \
}

template <typename Type>
class Tensor
{
    Type* devPtr;
    Type* hostPtr;
    std::uint64_t size;

public:
    explicit Tensor(const std::uint64_t size) : size(size)
    {
        checkCudaErrors(cudaMalloc((void**)&devPtr, (std::size_t)(size * sizeof(Type))));
        hostPtr = new Type[size];
        std::fill(hostPtr, hostPtr + size, Type{});
        checkCudaErrors(cudaMemcpy(devPtr, hostPtr, size_t(sizeof(hostPtr[0]) * size), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaDeviceSynchronize());
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

    void copyToDevice() const
    {
        checkCudaErrors(cudaMemcpy(devPtr, hostPtr, size * sizeof(Type), cudaMemcpyHostToDevice));
    }

    void copyToHost() const
    {
        checkCudaErrors(cudaMemcpy(hostPtr, devPtr, size * sizeof(Type), cudaMemcpyDeviceToHost));
    }

    Type* getDevPtr() const
    {
        return devPtr;
    }

    Type* getHostPtr() const
    {
        return hostPtr;
    }



};

#endif // TENSOR_H
