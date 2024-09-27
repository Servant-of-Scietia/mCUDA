//
// Created by servant-of-scietia on 9/27/24.
//

#include "kernel.cuh"
#include <iostream>

__global__ void myKernel(int* data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < 10) {
        data[idx] *= 2;
    }
}

void launchKernel(int* data, int size) {
    int* d_data;
    cudaMalloc((void**)&d_data, size * sizeof(int));
    cudaMemcpy(d_data, data, size * sizeof(int), cudaMemcpyHostToDevice);

    myKernel<<<1, size>>>(d_data);

    cudaMemcpy(data, d_data, size * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_data);
}

