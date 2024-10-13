//
// Created by servant-of-scietia on 9/27/24.
//

#ifndef KERNEL_CUH
#define KERNEL_CUH

#include <cuda_runtime.h>

__global__ void myKernel(int* data);

void launchKernel(int* data, int size);

#endif //KERNEL_CUH
