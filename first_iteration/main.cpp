//
// Created by servant-of-scietia on 9/27/24.
//
#include <iostream>
#include "kernel.cuh"

int main() {
    const int size = 10;
    int data[size] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

    launchKernel(data, size);

    for (int i = 0; i < size; ++i) {
        std::cout << data[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}

