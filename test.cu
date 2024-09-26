//
// Created by servant-of-scietia on 24.09.24.
//
#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

int main() {
    const size_t size = 1 << 20; // 1 million floats
    float *h_data = new float[size]; // Host data
    float *d_data;

    cudaMalloc((void**)&d_data, size * sizeof(float));

    // Start timing
    auto start = std::chrono::high_resolution_clock::now();

    // Copy data from host to device
    cudaMemcpy(d_data, h_data, size * sizeof(float), cudaMemcpyHostToDevice);

    // Stop timing
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;

    // Calculate bandwidth
    double bandwidth = (size * sizeof(float)) / duration.count() / (1024 * 1024); // MB/s

    std::cout << "Bandwidth: " << bandwidth << " MB/s" << std::endl;

    cudaFree(d_data);
    delete[] h_data;

    return 0;
}
