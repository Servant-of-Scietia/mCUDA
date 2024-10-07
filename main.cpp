//
// Created by servant-of-scietia on 10/2/24.
//

#include "matmul.h"
#include <cudnn.h>
#include <iostream>
#include <cudnn_frontend.h>

int main() {
    size_t version = cudnnGetVersion();
    std::cout << "cuDNN version: " << version << std::endl;
    namespace fe = cudnn_frontend;

    // matmul problem size
    int64_t const b = 16;
    int64_t const m = 32;
    int64_t const n = 64;
    int64_t const k = 128;

    // Make cudnn graph
    fe::graph::Graph graph{};

    // Create the two non-virtual input tensors A and B.
    // There are read from global memory.
    auto A_attributes = fe::graph::Tensor_attributes()
                            .set_name("A")
                            .set_dim({b, m, k})
                            .set_stride({m * k, k, 1})
                            .set_data_type(fe::DataType_t::INT8);
    auto A            = graph.tensor(A_attributes);
    auto B_attributes = fe::graph::Tensor_attributes()
                            .set_name("B")
                            .set_dim({b, k, n})
                            .set_stride({k * n, 1, n})
                            .set_data_type(fe::DataType_t::INT8);
    auto B = graph.tensor(B_attributes);
    return 0;
}