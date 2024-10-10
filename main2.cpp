//
// Created by servant-of-scietia on 10/9/24.
//
#include "pch.h"

#include "Tensor.h"

int main() {
    //cudnnSetCallback(15, nullptr, nullptr);
        namespace fe = cudnn_frontend;

    if (cudnnGetVersion() < 8600) {
        return 1;
    }

    if (cudnnGetCudartVersion() < 12000) {
        return 1;
    }

    // matmul problem size
    int64_t const b = 16;
    int64_t const m = 32;
    int64_t const n = 64;
    int64_t const k = 128;

    // Initialize input tensors
    Tensor<half> A_gpu(b * m * k);
    Tensor<half> B_gpu(b * k * n);
    Tensor<half> Bias_gpu(b * m * 1);

    // Make cudnn graph
    fe::graph::Graph graph{};

    // Create the two non-virtual input tensors A and B.
    // There are read from global memory.
    auto A_attributes = fe::graph::Tensor_attributes()
                            .set_name("A")
                            .set_dim({b, m, k})
                            .set_stride({m * k, k, 1})
                            .set_data_type(fe::DataType_t::BFLOAT16);
    auto A            = graph.tensor(A_attributes);
    auto B_attributes = fe::graph::Tensor_attributes()
                            .set_name("B")
                            .set_dim({b, k, n})
                            .set_stride({k * n, n, 1})
                            .set_data_type(fe::DataType_t::BFLOAT16);
    auto B = graph.tensor(B_attributes);

    // Create Bias vector
    auto Bias_attributes =
        fe::graph::Tensor_attributes().set_name("Bias").set_dim({b, m, 1}).set_stride({m, 1, 1}).set_data_type(
            fe::DataType_t::BFLOAT16);
    auto Bias = graph.tensor(Bias_attributes);

    // Add ADD operation
    auto pw_0_attributes = fe::graph::Pointwise_attributes()
                               .set_name("pw0_Add")
                               .set_mode(fe::PointwiseMode_t::ADD)
                               .set_compute_data_type(fe::DataType_t::FLOAT);
    auto A_after_pw_0 = graph.pointwise(A, Bias, pw_0_attributes);
    A_after_pw_0->set_data_type(fe::DataType_t::BFLOAT16);

    auto matmul_attributes =
        fe::graph::Matmul_attributes().set_name("GEMM").set_compute_data_type(fe::DataType_t::FLOAT);
    auto C = graph.matmul(A_after_pw_0, B, matmul_attributes);
    C->set_output(true).set_data_type(fe::DataType_t::FLOAT);

    checkFeErrors(graph.validate());

    cudnnHandle_t handle;
    if(cudnnCreate(&handle))return 1;

    checkFeErrors(graph.build_operation_graph(handle));
    checkFeErrors(graph.create_execution_plans({fe::HeurMode_t::A}));

    // checkFeErrors(graph.check_support(handle));
    //
    // checkFeErrors(graph.build_plans(handle, fe::BuildPlanPolicy_t::HEURISTICS_CHOICE));
    //
    // // Run cudnn graph
    //
    //
    // int64_t workspace_size;
    // checkFeErrors(graph.get_workspace_size(workspace_size));
    // std::cout << "Workspace size: " << workspace_size << std::endl;
    // Tensor<std::byte> workspace(workspace_size);
    //
    // std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {
    //     {A, A_gpu.getDevPtr()}, {Bias, Bias_gpu.getDevPtr()}, {C, C_gpu.getDevPtr()}};
    //
    // std::cout << graph.print() << std::endl;
    // checkFeErrors(graph.execute(handle, variant_pack, workspace.getDevPtr()));
    // if(cudnnDestroy(handle))
    //     std::cout << "Failed to destroy cuDNN handle" << std::endl;

    return 0;
}