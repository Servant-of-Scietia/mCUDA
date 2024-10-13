//
// Created by servant-of-scietia on 10/9/24.
//
#include "pch.h"

#include "Tensor.h"
#include <cudnn_frontend.h>

int main() {
    //cudnnSetCallback(15, nullptr, nullptr);
        namespace fe = cudnn_frontend;
    cudnnHandle_t handle;
    if(cudnnCreate(&handle))return 1;

    if (cudnnGetVersion() < 8600) {
        return 1;
    }

    if (cudnnGetCudartVersion() < 12000) {
        return 1;
    }

    // matmul problem size
    int64_t const b = 1;
    int64_t const m = 32;
    int64_t const n = 128;

    // Initialize input tensors
    // Tensor<half> A_gpu(b * m * k);
    // Tensor<half> B_gpu(b * k * n);
    // Tensor<half> Bias_gpu(b * m * 1);


    // Make cudnn graph
    fe::graph::Graph graph;

    // Create the two non-virtual input tensors A and B.
    // There are read from global memory.
    auto A_attributes = fe::graph::Tensor_attributes()
                            .set_name("A")
                            .set_dim({b, m, n})
                            .set_stride({m * n, n, 1})
                            .set_data_type(fe::DataType_t::FLOAT);
    auto A            = graph.tensor(A_attributes);

    // Create Bias vector
    auto Bias_attributes =
        fe::graph::Tensor_attributes().set_name("Bias").set_dim({b, m, n}).set_stride({m*n, n, 1}).set_data_type(
            fe::DataType_t::FLOAT);
    auto Bias = graph.tensor(Bias_attributes);

    // Add ADD operation
    auto pw_0_attributes = fe::graph::Pointwise_attributes()
                               .set_name("pw0_Add")
                               .set_mode(fe::PointwiseMode_t::ADD)
                               .set_compute_data_type(fe::DataType_t::FLOAT);
    auto A_after_pw_0 = graph.pointwise(A, Bias, pw_0_attributes);
    A_after_pw_0->set_output(true).set_data_type(fe::DataType_t::FLOAT);


    graph.validate();



    graph.build_operation_graph(handle);
    std::cout << graph.create_execution_plans({fe::HeurMode_t::A}).is_good();
    std::int64_t count = graph.get_execution_plan_count();
    std::cout << "Number of execution plans: " << count << std::endl;

    //
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