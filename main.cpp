#include "pch.h"
#include "Tensor.h"

int main() {
    //cudnnSetCallback(15, nullptr, nullptr);
    size_t version = cudnnGetVersion();
    std::cout << "cuDNN version: " << version << std::endl;
    namespace fe = cudnn_frontend;

    // matmul problem size
    int64_t const b = 1;
    int64_t const m = 100;
    int64_t const n = 1;
    int64_t const k = 784;

    // Make cudnn graph
    fe::graph::Graph graph{};

    // Create the two non-virtual input tensors A and B.
    // There are read from global memory.
    auto A_attributes = fe::graph::Tensor_attributes()
                            .set_name("A")
                            .set_dim({b, m, k})
                            .set_stride({m*k, k, 1})
                            .set_data_type(fe::DataType_t::FLOAT);
    auto A            = graph.tensor(A_attributes);
    auto B_attributes = fe::graph::Tensor_attributes()
                            .set_name("B")
                            .set_dim({b, k, n})
                            .set_stride({k*n, n, 1})
                            .set_data_type(fe::DataType_t::FLOAT);
    auto B = graph.tensor(B_attributes);

    // Add MATMUL operation
    auto matmul_attributes = cudnn_frontend::graph::Matmul_attributes()
                                 .set_compute_data_type(cudnn_frontend::DataType_t::FLOAT)
                                 .set_name("GEMM");
    auto C = graph.matmul(A, B, matmul_attributes);
    C->set_data_type(cudnn_frontend::DataType_t::FLOAT);

    auto Bias_attributes = cudnn_frontend::graph::Tensor_attributes()
                               .set_name("Bias")
                               .set_dim({b, m, n})
                               .set_data_type(cudnn_frontend::DataType_t::FLOAT)
                               .set_stride({m * n, n, 1});
    auto Bias = graph.tensor(Bias_attributes);

    // Add ADD operation
    auto add_attributes = cudnn_frontend::graph::Pointwise_attributes()
                              .set_name("pw1_add")
                              .set_mode(cudnn_frontend::PointwiseMode_t::ADD)
                              .set_compute_data_type(cudnn_frontend::DataType_t::FLOAT);

    auto C_after_add = graph.pointwise(C, Bias, add_attributes);
    C_after_add->set_output(true).set_data_type(cudnn_frontend::DataType_t::FLOAT);


    bool good = graph.validate().is_good();
    std::cout << "Graph validation: " << (good ? "good" : "bad") << std::endl;
    std::cout << graph.print() << std::endl;

    cudnnHandle_t handle;
    if(cudnnCreate(&handle))
        std::cout << "Failed to create cuDNN handle" << std::endl;

    good = graph.build_operation_graph(handle).is_good();
    std::cout << "Operation graph build: " << (good ? "good" : "bad") << std::endl;

    good = graph.create_execution_plans({fe::HeurMode_t::A}).is_good();
    std::cout << "Execution plan creation: " << (good ? "good" : "bad") << std::endl;

    good = graph.check_support(handle).is_good();
    std::cout << "Check support: " << (good ? "good" : "bad") << std::endl;

    good = graph.build_plans(handle, fe::BuildPlanPolicy_t::HEURISTICS_CHOICE).is_good();
    std::cout << "Build plans: " << (good ? "good" : "bad") << std::endl;

    // Run cudnn graph
    Tensor<float> A_gpu(b * m * k);
    Tensor<float> B_gpu(b * k * n);
    Tensor<float> C_gpu(b * m * n);
    Tensor<float> Bias_gpu(b * m * n);

    int64_t workspace_size;
    good = graph.get_workspace_size(workspace_size).is_good();
    std::cout << "Get workspace size: " << (good ? "good" : "bad") << std::endl;
    std::cout << "Workspace size: " << workspace_size << std::endl;

    Tensor<std::byte> workspace(workspace_size);
    //
    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {
        {A, A_gpu.getDevPtr()}, {B, B_gpu.getDevPtr()}, {C_after_add, C_gpu.getDevPtr()}, {Bias, Bias_gpu.getDevPtr()}};

    std::cout << graph.print() << std::endl;
    good = graph.execute(handle, variant_pack, workspace.getDevPtr()).is_good();
    std::cout << "Execution: " << (good ? "good" : "bad") << std::endl;
    cudnnDestroy(handle);

    return 0;
}