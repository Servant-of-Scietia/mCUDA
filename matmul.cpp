//
// Created by servant-of-scietia on 10/12/24.
//
#include "Tensor.h"

#define checkCudnnErr(status)                                                      \
{                                                                                    \
    if (status != CUDNN_STATUS_SUCCESS)                                                       \
    {                                                                                \
        std::cerr << "Error in " << __FILE__ << " at line " << __LINE__ << ": " << cudnnGetErrorString(status) << std::endl;      \
        exit(1);                                                                     \
    }                                                                                \
}

std::int32_t main()
{
    cudnnHandle_t handle;
    checkCudnnErr(cudnnCreate(&handle));

    std::int64_t const b = 1;
    std::int64_t const m = 32;
	std::int64_t const k = 64;
    std::int64_t const n = 128;

    cudnnBackendDescriptor_t A_desc;
    checkCudnnErr(cudnnBackendCreateDescriptor(CUDNN_BACKEND_TENSOR_DESCRIPTOR, &A_desc));

    std::int64_t A_ID = 0;
    std::int64_t A_dim[] = {b, m, k};
    std::int64_t A_stride[] = {m * k, k, 1};
    cudnnDataType_t A_data_type = CUDNN_DATA_FLOAT;
    std::int64_t A_alignment = 4;

    checkCudnnErr(cudnnBackendSetAttribute(A_desc, CUDNN_ATTR_TENSOR_UNIQUE_ID, CUDNN_TYPE_INT64, 1, &A_ID));
    checkCudnnErr(cudnnBackendSetAttribute(A_desc,CUDNN_ATTR_TENSOR_DATA_TYPE, CUDNN_TYPE_DATA_TYPE, 1, &A_data_type));
    checkCudnnErr(cudnnBackendSetAttribute(A_desc, CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT, CUDNN_TYPE_INT64, 1, &A_alignment));
    checkCudnnErr(cudnnBackendSetAttribute(A_desc, CUDNN_ATTR_TENSOR_DIMENSIONS, CUDNN_TYPE_INT64, 3, A_dim));
    checkCudnnErr(cudnnBackendSetAttribute(A_desc, CUDNN_ATTR_TENSOR_STRIDES, CUDNN_TYPE_INT64, 3, A_stride));

    checkCudnnErr(cudnnBackendFinalize(A_desc));

    cudnnBackendDescriptor_t B_desc;
    checkCudnnErr(cudnnBackendCreateDescriptor(CUDNN_BACKEND_TENSOR_DESCRIPTOR, &B_desc));

    std::int64_t B_ID = 1;
    std::int64_t B_dim[] = {b, k, n};
    std::int64_t B_stride[] = {k * n, n, 1};
    cudnnDataType_t B_data_type = CUDNN_DATA_FLOAT;
    std::int64_t B_alignment = 4;

    checkCudnnErr(cudnnBackendSetAttribute(B_desc, CUDNN_ATTR_TENSOR_UNIQUE_ID, CUDNN_TYPE_INT64, 1, &B_ID));
    checkCudnnErr(cudnnBackendSetAttribute(B_desc, CUDNN_ATTR_TENSOR_DATA_TYPE, CUDNN_TYPE_DATA_TYPE, 1, &B_data_type));
    checkCudnnErr(cudnnBackendSetAttribute(B_desc, CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT, CUDNN_TYPE_INT64, 1, &B_alignment));
    checkCudnnErr(cudnnBackendSetAttribute(B_desc, CUDNN_ATTR_TENSOR_DIMENSIONS, CUDNN_TYPE_INT64, 3, B_dim));
    checkCudnnErr(cudnnBackendSetAttribute(B_desc, CUDNN_ATTR_TENSOR_STRIDES, CUDNN_TYPE_INT64, 3, B_stride));

    checkCudnnErr(cudnnBackendFinalize(B_desc));

    cudnnBackendDescriptor_t C_desc;
	checkCudnnErr(cudnnBackendCreateDescriptor(CUDNN_BACKEND_TENSOR_DESCRIPTOR, &C_desc));

    std::int64_t C_ID = 2;
    std::int64_t C_dim[] = {b, m, n};
    std::int64_t C_stride[] = {m * n, n, 1};
    cudnnDataType_t C_data_type = CUDNN_DATA_FLOAT;
    std::int64_t C_alignment = 4;

    checkCudnnErr(cudnnBackendSetAttribute(C_desc, CUDNN_ATTR_TENSOR_UNIQUE_ID, CUDNN_TYPE_INT64, 1, &C_ID));
    checkCudnnErr(cudnnBackendSetAttribute(C_desc, CUDNN_ATTR_TENSOR_DATA_TYPE, CUDNN_TYPE_DATA_TYPE, 1, &C_data_type));
    checkCudnnErr(cudnnBackendSetAttribute(C_desc, CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT, CUDNN_TYPE_INT64, 1, &C_alignment));
    checkCudnnErr(cudnnBackendSetAttribute(C_desc, CUDNN_ATTR_TENSOR_DIMENSIONS, CUDNN_TYPE_INT64, 3, C_dim));
    checkCudnnErr(cudnnBackendSetAttribute(C_desc, CUDNN_ATTR_TENSOR_STRIDES, CUDNN_TYPE_INT64, 3, C_stride));

    checkCudnnErr(cudnnBackendFinalize(C_desc));

    cudnnBackendDescriptor_t matmul_desc;
    checkCudnnErr(cudnnBackendCreateDescriptor(CUDNN_BACKEND_MATMUL_DESCRIPTOR, &matmul_desc));

    cudnnDataType_t compute_data_type = CUDNN_DATA_FLOAT;

    checkCudnnErr(cudnnBackendSetAttribute(matmul_desc, CUDNN_ATTR_MATMUL_COMP_TYPE, CUDNN_TYPE_DATA_TYPE, 1, &compute_data_type));

    checkCudnnErr(cudnnBackendFinalize(matmul_desc));

    cudnnBackendDescriptor_t matmul_op_desc;
    checkCudnnErr(cudnnBackendCreateDescriptor(CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR, &matmul_op_desc));

    std::int64_t batch_count = 1;

    checkCudnnErr(cudnnBackendSetAttribute(matmul_op_desc, CUDNN_ATTR_OPERATION_MATMUL_ADESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &A_desc));
    checkCudnnErr(cudnnBackendSetAttribute(matmul_op_desc, CUDNN_ATTR_OPERATION_MATMUL_BDESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &B_desc));
    checkCudnnErr(cudnnBackendSetAttribute(matmul_op_desc, CUDNN_ATTR_OPERATION_MATMUL_CDESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &C_desc));
    checkCudnnErr(cudnnBackendSetAttribute(matmul_op_desc, CUDNN_ATTR_OPERATION_MATMUL_DESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &matmul_desc));
    checkCudnnErr(cudnnBackendSetAttribute(matmul_op_desc, CUDNN_ATTR_OPERATION_MATMUL_IRREGULARLY_STRIDED_BATCH_COUNT, CUDNN_TYPE_INT64, 1, &batch_count));

    checkCudnnErr(cudnnBackendFinalize(matmul_op_desc));

	cudnnBackendDescriptor_t op_graph;
    checkCudnnErr(cudnnBackendCreateDescriptor(CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR, &op_graph));

    checkCudnnErr(cudnnBackendSetAttribute(op_graph, CUDNN_ATTR_OPERATIONGRAPH_OPS, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &matmul_op_desc));
    checkCudnnErr(cudnnBackendSetAttribute(op_graph, CUDNN_ATTR_OPERATIONGRAPH_HANDLE, CUDNN_TYPE_HANDLE, 1, &handle));

    checkCudnnErr(cudnnBackendFinalize(op_graph));

    cudnnBackendDescriptor_t engine;
    checkCudnnErr(cudnnBackendCreateDescriptor(CUDNN_BACKEND_ENGINE_DESCRIPTOR, &engine));

    std::int64_t gidx = 0;
    checkCudnnErr(cudnnBackendSetAttribute(engine, CUDNN_ATTR_ENGINE_OPERATION_GRAPH, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &op_graph));
    checkCudnnErr(cudnnBackendSetAttribute(engine, CUDNN_ATTR_ENGINE_GLOBAL_INDEX, CUDNN_TYPE_INT64, 1, &gidx));

    checkCudnnErr(cudnnBackendFinalize(engine));

    cudnnBackendDescriptor_t engcfg;
    checkCudnnErr(cudnnBackendCreateDescriptor(CUDNN_BACKEND_ENGINECFG_DESCRIPTOR, &engcfg));

    checkCudnnErr(cudnnBackendSetAttribute(engcfg, CUDNN_ATTR_ENGINECFG_ENGINE, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &engine));

    checkCudnnErr(cudnnBackendFinalize(engcfg));

    cudnnBackendDescriptor_t plan;
    checkCudnnErr(cudnnBackendCreateDescriptor(CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR, &plan));

    checkCudnnErr(cudnnBackendSetAttribute(plan, CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &engcfg));
    checkCudnnErr(cudnnBackendSetAttribute(plan, CUDNN_ATTR_EXECUTION_PLAN_HANDLE, CUDNN_TYPE_HANDLE, 1, &handle));

    checkCudnnErr(cudnnBackendFinalize(plan));

	std::int64_t workspace_size;
    checkCudnnErr(cudnnBackendGetAttribute(plan, CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE, CUDNN_TYPE_INT64, 1, NULL, &workspace_size));

	std::cout << "Workspace size: " << workspace_size << std::endl;

    Tensor<float> A_gpu(b * m * k);
    Tensor<float> B_gpu(b * k * n);
    Tensor<float> C_gpu(b * m * n);
    Tensor<std::byte> workspace(workspace_size);

    cudnnBackendDescriptor_t varpack;
    checkCudnnErr(cudnnBackendCreateDescriptor(CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR, &varpack));

    void *devPtrs[3] = {A_gpu.getDevPtr(), B_gpu.getDevPtr(), C_gpu.getDevPtr()};
    std::int64_t uids[3] = {A_ID, B_ID, C_ID};
    void *workspace_ptr = workspace.getDevPtr();

    checkCudnnErr(cudnnBackendSetAttribute(varpack, CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS, CUDNN_TYPE_VOID_PTR, 3, devPtrs));
    checkCudnnErr(cudnnBackendSetAttribute(varpack, CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS, CUDNN_TYPE_INT64, 3, uids));
    checkCudnnErr(cudnnBackendSetAttribute(varpack, CUDNN_ATTR_VARIANT_PACK_WORKSPACE, CUDNN_TYPE_VOID_PTR, 1, &workspace_ptr));

    checkCudnnErr(cudnnBackendFinalize(varpack));

    checkCudnnErr(cudnnBackendExecute(handle, plan, varpack));
}