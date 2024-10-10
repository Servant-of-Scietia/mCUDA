//
// Created by servant-of-scietia on 10/6/24.
//

#ifndef MATMUL_H
#define MATMUL_H

#include <cudnn_frontend.h>


class Matmul
{
    std::vector<std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>> mInputs;
public:

    std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> create(const std::vector<std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>> &inputs, cudnn_frontend::graph::Graph &graph)
    {
        mInputs = inputs;
        const auto matmulAttributes = cudnn_frontend::graph::Matmul_attributes()
            .set_compute_data_type(cudnn_frontend::DataType_t::FLOAT);
        return graph.matmul(inputs.at(0), inputs.at(1), matmulAttributes);
    }

    std::vector<std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>> createDerivative(const std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> &dOutput, cudnn_frontend::graph::Graph &graph) const
    {
        const auto inputZeroDerivative = cudnn_frontend::graph::Matmul_attributes()
            .set_compute_data_type(cudnn_frontend::DataType_t::FLOAT);

        const auto inputOneDerivative = cudnn_frontend::graph::Matmul_attributes()
            .set_compute_data_type(cudnn_frontend::DataType_t::FLOAT);

        return {graph.matmul(dOutput, mInputs.at(1), inputZeroDerivative), graph.matmul(mInputs.at(0), dOutput, inputOneDerivative)};
    }

};



#endif //MATMUL_H
