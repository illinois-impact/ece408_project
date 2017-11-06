
#ifndef MXNET_OPERATOR_NEW_FORWARD_H_
#define MXNET_OPERATOR_NEW_FORWARD_H_

#include <mxnet/base.h>

namespace mxnet
{
namespace op
{

// This function is called by new-inl.h
// Any code you write should be executed by this function
template <typename cpu, typename DType>
void forward(mshadow::Tensor<cpu, 4, DType> &y, const mshadow::Tensor<cpu, 4, DType> &x, const mshadow::Tensor<cpu, 4, DType> &k)
{
    /*
    Modify this function to implement the forward pass described in Chapter 16.
    The code in 16 is for a single image.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct, not fast (this is the CPU implementation.)
    */

    
    const int B = x.shape_[0];
    // const int M = y.shape_[1];
    // const int C = x.shape_[1];
    // const int H = x.shape_[2];
    // const int W = x.shape_[3];
    // const int K = w.shape_[3];

    for (int b = 0; b < B; ++b) {
        CHECK_EQ(0, 1) << "Missing an ECE408 CPU implementation!";

        /* ... a bunch of nested loops later...
            y[b][m][h][w] += x[b][c][h + p][w + q] * k[m][c][p][q];
        */
    }


}
}
}

#endif