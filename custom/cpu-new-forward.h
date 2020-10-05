#ifndef SRC_LAYER_CPU_NEW_FORWARD_H
#define SRC_LAYER_CPU_NEW_FORWARD_H

void conv_forward_cpu(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K);

#endif // SRC_LAYER_CPU_NEW_FORWARD_H
