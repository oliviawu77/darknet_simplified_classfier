#ifndef BLAS_H
#define BLAS_H
#include <stdlib.h>
#include "darknet.h"


#ifdef __cplusplus
extern "C" {
#endif

void scal_cpu(int N, float ALPHA, float *X, int INCX);
void fill_cpu(int N, float ALPHA, float * X, int INCX);
void test_gpu_blas();

void add_bias(float *output, float *biases, int batch, int n, int size);

void l2_cpu(int n, float *pred, float *truth, float *delta, float *error);

void softmax(float *input, int n, float temp, float *output, int stride);
void softmax_cpu(float *input, int n, int batch, int batch_offset, int groups, int group_offset, int stride, float temp, float *output);

#ifdef __cplusplus
}
#endif
#endif
