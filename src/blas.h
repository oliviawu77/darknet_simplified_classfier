#ifndef BLAS_H
#define BLAS_H
#include <stdlib.h>
#include "darknet.h"


#ifdef __cplusplus
extern "C" {
#endif

void axpy_cpu(int N, float ALPHA, float *X, int INCX, float *Y, int INCY);
void scal_cpu(int N, float ALPHA, float *X, int INCX);
void fill_cpu(int N, float ALPHA, float * X, int INCX);
void test_gpu_blas();

void add_bias(float *output, float *biases, int batch, int n, int size);
void backward_scale_cpu(float *x_norm, float *delta, int batch, int n, int size, float *scale_updates);
void mean_delta_cpu(float *delta, float *variance, int batch, int filters, int spatial, float *mean_delta);
void  variance_delta_cpu(float *x, float *delta, float *mean, float *variance, int batch, int filters, int spatial, float *variance_delta);
void normalize_delta_cpu(float *x, float *mean, float *variance, float *mean_delta, float *variance_delta, int batch, int filters, int spatial, float *delta);

void smooth_l1_cpu(int n, float *pred, float *truth, float *delta, float *error);
void l2_cpu(int n, float *pred, float *truth, float *delta, float *error);

void softmax(float *input, int n, float temp, float *output, int stride);
void softmax_cpu(float *input, int n, int batch, int batch_offset, int groups, int group_offset, int stride, float temp, float *output);
void softmax_x_ent_cpu(int n, float *pred, float *truth, float *delta, float *error);

#ifdef __cplusplus
}
#endif
#endif
