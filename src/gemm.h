#ifndef GEMM_H
#define GEMM_H
#include "activations.h"
#include <stdint.h>
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif


int is_avx();
int is_fma_avx2();

void forward_maxpool_layer_avx(float *src, float *dst, int *indexes, int size, int w, int h, int out_w, int out_h, int c,
    int pad, int stride, int batch);


void gemm(int TA, int TB, int M, int N, int K, float ALPHA,
                    float *A, int lda,
                    float *B, int ldb,
                    float BETA,
                    float *C, int ldc);

void gemm_cpu(int TA, int TB, int M, int N, int K, float ALPHA,
        float *A, int lda,
        float *B, int ldb,
        float BETA,
        float *C, int ldc);

#ifdef __cplusplus
}
#endif
#endif