#include "blas.h"
#include "utils.h"

#include <math.h>
#include <assert.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void mean_cpu(float *x, int batch, int filters, int spatial, float *mean)
{
    float scale = 1./(batch * spatial);
    int i,j,k;
    for(i = 0; i < filters; ++i){
        mean[i] = 0;
        for(j = 0; j < batch; ++j){
            for(k = 0; k < spatial; ++k){
                int index = j*filters*spatial + i*spatial + k;
                mean[i] += x[index];
            }
        }
        mean[i] *= scale;
    }
}

void variance_cpu(float *x, float *mean, int batch, int filters, int spatial, float *variance)
{
    float scale = 1./(batch * spatial - 1);
    int i,j,k;
    for(i = 0; i < filters; ++i){
        variance[i] = 0;
        for(j = 0; j < batch; ++j){
            for(k = 0; k < spatial; ++k){
                int index = j*filters*spatial + i*spatial + k;
                variance[i] += pow((x[index] - mean[i]), 2);
            }
        }
        variance[i] *= scale;
    }
}

void normalize_cpu(float *x, float *mean, float *variance, int batch, int filters, int spatial)
{
    int b, f, i;
    for(b = 0; b < batch; ++b){
        for(f = 0; f < filters; ++f){
            for(i = 0; i < spatial; ++i){
                int index = b*filters*spatial + f*spatial + i;
                x[index] = (x[index] - mean[f])/(sqrt(variance[f] + .00001f));
            }
        }
    }
}

void const_cpu(int N, float ALPHA, float *X, int INCX)
{
    int i;
    for(i = 0; i < N; ++i) X[i*INCX] = ALPHA;
}

void mul_cpu(int N, float *X, int INCX, float *Y, int INCY)
{
    int i;
    for(i = 0; i < N; ++i) Y[i*INCY] *= X[i*INCX];
}

void pow_cpu(int N, float ALPHA, float *X, int INCX, float *Y, int INCY)
{
    int i;
    for(i = 0; i < N; ++i) Y[i*INCY] = pow(X[i*INCX], ALPHA);
}

void axpy_cpu(int N, float ALPHA, float *X, int INCX, float *Y, int INCY)
{
    int i;
    for(i = 0; i < N; ++i) Y[i*INCY] += ALPHA*X[i*INCX];
}

void scal_cpu(int N, float ALPHA, float *X, int INCX)
{
    int i;
    for(i = 0; i < N; ++i) X[i*INCX] *= ALPHA;
}

void scal_add_cpu(int N, float ALPHA, float BETA, float *X, int INCX)
{
    int i;
    for (i = 0; i < N; ++i) X[i*INCX] = X[i*INCX] * ALPHA + BETA;
}

void fill_cpu(int N, float ALPHA, float *X, int INCX)
{
    int i;
    if (INCX == 1 && ALPHA == 0) {
        memset(X, 0, N * sizeof(float));
    }
    else {
        for (i = 0; i < N; ++i) X[i*INCX] = ALPHA;
    }
}

void deinter_cpu(int NX, float *X, int NY, float *Y, int B, float *OUT)
{
    int i, j;
    int index = 0;
    for(j = 0; j < B; ++j) {
        for(i = 0; i < NX; ++i){
            if(X) X[j*NX + i] += OUT[index];
            ++index;
        }
        for(i = 0; i < NY; ++i){
            if(Y) Y[j*NY + i] += OUT[index];
            ++index;
        }
    }
}

void inter_cpu(int NX, float *X, int NY, float *Y, int B, float *OUT)
{
    int i, j;
    int index = 0;
    for(j = 0; j < B; ++j) {
        for(i = 0; i < NX; ++i){
            OUT[index++] = X[j*NX + i];
        }
        for(i = 0; i < NY; ++i){
            OUT[index++] = Y[j*NY + i];
        }
    }
}

void copy_cpu(int N, float *X, int INCX, float *Y, int INCY)
{
    int i;
    for(i = 0; i < N; ++i) Y[i*INCY] = X[i*INCX];
}

void smooth_l1_cpu(int n, float *pred, float *truth, float *delta, float *error)
{
    int i;
    for(i = 0; i < n; ++i){
        float diff = truth[i] - pred[i];
        float abs_val = fabs(diff);
        if(abs_val < 1) {
            error[i] = diff * diff;
            delta[i] = diff;
        }
        else {
            error[i] = 2*abs_val - 1;
            delta[i] = (diff > 0) ? 1 : -1;
        }
    }
}

void softmax_x_ent_cpu(int n, float *pred, float *truth, float *delta, float *error)
{
    int i;
    for(i = 0; i < n; ++i){
        float t = truth[i];
        float p = pred[i];
        error[i] = (t) ? -log(p) : 0;
        delta[i] = t-p;
    }
}


void l2_cpu(int n, float *pred, float *truth, float *delta, float *error)
{
    int i;
    for(i = 0; i < n; ++i){
        float diff = truth[i] - pred[i];
        error[i] = diff * diff;
        delta[i] = diff;
    }
}

void softmax(float *input, int n, float temp, float *output, int stride)
{
    int i;
    float sum = 0;
    float largest = -FLT_MAX;
    for(i = 0; i < n; ++i){
        if(input[i*stride] > largest) largest = input[i*stride];
    }
    for(i = 0; i < n; ++i){
        float e = exp(input[i*stride]/temp - largest/temp);
        sum += e;
        output[i*stride] = e;
    }
    for(i = 0; i < n; ++i){
        output[i*stride] /= sum;
    }
}


void softmax_cpu(float *input, int n, int batch, int batch_offset, int groups, int group_offset, int stride, float temp, float *output)
{
    int g, b;
    for(b = 0; b < batch; ++b){
        for(g = 0; g < groups; ++g){
            softmax(input + b*batch_offset + g*group_offset, n, temp, output + b*batch_offset + g*group_offset, stride);
        }
    }
}

void get_embedding(float *src, int src_w, int src_h, int src_c, int embedding_size, int cur_w, int cur_h, int cur_n, int cur_b, float *dst)
{
    int i;
    for (i = 0; i < embedding_size; ++i) {
        const int src_index = cur_b*(src_c*src_h*src_w) + cur_n*(embedding_size*src_h*src_w) + i*src_h*src_w + cur_h*(src_w) + cur_w;

        const float val = src[src_index];
        dst[i] = val;
        //printf(" val = %f, ", val);
    }
}


// Euclidean_norm
float math_vector_length(float *A, unsigned int feature_size)
{
    float sum = 0;
    int i;
    for (i = 0; i < feature_size; ++i)
    {
        sum += A[i] * A[i];
    }
    float vector_length = sqrtf(sum);
    return vector_length;
}

float cosine_similarity(float *A, float *B, unsigned int feature_size)
{
    float mul = 0.0, d_a = 0.0, d_b = 0.0;

    int i;
    for(i = 0; i < feature_size; ++i)
    {
        mul += A[i] * B[i];
        d_a += A[i] * A[i];
        d_b += B[i] * B[i];
    }
    float similarity;
    float divider = sqrtf(d_a) * sqrtf(d_b);
    if (divider > 0) similarity = mul / divider;
    else similarity = 0;

    return similarity;
}

int get_sim_P_index(size_t i, size_t j, contrastive_params *contrast_p, int contrast_p_size)
{
    size_t z;
    for (z = 0; z < contrast_p_size; ++z) {
        if (contrast_p[z].i == i && contrast_p[z].j == j) break;
    }
    if (z == contrast_p_size) {
        return -1;   // not found
    }

    return z;   // found
}

int check_sim(size_t i, size_t j, contrastive_params *contrast_p, int contrast_p_size)
{
    size_t z;
    for (z = 0; z < contrast_p_size; ++z) {
        if (contrast_p[z].i == i && contrast_p[z].j == j) break;
    }
    if (z == contrast_p_size) {
        return 0;   // not found
    }

    return 1;   // found
}

float find_sim(size_t i, size_t j, contrastive_params *contrast_p, int contrast_p_size)
{
    size_t z;
    for (z = 0; z < contrast_p_size; ++z) {
        if (contrast_p[z].i == i && contrast_p[z].j == j) break;
    }
    if (z == contrast_p_size) {
        printf(" Error: find_sim(): sim isn't found: i = %d, j = %d, z = %d \n", i, j, z);
        getchar();
    }

    return contrast_p[z].sim;
}

float find_P_constrastive(size_t i, size_t j, contrastive_params *contrast_p, int contrast_p_size)
{
    size_t z;
    for (z = 0; z < contrast_p_size; ++z) {
        if (contrast_p[z].i == i && contrast_p[z].j == j) break;
    }
    if (z == contrast_p_size) {
        printf(" Error: find_P_constrastive(): P isn't found: i = %d, j = %d, z = %d \n", i, j, z);
        getchar();
    }

    return contrast_p[z].P;
}

// num_of_samples = 2 * loaded_images = mini_batch_size
float P_constrastive_f_det(size_t il, int *labels, float **z, unsigned int feature_size, float temperature, contrastive_params *contrast_p, int contrast_p_size)
{
    const float sim = contrast_p[il].sim;
    const size_t i = contrast_p[il].i;
    const size_t j = contrast_p[il].j;

    const float numerator = expf(sim / temperature);

    float denominator = 0;
    int k;
    for (k = 0; k < contrast_p_size; ++k) {
        contrastive_params cp = contrast_p[k];
        //if (k != i && labels[k] != labels[i]) {
        //if (k != i) {
        if (cp.i != i && cp.j == j) {
            //const float sim_den = cp.sim;
            ////const float sim_den = find_sim(k, l, contrast_p, contrast_p_size); // cosine_similarity(z[k], z[l], feature_size);
            //denominator += expf(sim_den / temperature);
            denominator += cp.exp_sim;
        }
    }

    float result = 0.9999;
    if (denominator != 0) result = numerator / denominator;
    if (result > 1) result = 0.9999;
    return result;
}

// num_of_samples = 2 * loaded_images = mini_batch_size
float P_constrastive_f(size_t i, size_t l, int *labels, float **z, unsigned int feature_size, float temperature, contrastive_params *contrast_p, int contrast_p_size)
{
    if (i == l) {
        fprintf(stderr, " Error: in P_constrastive must be i != l, while i = %d, l = %d \n", i, l);
        getchar();
    }

    const float sim = find_sim(i, l, contrast_p, contrast_p_size); // cosine_similarity(z[i], z[l], feature_size);
    const float numerator = expf(sim / temperature);

    float denominator = 0;
    int k;
    for (k = 0; k < contrast_p_size; ++k) {
        contrastive_params cp = contrast_p[k];
        //if (k != i && labels[k] != labels[i]) {
        //if (k != i) {
        if (cp.i != i && cp.j == l) {
            //const float sim_den = cp.sim;
            ////const float sim_den = find_sim(k, l, contrast_p, contrast_p_size); // cosine_similarity(z[k], z[l], feature_size);
            //denominator += expf(sim_den / temperature);
            denominator += cp.exp_sim;
        }
    }

    float result = 0.9999;
    if (denominator != 0) result = numerator / denominator;
    if (result > 1) result = 0.9999;
    return result;
}

void grad_contrastive_loss_positive_f(size_t i, int *class_ids, int *labels, size_t num_of_samples, float **z, unsigned int feature_size, float temperature, float *delta, int wh, contrastive_params *contrast_p, int contrast_p_size)
{
    const float vec_len = math_vector_length(z[i], feature_size);
    size_t j;
    float N = 0;
    for (j = 0; j < num_of_samples; ++j) {
        if (labels[i] == labels[j] && labels[i] >= 0) N++;
    }
    if (N == 0 || temperature == 0 || vec_len == 0) {
        fprintf(stderr, " Error: N == 0 || temperature == 0 || vec_len == 0. N=%f, temperature=%f, vec_len=%f, labels[i] = %d \n",
            N, temperature, vec_len, labels[i]);
        getchar();
        return;
    }
    const float mult = 1 / ((N - 1) * temperature * vec_len);

    for (j = 0; j < num_of_samples; ++j) {
        //if (i != j && (i/2) == (j/2)) {
        if (i != j && labels[i] == labels[j] && labels[i] >= 0) {
            //printf(" i = %d, j = %d, num_of_samples = %d, labels[i] = %d, labels[j] = %d \n",
            //    i, j, num_of_samples, labels[i], labels[j]);
            const int sim_P_i = get_sim_P_index(i, j, contrast_p, contrast_p_size);
            if (sim_P_i < 0) continue;
            const float sim = contrast_p[sim_P_i].sim;
            const float P = contrast_p[sim_P_i].P;
            //if (!check_sim(i, j, contrast_p, contrast_p_size)) continue;
            //const float sim = find_sim(i, j, contrast_p, contrast_p_size); //cos_sim[i*num_of_samples + j];        // cosine_similarity(z[i], z[j], feature_size);
            //const float P = find_P_constrastive(i, j, contrast_p, contrast_p_size); //p_constrastive[i*num_of_samples + j];   // P_constrastive(i, j, labels, num_of_samples, z, feature_size, temperature, cos_sim);
                                                                    //const float custom_pos_mult = 1 - sim;


            int m;
            //const float d = mult*(sim * z[i][m] - z[j][m]) * (1 - P); // 1
            for (m = 0; m < feature_size; ++m) {
                //const float d = mult*(sim * z[j][m] - z[j][m]) * (1 - P); // my
                //const float d = mult*(sim * z[i][m] + sim * z[j][m] - z[j][m]) *(1 - P); // 1+2
                const float d = mult*(sim * z[i][m] - z[j][m]) *(1 - P); // 1 (70%)
                //const float d = mult*(sim * z[j][m] - z[j][m]) * (1 - P); // 2
                // printf(" pos: z[j][m] = %f, z[i][m] = %f, d = %f, sim = %f \n", z[j][m], z[i][m], d, sim);
                const int out_i = m * wh;
                delta[out_i] -= d;
            }
        }
    }
}

void grad_contrastive_loss_negative_f(size_t i, int *class_ids, int *labels, size_t num_of_samples, float **z, unsigned int feature_size, float temperature, float *delta, int wh, contrastive_params *contrast_p, int contrast_p_size, int neg_max)
{
    const float vec_len = math_vector_length(z[i], feature_size);
    size_t j;
    float N = 0;
    for (j = 0; j < num_of_samples; ++j) {
        if (labels[i] == labels[j] && labels[i] >= 0) N++;
    }
    if (N == 0 || temperature == 0 || vec_len == 0) {
        fprintf(stderr, " Error: N == 0 || temperature == 0 || vec_len == 0. N=%f, temperature=%f, vec_len=%f, labels[i] = %d \n",
            N, temperature, vec_len, labels[i]);
        getchar();
        return;
    }
    const float mult = 1 / ((N - 1) * temperature * vec_len);

    int neg_counter = 0;

    for (j = 0; j < num_of_samples; ++j) {
        //if (i != j && (i/2) == (j/2)) {
        if (labels[i] >= 0 && labels[i] == labels[j] && i != j) {

            size_t k;
            for (k = 0; k < num_of_samples; ++k) {
                //if (k != i && k != j && labels[k] != labels[i]) {
                if (k != i && k != j && labels[k] != labels[i] && class_ids[j] == class_ids[k]) {
                    neg_counter++;
                    const int sim_P_i = get_sim_P_index(i, k, contrast_p, contrast_p_size);
                    if (sim_P_i < 0) continue;
                    const float sim = contrast_p[sim_P_i].sim;
                    const float P = contrast_p[sim_P_i].P;
                    //if (!check_sim(i, k, contrast_p, contrast_p_size)) continue;
                    //const float sim = find_sim(i, k, contrast_p, contrast_p_size); //cos_sim[i*num_of_samples + k];        // cosine_similarity(z[i], z[k], feature_size);
                    //const float P = find_P_constrastive(i, k, contrast_p, contrast_p_size); //p_constrastive[i*num_of_samples + k];   // P_constrastive(i, k, labels, num_of_samples, z, feature_size, temperature, cos_sim);
                                                                            //const float custom_pos_mult = 1 + sim;

                    int m;
                    //const float d = mult*(z[k][m] + sim * z[i][m]) * P;   // my1
                    for (m = 0; m < feature_size; ++m) {
                        //const float d = mult*(z[k][m] + sim * z[i][m]) * P;   // 1 (70%)
                        //const float d = mult*(z[k][m] - sim * z[k][m] - sim * z[i][m]) * P;   // 1+2
                        const float d = mult*(z[k][m] - sim * z[i][m]) * P;   // 1 (70%)
                        //const float d = mult*(z[k][m] - sim * z[k][m]) * P; // 2
                        //printf(" neg: z[k][m] = %f, z[i][m] = %f, d = %f, sim = %f \n", z[k][m], z[i][m], d, sim);
                        const int out_i = m * wh;
                        delta[out_i] -= d;
                    }

                    if (neg_counter >= neg_max) return;
                }
            }
        }
    }
}



// num_of_samples = 2 * loaded_images = mini_batch_size
float P_constrastive(size_t i, size_t l, int *labels, size_t num_of_samples, float **z, unsigned int feature_size, float temperature, float *cos_sim, float *exp_cos_sim)
{
    if (i == l) {
        fprintf(stderr, " Error: in P_constrastive must be i != l, while i = %d, l = %d \n", i, l);
        getchar();
    }

    //const float sim = cos_sim[i*num_of_samples + l]; // cosine_similarity(z[i], z[l], feature_size);
    //const float numerator = expf(sim / temperature);
    const float numerator = exp_cos_sim[i*num_of_samples + l];

    float denominator = 0;
    int k;
    for (k = 0; k < num_of_samples; ++k) {
        //if (k != i && labels[k] != labels[i]) {
        if (k != i) {
            //const float sim_den = cos_sim[k*num_of_samples + l]; // cosine_similarity(z[k], z[l], feature_size);
            //denominator += expf(sim_den / temperature);
            denominator += exp_cos_sim[k*num_of_samples + l];
        }
    }

    float result = numerator / denominator;
    return result;
}

// i - id of the current sample in mini_batch
// labels[num_of_samples] - array with class_id for each sample in the current mini_batch
// z[feature_size][num_of_samples] - array of arrays with contrastive features (output of conv-layer, f.e. 128 floats for each sample)
// delta[feature_size] - array with deltas for backpropagation
// temperature - scalar temperature param (temperature > 0), f.e. temperature = 0.07: Supervised Contrastive Learning
void grad_contrastive_loss_positive(size_t i, int *labels, size_t num_of_samples, float **z, unsigned int feature_size, float temperature, float *cos_sim, float *p_constrastive, float *delta, int wh)
{
    const float vec_len = math_vector_length(z[i], feature_size);
    size_t j;
    float N = 0;
    for (j = 0; j < num_of_samples; ++j) {
        if (labels[i] == labels[j]) N++;
    }
    if (N == 0 || temperature == 0 || vec_len == 0) {
        fprintf(stderr, " Error: N == 0 || temperature == 0 || vec_len == 0. N=%f, temperature=%f, vec_len=%f \n", N, temperature, vec_len);
        getchar();
    }
    const float mult = 1 / ((N - 1) * temperature * vec_len);

    for (j = 0; j < num_of_samples; ++j) {
        //if (i != j && (i/2) == (j/2)) {
        if (i != j && labels[i] == labels[j]) {
            //printf(" i = %d, j = %d, num_of_samples = %d, labels[i] = %d, labels[j] = %d \n",
            //    i, j, num_of_samples, labels[i], labels[j]);
            const float sim = cos_sim[i*num_of_samples + j];        // cosine_similarity(z[i], z[j], feature_size);
            const float P = p_constrastive[i*num_of_samples + j];   // P_constrastive(i, j, labels, num_of_samples, z, feature_size, temperature, cos_sim);
            //const float custom_pos_mult = 1 - sim;

            int m;
            for (m = 0; m < feature_size; ++m) {
                const float d = mult*(sim * z[i][m] - z[j][m]) * (1 - P); // good
                //const float d = mult*(sim * z[j][m] - z[j][m]) * (1 - P); // bad
               // printf(" pos: z[j][m] = %f, z[i][m] = %f, d = %f, sim = %f \n", z[j][m], z[i][m], d, sim);
                const int out_i = m * wh;
                delta[out_i] -= d;
            }
        }
    }
}

// i - id of the current sample in mini_batch
// labels[num_of_samples] - array with class_id for each sample in the current mini_batch
// z[feature_size][num_of_samples] - array of arrays with contrastive features (output of conv-layer, f.e. 128 floats for each sample)
// delta[feature_size] - array with deltas for backpropagation
// temperature - scalar temperature param (temperature > 0), f.e. temperature = 0.07: Supervised Contrastive Learning
void grad_contrastive_loss_negative(size_t i, int *labels, size_t num_of_samples, float **z, unsigned int feature_size, float temperature, float *cos_sim, float *p_constrastive, float *delta, int wh)
{
    const float vec_len = math_vector_length(z[i], feature_size);
    size_t j;
    float N = 0;
    for (j = 0; j < num_of_samples; ++j) {
        if (labels[i] == labels[j]) N++;
    }
    if (N == 0 || temperature == 0 || vec_len == 0) {
        fprintf(stderr, " Error: N == 0 || temperature == 0 || vec_len == 0. N=%f, temperature=%f, vec_len=%f \n", N, temperature, vec_len);
        getchar();
    }
    const float mult = 1 / ((N - 1) * temperature * vec_len);

    for (j = 0; j < num_of_samples; ++j) {
        //if (i != j && (i/2) == (j/2)) {
        if (i != j && labels[i] == labels[j]) {

            size_t k;
            for (k = 0; k < num_of_samples; ++k) {
                //if (k != i && k != j && labels[k] != labels[i]) {
                if (k != i && k != j && labels[k] >= 0) {
                    const float sim = cos_sim[i*num_of_samples + k];        // cosine_similarity(z[i], z[k], feature_size);
                    const float P = p_constrastive[i*num_of_samples + k];   // P_constrastive(i, k, labels, num_of_samples, z, feature_size, temperature, cos_sim);
                    //const float custom_pos_mult = 1 + sim;

                    int m;
                    for (m = 0; m < feature_size; ++m) {
                        const float d = mult*(z[k][m] - sim * z[i][m]) * P;   // good
                        //const float d = mult*(z[k][m] - sim * z[k][m]) * P; // bad
                        //printf(" neg: z[k][m] = %f, z[i][m] = %f, d = %f, sim = %f \n", z[k][m], z[i][m], d, sim);
                        const int out_i = m * wh;
                        delta[out_i] -= d;
                    }
                }
            }
        }
    }
}