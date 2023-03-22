#include "convolutional_layer.h"
#include "utils.h"
#include "im2col.h"
#include "col2im.h"
#include "blas.h"
#include "gemm.h"
#include "box.h"
#include <stdio.h>
#include <time.h>

#ifdef AI2
#include "xnor_layer.h"
#endif

#ifdef __cplusplus
#define PUT_IN_REGISTER
#else
#define PUT_IN_REGISTER register
#endif

#ifndef AI2
#define AI2 0
void forward_xnor_layer(layer l, network_state state);
#endif

void swap_binary(convolutional_layer *l)
{
    float *swap = l->weights;
    l->weights = l->binary_weights;
    l->binary_weights = swap;

    #ifdef GPU
    swap = l->weights_gpu;
    l->weights_gpu = l->binary_weights_gpu;
    l->binary_weights_gpu = swap;
    #endif
}

void binarize_weights(float *weights, int n, int size, float *binary)
{
    int i, f;
    for(f = 0; f < n; ++f){
        float mean = 0;
        for(i = 0; i < size; ++i){
            mean += fabs(weights[f*size + i]);
        }
        mean = mean / size;
        for(i = 0; i < size; ++i){
            binary[f*size + i] = (weights[f*size + i] > 0) ? mean: -mean;
        }
    }
}

void binarize_cpu(float *input, int n, float *binary)
{
    int i;
    for(i = 0; i < n; ++i){
        binary[i] = (input[i] > 0) ? 1 : -1;
    }
}

int convolutional_out_height(convolutional_layer l)
{
    return (l.h + 2*l.pad - l.size) / l.stride_y + 1;
}

int convolutional_out_width(convolutional_layer l)
{
    return (l.w + 2*l.pad - l.size) / l.stride_x + 1;
}

size_t get_workspace_size32(layer l){

    if (l.xnor) {
        size_t re_packed_input_size = l.c * l.w * l.h * sizeof(float);
        size_t workspace_size = (size_t)l.bit_align*l.size*l.size*l.c * sizeof(float);
        if (workspace_size < re_packed_input_size) workspace_size = re_packed_input_size;
        return workspace_size;
    }
    return (size_t)l.out_h*l.out_w*l.size*l.size*(l.c / l.groups)*sizeof(float);
}

size_t get_workspace_size16(layer l) {

    return 0;

}

size_t get_convolutional_workspace_size(layer l) {
    size_t workspace_size = get_workspace_size32(l);
    size_t workspace_size16 = get_workspace_size16(l);
    if (workspace_size16 > workspace_size) workspace_size = workspace_size16;
    return workspace_size;
}

void free_convolutional_batchnorm(convolutional_layer *l)
{
    if (!l->share_layer) {
        if (l->scales)          free(l->scales),            l->scales = NULL;
        if (l->scale_updates)   free(l->scale_updates),     l->scale_updates = NULL;
        if (l->mean)            free(l->mean),              l->mean = NULL;
        if (l->variance)        free(l->variance),          l->variance = NULL;
        if (l->mean_delta)      free(l->mean_delta),        l->mean_delta = NULL;
        if (l->variance_delta)  free(l->variance_delta),    l->variance_delta = NULL;
        if (l->rolling_mean)    free(l->rolling_mean),      l->rolling_mean = NULL;
        if (l->rolling_variance) free(l->rolling_variance),  l->rolling_variance = NULL;
        if (l->x)               free(l->x),                 l->x = NULL;
        if (l->x_norm)          free(l->x_norm),            l->x_norm = NULL;

    }
}

convolutional_layer make_convolutional_layer(int batch, int steps, int h, int w, int c, int n, int groups, int size, int stride_x, int stride_y, int dilation, int padding, ACTIVATION activation, int batch_normalize, int binary, int xnor, int adam, int use_bin_output, int index, int antialiasing, convolutional_layer *share_layer, int assisted_excitation, int deform, int train)
{
    int total_batch = batch*steps;
    int i;
    convolutional_layer l = { (LAYER_TYPE)0 };
    l.type = CONVOLUTIONAL;
    l.train = train;

    if (xnor) groups = 1;   // disable groups for XNOR-net
    if (groups < 1) groups = 1;

    const int blur_stride_x = stride_x;
    const int blur_stride_y = stride_y;
    l.antialiasing = antialiasing;
    if (antialiasing) {
        stride_x = stride_y = l.stride = l.stride_x = l.stride_y = 1; // use stride=1 in host-layer
    }

    l.wait_stream_id = -1;
    l.deform = deform;
    l.assisted_excitation = assisted_excitation;
    l.share_layer = share_layer;
    l.index = index;
    l.h = h;
    l.w = w;
    l.c = c;
    l.groups = groups;
    l.n = n;
    l.binary = binary;
    l.xnor = xnor;
    l.use_bin_output = use_bin_output;
    l.batch = batch;
    l.steps = steps;
    l.stride = stride_x;
    l.stride_x = stride_x;
    l.stride_y = stride_y;
    l.dilation = dilation;
    l.size = size;
    l.pad = padding;
    l.batch_normalize = batch_normalize;
    l.learning_rate_scale = 1;
    l.nweights = (c / groups) * n * size * size;

    if (l.share_layer) {
        if (l.size != l.share_layer->size || l.nweights != l.share_layer->nweights || l.c != l.share_layer->c || l.n != l.share_layer->n) {
            printf(" Layer size, nweights, channels or filters don't match for the share_layer");
            getchar();
        }

        l.weights = l.share_layer->weights;
        l.weight_updates = l.share_layer->weight_updates;

        l.biases = l.share_layer->biases;
        l.bias_updates = l.share_layer->bias_updates;
    }
    else {
        l.weights = (float*)xcalloc(l.nweights, sizeof(float));
        l.biases = (float*)xcalloc(n, sizeof(float));

        if (train) {
            l.weight_updates = (float*)xcalloc(l.nweights, sizeof(float));
            l.bias_updates = (float*)xcalloc(n, sizeof(float));

            l.weights_ema = (float*)xcalloc(l.nweights, sizeof(float));
            l.biases_ema = (float*)xcalloc(n, sizeof(float));
        }
    }

    // float scale = 1./sqrt(size*size*c);
    float scale = sqrt(2./(size*size*c/groups));
    if (l.activation == NORM_CHAN || l.activation == NORM_CHAN_SOFTMAX || l.activation == NORM_CHAN_SOFTMAX_MAXVAL) {
        for (i = 0; i < l.nweights; ++i) l.weights[i] = 1;   // rand_normal();
    }
    else {
        for (i = 0; i < l.nweights; ++i) l.weights[i] = scale*rand_uniform(-1, 1);   // rand_normal();
    }
    int out_h = convolutional_out_height(l);
    int out_w = convolutional_out_width(l);
    l.out_h = out_h;
    l.out_w = out_w;
    l.out_c = n;
    l.outputs = l.out_h * l.out_w * l.out_c;
    l.inputs = l.w * l.h * l.c;
    l.activation = activation;

    l.output = (float*)xcalloc(total_batch*l.outputs, sizeof(float));
#ifndef GPU
    if (train) l.delta = (float*)xcalloc(total_batch*l.outputs, sizeof(float));
#endif  // not GPU

    l.forward = forward_convolutional_layer;
    //remove backward layer and update layer
    if(binary){
        l.binary_weights = (float*)xcalloc(l.nweights, sizeof(float));
        l.cweights = (char*)xcalloc(l.nweights, sizeof(char));
        l.scales = (float*)xcalloc(n, sizeof(float));
    }
    if(xnor){
        l.binary_weights = (float*)xcalloc(l.nweights, sizeof(float));
        l.binary_input = (float*)xcalloc(l.inputs * l.batch, sizeof(float));

        int align = 32;// 8;
        int src_align = l.out_h*l.out_w;
        l.bit_align = src_align + (align - src_align % align);

        l.mean_arr = (float*)xcalloc(l.n, sizeof(float));

        const size_t new_c = l.c / 32;
        size_t in_re_packed_input_size = new_c * l.w * l.h + 1;
        l.bin_re_packed_input = (uint32_t*)xcalloc(in_re_packed_input_size, sizeof(uint32_t));

        l.lda_align = 256;  // AVX2
        int k = l.size*l.size*l.c;
        size_t k_aligned = k + (l.lda_align - k%l.lda_align);
        size_t t_bit_input_size = k_aligned * l.bit_align / 8;
        l.t_bit_input = (char*)xcalloc(t_bit_input_size, sizeof(char));
    }

    if(batch_normalize){
        if (l.share_layer) {
            l.scales = l.share_layer->scales;
            l.scale_updates = l.share_layer->scale_updates;
            l.mean = l.share_layer->mean;
            l.variance = l.share_layer->variance;
            l.mean_delta = l.share_layer->mean_delta;
            l.variance_delta = l.share_layer->variance_delta;
            l.rolling_mean = l.share_layer->rolling_mean;
            l.rolling_variance = l.share_layer->rolling_variance;
        }
        else {
            l.scales = (float*)xcalloc(n, sizeof(float));
            for (i = 0; i < n; ++i) {
                l.scales[i] = 1;
            }
            if (train) {
                l.scales_ema = (float*)xcalloc(n, sizeof(float));
                l.scale_updates = (float*)xcalloc(n, sizeof(float));

                l.mean = (float*)xcalloc(n, sizeof(float));
                l.variance = (float*)xcalloc(n, sizeof(float));

                l.mean_delta = (float*)xcalloc(n, sizeof(float));
                l.variance_delta = (float*)xcalloc(n, sizeof(float));
            }
            l.rolling_mean = (float*)xcalloc(n, sizeof(float));
            l.rolling_variance = (float*)xcalloc(n, sizeof(float));
        }

#ifndef GPU
        if (train) {
            l.x = (float*)xcalloc(total_batch * l.outputs, sizeof(float));
            l.x_norm = (float*)xcalloc(total_batch * l.outputs, sizeof(float));
        }
#endif  // not GPU
    }

#ifndef GPU
    if (l.activation == SWISH || l.activation == MISH || l.activation == HARD_MISH) l.activation_input = (float*)calloc(total_batch*l.outputs, sizeof(float));
#endif  // not GPU

    if(adam){
        l.adam = 1;
        l.m = (float*)xcalloc(l.nweights, sizeof(float));
        l.v = (float*)xcalloc(l.nweights, sizeof(float));
        l.bias_m = (float*)xcalloc(n, sizeof(float));
        l.scale_m = (float*)xcalloc(n, sizeof(float));
        l.bias_v = (float*)xcalloc(n, sizeof(float));
        l.scale_v = (float*)xcalloc(n, sizeof(float));
    }


    l.workspace_size = get_convolutional_workspace_size(l);

    //fprintf(stderr, "conv  %5d %2d x%2d /%2d  %4d x%4d x%4d   ->  %4d x%4d x%4d\n", n, size, size, stride, w, h, c, l.out_w, l.out_h, l.out_c);
    l.bflops = (2.0 * l.nweights * l.out_h*l.out_w) / 1000000000.;
    if (l.xnor) l.bflops = l.bflops / 32;
    if (l.xnor && l.use_bin_output) fprintf(stderr, "convXB");
    else if (l.xnor) fprintf(stderr, "convX ");
    else if (l.share_layer) fprintf(stderr, "convS ");
    else if (l.assisted_excitation) fprintf(stderr, "convAE");
    else fprintf(stderr, "conv  ");

    if (groups > 1) fprintf(stderr, "%5d/%4d ", n, groups);
    else           fprintf(stderr, "%5d      ", n);

    if (stride_x != stride_y) fprintf(stderr, "%2dx%2d/%2dx%2d ", size, size, stride_x, stride_y);
    else {
        if (dilation > 1) fprintf(stderr, "%2d x%2d/%2d(%1d)", size, size, stride_x, dilation);
        else             fprintf(stderr, "%2d x%2d/%2d   ", size, size, stride_x);
    }

    fprintf(stderr, "%4d x%4d x%4d -> %4d x%4d x%4d %5.3f BF\n", w, h, c, l.out_w, l.out_h, l.out_c, l.bflops);


    if (l.antialiasing) {
        printf("AA:  ");
        l.input_layer = (layer*)calloc(1, sizeof(layer));
        int blur_size = 3;
        int blur_pad = blur_size / 2;
        if (l.antialiasing == 2) {
            blur_size = 2;
            blur_pad = 0;
        }
        *(l.input_layer) = make_convolutional_layer(batch, steps, out_h, out_w, n, n, n, blur_size, blur_stride_x, blur_stride_y, 1, blur_pad, LINEAR, 0, 0, 0, 0, 0, index, 0, NULL, 0, 0, train);
        const int blur_nweights = n * blur_size * blur_size;  // (n / n) * n * blur_size * blur_size;
        int i;
        if (blur_size == 2) {
            for (i = 0; i < blur_nweights; i += (blur_size*blur_size)) {
                l.input_layer->weights[i + 0] = 1 / 4.f;
                l.input_layer->weights[i + 1] = 1 / 4.f;
                l.input_layer->weights[i + 2] = 1 / 4.f;
                l.input_layer->weights[i + 3] = 1 / 4.f;
            }
        }
        else {
            for (i = 0; i < blur_nweights; i += (blur_size*blur_size)) {
                l.input_layer->weights[i + 0] = 1 / 16.f;
                l.input_layer->weights[i + 1] = 2 / 16.f;
                l.input_layer->weights[i + 2] = 1 / 16.f;

                l.input_layer->weights[i + 3] = 2 / 16.f;
                l.input_layer->weights[i + 4] = 4 / 16.f;
                l.input_layer->weights[i + 5] = 2 / 16.f;

                l.input_layer->weights[i + 6] = 1 / 16.f;
                l.input_layer->weights[i + 7] = 2 / 16.f;
                l.input_layer->weights[i + 8] = 1 / 16.f;
            }
        }
        for (i = 0; i < n; ++i) l.input_layer->biases[i] = 0;
    }

    return l;
}

void add_bias(float *output, float *biases, int batch, int n, int size)
{
    int i,j,b;
    for(b = 0; b < batch; ++b){
        for(i = 0; i < n; ++i){
            for(j = 0; j < size; ++j){
                output[(b*n + i)*size + j] += biases[i];
            }
        }
    }
}

void get_mean_array(float *src, size_t size, size_t filters, float *mean_arr) {
    size_t i, counter;
    counter = 0;
    for (i = 0; i < size; i += size / filters) {
        mean_arr[counter++] = fabs(src[i]);
    }
}

void binary_align_weights(convolutional_layer *l)
{
    int m = l->n;   // (l->n / l->groups)
    int k = l->size*l->size*l->c;   // ->size*l->size*(l->c / l->groups)
    size_t new_lda = k + (l->lda_align - k % l->lda_align); // (k / 8 + 1) * 8;
    l->new_lda = new_lda;

    binarize_weights(l->weights, m, k, l->binary_weights);

    size_t align_weights_size = new_lda * m;
    l->align_bit_weights_size = align_weights_size / 8 + 1;
    float* align_weights = (float*)xcalloc(align_weights_size, sizeof(float));
    l->align_bit_weights = (char*)xcalloc(l->align_bit_weights_size, sizeof(char));

    size_t i, j;
    // align A without transpose
    for (i = 0; i < m; ++i) {
        for (j = 0; j < k; ++j) {
            align_weights[i*new_lda + j] = l->binary_weights[i*k + j];
        }
    }


    if (l->c % 32 == 0)
    //if(gpu_index < 0 && l->stride == 1 && l->pad == 1 && l->c % 32 == 0)
    //if (l->stride == 1 && l->pad == 1 && l->c % 32 == 0)
    {
        int fil, chan;
        const int items_per_filter = l->c * l->size * l->size;
        //const int dst_items_per_filter = new_lda;
        for (fil = 0; fil < l->n; ++fil)
        {
            for (chan = 0; chan < l->c; chan += 32)
            {
                const int items_per_channel = l->size*l->size;
                for (i = 0; i < items_per_channel; ++i)
                {
                    //uint32_t val = 0;
                    int c_pack;
                    for (c_pack = 0; c_pack < 32; ++c_pack) {
                        float src = l->binary_weights[fil*items_per_filter + (chan + c_pack)*items_per_channel + i];

                        //align_weights[fil*items_per_filter + chan*items_per_channel + i * 32 + c_pack] = src;

                        align_weights[fil*new_lda + chan*items_per_channel + i*32 + c_pack] = src;
                        //val |= (src << c);
                    }

                }
            }
        }

        //printf("\n l.index = %d \t aw[0] = %f, aw[1] = %f, aw[2] = %f, aw[3] = %f \n", l->index, align_weights[0], align_weights[1], align_weights[2], align_weights[3]);
        //memcpy(l->binary_weights, align_weights, (l->size * l->size * l->c * l->n) * sizeof(float));

        float_to_bit(align_weights, (unsigned char*)l->align_bit_weights, align_weights_size);

        //if (l->n >= 32)

        get_mean_array(l->binary_weights, m*k, l->n, l->mean_arr);
        //get_mean_array(l->binary_weights, m*new_lda, l->n, l->mean_arr);
    }
    else {
        float_to_bit(align_weights, (unsigned char*)l->align_bit_weights, align_weights_size);

        get_mean_array(l->binary_weights, m*k, l->n, l->mean_arr);
    }

    //l->mean_arr = calloc(l->n, sizeof(float));

    //get_mean_array(align_weights, align_weights_size, l->n, l->mean_arr);

    free(align_weights);
}

//copy from https://github.com/AlexeyAB/darknet/blob/bc810016a1f2eadf33a1ac800b64962a42f3d402/src/convolutional_layer.c
void forward_convolutional_layer(convolutional_layer l, network_state state)
{
    int out_h = convolutional_out_height(l);
    int out_w = convolutional_out_width(l);
    int i;

    fill_cpu(l.outputs*l.batch, 0, l.output, 1);

    if(l.xnor){
        binarize_weights(l.weights, l.n, l.c*l.size*l.size, l.binary_weights);
        swap_binary(&l);
        binarize_cpu(state.input, l.c*l.h*l.w*l.batch, l.binary_input);
        state.input = l.binary_input;
    }

    int m = l.n;
    int k = l.size*l.size*l.c;
    int n = out_h*out_w;


    float *a = l.weights;
    float *b = state.workspace;
    float *c = l.output;

    for(i = 0; i < l.batch; ++i){
        im2col_cpu(state.input, l.c, l.h, l.w, 
                l.size, l.stride, l.pad, b);
        gemm(0,0,m,n,k,1,a,k,b,n,1,c,n);
        c += n*m;
        state.input += l.c*l.h*l.w;
    }

    //remove batch normalize here
    add_bias(l.output, l.biases, l.batch, l.n, out_h*out_w);

    activate_array(l.output, m*n*l.batch, l.activation);
    if(l.binary || l.xnor) swap_binary(&l);
}

