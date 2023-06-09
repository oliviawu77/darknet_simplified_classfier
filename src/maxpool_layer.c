#include "maxpool_layer.h"
#include "convolutional_layer.h"
#include "utils.h"
#include "gemm.h"
#include <stdio.h>

maxpool_layer make_maxpool_layer(int batch, int h, int w, int c, int size, int stride_x, int stride_y, int padding, int maxpool_depth, int out_channels, int antialiasing, int avgpool, int train)
{
    maxpool_layer l = { (LAYER_TYPE)0 };
    l.avgpool = avgpool;
    //remove unused part
    l.type = MAXPOOL;
    l.train = train;

    const int blur_stride_x = stride_x;
    const int blur_stride_y = stride_y;
    l.antialiasing = antialiasing;
    if (antialiasing) {
        stride_x = stride_y = l.stride = l.stride_x = l.stride_y = 1; // use stride=1 in host-layer
    }

    l.batch = batch;
    l.h = h;
    l.w = w;
    l.c = c;
    l.pad = padding;
    l.maxpool_depth = maxpool_depth;
    l.out_channels = out_channels;
    if (maxpool_depth) {
        l.out_c = out_channels;
        l.out_w = l.w;
        l.out_h = l.h;
    }
    else {
        l.out_w = (w + padding - size) / stride_x + 1;
        l.out_h = (h + padding - size) / stride_y + 1;
        l.out_c = c;
    }
    l.outputs = l.out_h * l.out_w * l.out_c;
    l.inputs = h*w*c;
    l.size = size;
    l.stride = stride_x;
    l.stride_x = stride_x;
    l.stride_y = stride_y;
    int output_size = l.out_h * l.out_w * l.out_c * batch;

    //remove unused part
    l.output = (float*)xcalloc(output_size, sizeof(float));

    //remove avgpool

    l.forward = forward_maxpool_layer;
    //remove backward layer
    //l.backward = backward_maxpool_layer;
    

	l.bflops = (l.size*l.size*l.c * l.out_h*l.out_w) / 1000000000.;
    
    //remove unused part
    if (maxpool_depth)
        fprintf(stderr, "max-depth         %2dx%2d/%2d   %4d x%4d x%4d -> %4d x%4d x%4d %5.3f BF\n", size, size, stride_x, w, h, c, l.out_w, l.out_h, l.out_c, l.bflops);
    else if (stride_x == stride_y)
        fprintf(stderr, "max               %2dx%2d/%2d   %4d x%4d x%4d -> %4d x%4d x%4d %5.3f BF\n", size, size, stride_x, w, h, c, l.out_w, l.out_h, l.out_c, l.bflops);
    else
        fprintf(stderr, "max              %2dx%2d/%2dx%2d %4d x%4d x%4d -> %4d x%4d x%4d %5.3f BF\n", size, size, stride_x, stride_y, w, h, c, l.out_w, l.out_h, l.out_c, l.bflops);
    
    //remove unused part

    return l;
}


void forward_maxpool_layer(const maxpool_layer l, network_state state)
{
    if (l.maxpool_depth)
    {
        int b, i, j, k, g;
        for (b = 0; b < l.batch; ++b) {
            #pragma omp parallel for
            for (i = 0; i < l.h; ++i) {
                for (j = 0; j < l.w; ++j) {
                    for (g = 0; g < l.out_c; ++g)
                    {
                        int out_index = j + l.w*(i + l.h*(g + l.out_c*b));
                        float max = -FLT_MAX;
                        int max_i = -1;

                        for (k = g; k < l.c; k += l.out_c)
                        {
                            int in_index = j + l.w*(i + l.h*(k + l.c*b));
                            float val = state.input[in_index];

                            max_i = (val > max) ? in_index : max_i;
                            max = (val > max) ? val : max;
                        }
                        l.output[out_index] = max;
                        if (l.indexes) l.indexes[out_index] = max_i;
                    }
                }
            }
        }
        return;
    }


    if (!state.train && l.stride_x == l.stride_y) {
        forward_maxpool_layer_avx(state.input, l.output, l.indexes, l.size, l.w, l.h, l.out_w, l.out_h, l.c, l.pad, l.stride, l.batch);
    }
    //remove unused part
}
