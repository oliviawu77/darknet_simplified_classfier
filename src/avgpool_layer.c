#include "avgpool_layer.h"
#include "utils.h"
#include <stdio.h>

avgpool_layer make_avgpool_layer(int batch, int w, int h, int c)
{
    fprintf(stderr, "avg                          %4d x%4d x%4d ->   %4d\n",  w, h, c, c);
    avgpool_layer l = { (LAYER_TYPE)0 };
    l.type = AVGPOOL;
    l.batch = batch;
    l.h = h;
    l.w = w;
    l.c = c;
    l.out_w = 1;
    l.out_h = 1;
    l.out_c = c;
    l.outputs = l.out_c;
    l.inputs = h*w*c;
    int output_size = l.outputs * batch;
    l.output = (float*)xcalloc(output_size, sizeof(float));
    l.delta = (float*)xcalloc(output_size, sizeof(float));
    l.forward = forward_avgpool_layer;
    //l.backward = backward_avgpool_layer;

    return l;
}

void forward_avgpool_layer(const avgpool_layer l, network_state state)
{
    int b,i,k;

    for(b = 0; b < l.batch; ++b){
        for(k = 0; k < l.c; ++k){
            int out_index = k + b*l.c;
            l.output[out_index] = 0;
            for(i = 0; i < l.h*l.w; ++i){
                int in_index = i + l.h*l.w*(k + b*l.c);
                l.output[out_index] += state.input[in_index];
            }
            l.output[out_index] /= l.h*l.w;
        }
    }
}

