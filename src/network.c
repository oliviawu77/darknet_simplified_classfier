#include "darknet.h"

#include <stdio.h>
#include <time.h>
#include <assert.h>

#include "network.h"
#include "image.h"
#include "data.h"
#include "utils.h"
#include "blas.h"

#include "convolutional_layer.h"
#include "maxpool_layer.h"
#include "avgpool_layer.h"
#include "cost_layer.h"
#include "softmax_layer.h"
#include "parser.h"

int get_current_batch(network net)
{
    int batch_num = (*net.seen)/(net.batch*net.subdivisions);
    return batch_num;
}

network make_network(int n)
{
    network net = {0};
    net.n = n;
    net.layers = (layer*)xcalloc(net.n, sizeof(layer));
    net.seen = (uint64_t*)xcalloc(1, sizeof(uint64_t));
    net.cuda_graph_ready = (int*)xcalloc(1, sizeof(int));
    net.badlabels_reject_threshold = (float*)xcalloc(1, sizeof(float));
    net.delta_rolling_max = (float*)xcalloc(1, sizeof(float));
    net.delta_rolling_avg = (float*)xcalloc(1, sizeof(float));
    net.delta_rolling_std = (float*)xcalloc(1, sizeof(float));
    net.cur_iteration = (int*)xcalloc(1, sizeof(int));
    net.total_bbox = (int*)xcalloc(1, sizeof(int));
    net.rewritten_bbox = (int*)xcalloc(1, sizeof(int));
    *net.rewritten_bbox = *net.total_bbox = 0;

    return net;
}

void forward_network(network net, network_state state)
{
    state.workspace = net.workspace;
    int i;
    for(i = 0; i < net.n; ++i){
        state.index = i;
        layer l = net.layers[i];
        if(l.delta && state.train && l.train){
            scal_cpu(l.outputs * l.batch, 0, l.delta, 1);
        }
        l.forward(l, state);
        state.input = l.output;
    }
}

float *get_network_output(network net)
{
    int i;
    for(i = net.n-1; i > 0; --i) if(net.layers[i].type != COST) break;
    return net.layers[i].output;
}



int recalculate_workspace_size(network *net)
{

    int i;
    size_t workspace_size = 0;
    for (i = 0; i < net->n; ++i) {
        layer l = net->layers[i];
        //printf(" %d: layer = %d,", i, l.type);
        if (l.type == CONVOLUTIONAL) {
            l.workspace_size = get_convolutional_workspace_size(l);
        }
        if (l.workspace_size > workspace_size) workspace_size = l.workspace_size;
        net->layers[i] = l;
    }


    free(net->workspace);
    net->workspace = (float*)xcalloc(1, workspace_size);
    //fprintf(stderr, " Done!\n");
    return 0;
}

void set_batch_network(network *net, int b)
{
    net->batch = b;
    int i;
    for(i = 0; i < net->n; ++i){
        net->layers[i].batch = b;
    }
    recalculate_workspace_size(net); // recalculate workspace size
}


int get_network_output_size(network net)
{
    int i;
    for(i = net.n-1; i > 0; --i) if(net.layers[i].type != COST) break;
    return net.layers[i].outputs;
}



float *network_predict(network net, float *input)
{
    network_state state = {0};
    state.net = net;
    state.index = 0;
    state.input = input;
    state.truth = 0;
    state.train = 0;
    state.delta = 0;
    forward_network(net, state);
    float *out = get_network_output(net);
    return out;
}



void free_network(network net)
{
    int i;
    for (i = 0; i < net.n; ++i) {
        free_layer(net.layers[i]);
    }
    free(net.layers);

    free(net.seq_scales);
    free(net.scales);
    free(net.steps);
    free(net.seen);
    free(net.cuda_graph_ready);
    free(net.badlabels_reject_threshold);
    free(net.delta_rolling_max);
    free(net.delta_rolling_avg);
    free(net.delta_rolling_std);
    free(net.cur_iteration);
    free(net.total_bbox);
    free(net.rewritten_bbox);


    free(net.workspace);
}

void fuse_conv_batchnorm(network net)
{
    int j;
    for (j = 0; j < net.n; ++j) {
        layer *l = &net.layers[j];

        if (l->type == CONVOLUTIONAL) {
            if (l->share_layer != NULL) {
                l->batch_normalize = 0;
            }

            if (l->batch_normalize) {
                int f;
                for (f = 0; f < l->n; ++f)
                {
                    l->biases[f] = l->biases[f] - (double)l->scales[f] * l->rolling_mean[f] / (sqrt((double)l->rolling_variance[f] + .00001));

                    double precomputed = l->scales[f] / (sqrt((double)l->rolling_variance[f] + .00001));

                    const size_t filter_size = l->size*l->size*l->c / l->groups;
                    int i;
                    for (i = 0; i < filter_size; ++i) {
                        int w_index = f*filter_size + i;

                        l->weights[w_index] *= precomputed;
                    }
                }
                free_convolutional_batchnorm(l);
                l->batch_normalize = 0;

            }
        }
    }
}








