#ifndef CONVOLUTIONAL_LAYER_H
#define CONVOLUTIONAL_LAYER_H

#include "image.h"
#include "activations.h"
#include "layer.h"
#include "network.h"

typedef layer convolutional_layer;

#ifdef __cplusplus
extern "C" {
#endif

void free_convolutional_batchnorm(convolutional_layer *l);

size_t get_convolutional_workspace_size(layer l);
convolutional_layer make_convolutional_layer(int batch, int steps, int h, int w, int c, int n, int groups, int size, int stride_x, int stride_y, int dilation, int padding, ACTIVATION activation, int batch_normalize, int binary, int xnor, int adam, int use_bin_output, int index, int antialiasing, convolutional_layer *share_layer, int assisted_excitation, int deform, int train);
void forward_convolutional_layer(const convolutional_layer layer, network_state state);

void add_bias(float *output, float *biases, int batch, int n, int size);

int convolutional_out_height(convolutional_layer layer);
int convolutional_out_width(convolutional_layer layer);

#ifdef __cplusplus
}
#endif

#endif
