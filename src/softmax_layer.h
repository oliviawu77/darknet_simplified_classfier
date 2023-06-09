#ifndef SOFTMAX_LAYER_H
#define SOFTMAX_LAYER_H
#include "layer.h"
#include "network.h"

typedef layer softmax_layer;
typedef layer contrastive_layer;

#ifdef __cplusplus
extern "C" {
#endif
softmax_layer make_softmax_layer(int batch, int inputs, int groups);
void forward_softmax_layer(const softmax_layer l, network_state state);

#ifdef __cplusplus
}
#endif
#endif
