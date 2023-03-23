#include "softmax_layer.h"
#include "blas.h"
#include "utils.h"
#include "blas.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

#define SECRET_NUM -1234

softmax_layer make_softmax_layer(int batch, int inputs, int groups)
{
    assert(inputs%groups == 0);
    fprintf(stderr, "softmax                                        %4d\n",  inputs);
    softmax_layer l = { (LAYER_TYPE)0 };
    l.type = SOFTMAX;
    l.batch = batch;
    l.groups = groups;
    l.inputs = inputs;
    l.outputs = inputs;
    l.loss = (float*)xcalloc(inputs * batch, sizeof(float));
    l.output = (float*)xcalloc(inputs * batch, sizeof(float));
    l.delta = (float*)xcalloc(inputs * batch, sizeof(float));
    l.cost = (float*)xcalloc(1, sizeof(float));

    l.forward = forward_softmax_layer;
    //l.backward = backward_softmax_layer;

    return l;
}

void forward_softmax_layer(const softmax_layer l, network_state net)
{
    //remove softmax tree
    softmax_cpu(net.input, l.inputs/l.groups, l.batch, l.inputs, l.groups, l.inputs/l.groups, 1, l.temperature, l.output);
    //remove softmax_x_ent_cpu()
}

