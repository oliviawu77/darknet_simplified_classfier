#include "cost_layer.h"
#include "utils.h"
#include "blas.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

COST_TYPE get_cost_type(char *s)
{
    if (strcmp(s, "sse")==0) return SSE;
    if (strcmp(s, "masked")==0) return MASKED;
    if (strcmp(s, "smooth")==0) return SMOOTH;
    fprintf(stderr, "Couldn't find cost type %s, going with SSE\n", s);
    return SSE;
}


cost_layer make_cost_layer(int batch, int inputs, COST_TYPE cost_type, float scale)
{
    fprintf(stderr, "cost                                           %4d\n",  inputs);
    cost_layer l = { (LAYER_TYPE)0 };
    l.type = COST;

    l.scale = scale;
    l.batch = batch;
    l.inputs = inputs;
    l.outputs = inputs;
    l.cost_type = cost_type;
    l.delta = (float*)xcalloc(inputs * batch, sizeof(float));
    l.output = (float*)xcalloc(inputs * batch, sizeof(float));
    l.cost = (float*)xcalloc(1, sizeof(float));

    l.forward = forward_cost_layer;
    //remove backward layer
    return l;
}

void forward_cost_layer(cost_layer l, network_state state)
{
    if (!state.truth) return;
    
    //remove unused part

    l2_cpu(l.batch*l.inputs, state.input, state.truth, l.delta, l.output);

    l.cost[0] = sum_array(l.output, l.batch*l.inputs);
}

