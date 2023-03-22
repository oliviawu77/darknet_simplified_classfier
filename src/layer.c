#include "layer.h"
#include <stdlib.h>

void free_sublayer(layer *l)
{
    if (l) {
        free_layer(*l);
        free(l);
    }
}

void free_layer(layer l)
{
    free_layer_custom(l, 0);
}

void free_layer_custom(layer l, int keep_cudnn_desc)
{
    if (l.share_layer != NULL) return;    // don't free shared layers
    if (l.antialiasing) {
        free_sublayer(l.input_layer);
    }
    //remove unused part
    if (l.mask)               free(l.mask);
    if (l.classes_multipliers)free(l.classes_multipliers);
    if (l.cweights)           free(l.cweights);
    if (l.indexes)            free(l.indexes);
    if (l.input_layers)       free(l.input_layers);
    if (l.input_sizes)        free(l.input_sizes);
    if (l.layers_output)      free(l.layers_output);
    if (l.layers_delta)       free(l.layers_delta);
    if (l.map)                free(l.map);
    if (l.rand)               free(l.rand);
    if (l.cost)               free(l.cost);
    if (l.labels && !l.detection) free(l.labels);
    if (l.class_ids && !l.detection) free(l.class_ids);
    if (l.cos_sim)            free(l.cos_sim);
    if (l.exp_cos_sim)        free(l.exp_cos_sim);
    if (l.p_constrastive)     free(l.p_constrastive);
    if (l.embedding_output)   free(l.embedding_output);
    if (l.state)              free(l.state);
    if (l.prev_state)         free(l.prev_state);
    if (l.forgot_state)       free(l.forgot_state);
    if (l.forgot_delta)       free(l.forgot_delta);
    if (l.state_delta)        free(l.state_delta);
    if (l.concat)             free(l.concat);
    if (l.concat_delta)       free(l.concat_delta);
    if (l.binary_weights)     free(l.binary_weights);
    if (l.biases)             free(l.biases), l.biases = NULL;
    if (l.bias_updates)       free(l.bias_updates), l.bias_updates = NULL;
    if (l.scales)             free(l.scales), l.scales = NULL;
    if (l.scale_updates)      free(l.scale_updates), l.scale_updates = NULL;
    if (l.biases_ema)         free(l.biases_ema), l.biases = NULL;
    if (l.scales_ema)         free(l.scales_ema), l.scales = NULL;
    if (l.weights_ema)        free(l.weights_ema), l.weights = NULL;
    if (l.weights)            free(l.weights), l.weights = NULL;
    if (l.weight_updates)     free(l.weight_updates), l.weight_updates = NULL;
    if (l.align_bit_weights)  free(l.align_bit_weights);
    if (l.mean_arr)           free(l.mean_arr);

    if (l.delta)              free(l.delta), l.delta = NULL;
    if (l.output)             free(l.output), l.output = NULL;
    if (l.activation_input)   free(l.activation_input), l.activation_input = NULL;
    if (l.squared)            free(l.squared);
    if (l.norms)              free(l.norms);
    if (l.spatial_mean)       free(l.spatial_mean);
    if (l.mean)               free(l.mean), l.mean = NULL;
    if (l.variance)           free(l.variance), l.variance = NULL;
    if (l.mean_delta)         free(l.mean_delta), l.mean_delta = NULL;
    if (l.variance_delta)     free(l.variance_delta), l.variance_delta = NULL;
    if (l.rolling_mean)       free(l.rolling_mean), l.rolling_mean = NULL;
    if (l.rolling_variance)   free(l.rolling_variance), l.rolling_variance = NULL;
    if (l.x)                  free(l.x);
    if (l.x_norm)             free(l.x_norm);
    if (l.m)                  free(l.m);
    if (l.v)                  free(l.v);
    if (l.z_cpu)              free(l.z_cpu);
    if (l.r_cpu)              free(l.r_cpu);
    if (l.binary_input)       free(l.binary_input);
    if (l.bin_re_packed_input) free(l.bin_re_packed_input);
    if (l.t_bit_input)        free(l.t_bit_input);
    if (l.loss)               free(l.loss);

    // CONV-LSTM
    if (l.f_cpu)               free(l.f_cpu);
    if (l.i_cpu)               free(l.i_cpu);
    if (l.g_cpu)               free(l.g_cpu);
    if (l.o_cpu)               free(l.o_cpu);
    if (l.c_cpu)               free(l.c_cpu);
    if (l.h_cpu)               free(l.h_cpu);
    if (l.temp_cpu)            free(l.temp_cpu);
    if (l.temp2_cpu)           free(l.temp2_cpu);
    if (l.temp3_cpu)           free(l.temp3_cpu);
    if (l.dc_cpu)              free(l.dc_cpu);
    if (l.dh_cpu)              free(l.dh_cpu);
    if (l.prev_state_cpu)      free(l.prev_state_cpu);
    if (l.prev_cell_cpu)       free(l.prev_cell_cpu);
    if (l.stored_c_cpu)        free(l.stored_c_cpu);
    if (l.stored_h_cpu)        free(l.stored_h_cpu);
    if (l.cell_cpu)            free(l.cell_cpu);

}
