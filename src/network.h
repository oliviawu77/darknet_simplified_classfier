// Oh boy, why am I about to do this....
#ifndef NETWORK_H
#define NETWORK_H
#include "darknet.h"

#include <stdint.h>
#include "layer.h"


#include "image.h"
#include "data.h"
#include "tree.h"

#ifdef __cplusplus
extern "C" {
#endif



void compare_networks(network n1, network n2, data d);
char *get_layer_string(LAYER_TYPE a);

network make_network(int n);
void forward_network(network net, network_state state);

float train_network_batch(network net, data d, int n);


float *get_network_output(network net);
float *get_network_output_layer(network net, int i);
float *get_network_delta_layer(network net, int i);
float *get_network_delta(network net);
int get_network_output_size_layer(network net, int i);
int get_network_output_size(network net);
void set_batch_network(network *net, int b);

int get_network_nuisance(network net);
int get_network_background(network net);

network combine_train_valid_networks(network net_train, network net_map);
void copy_weights_net(network net_train, network *net_map);
void free_network_recurrent_state(network net);
void randomize_network_recurrent_state(network net);
void remember_network_recurrent_state(network net);
void restore_network_recurrent_state(network net);
int is_ema_initialized(network net);
void ema_update(network net, float ema_alpha);
void ema_apply(network net);
void reject_similar_weights(network net, float sim_threshold);


#ifdef __cplusplus
}
#endif

#endif
