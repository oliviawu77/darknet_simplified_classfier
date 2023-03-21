#ifndef PARSER_H
#define PARSER_H
#include "network.h"

#ifdef __cplusplus
extern "C" {
#endif
network parse_network_cfg(char *filename);
network parse_network_cfg_custom(char *filename, int batch, int time_steps);
void load_weights(network *net, char *filename);
void load_weights_upto(network *net, char *filename, int cutoff);

#ifdef __cplusplus
}
#endif
#endif
