#ifndef DATA_H
#define DATA_H

#include "darknet.h"
#include "darknet.h"
#include "list.h"
#include "image.h"
#ifdef __cplusplus
extern "C" {
#endif

list *get_paths(char *filename);
char **get_labels(char *filename);
char **get_labels_custom(char *filename, int *size);
#ifdef __cplusplus
}

#endif
#endif
