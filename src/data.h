#ifndef DATA_H
#define DATA_H

#include "darknet.h"
#include "darknet.h"
#include "matrix.h"
#include "list.h"
#include "image.h"
#ifdef __cplusplus
extern "C" {
#endif
#include "tree.h"

list *get_paths(char *filename);
char **get_labels(char *filename);
char **get_labels_custom(char *filename, int *size);
#ifdef __cplusplus
}

#endif
#endif
