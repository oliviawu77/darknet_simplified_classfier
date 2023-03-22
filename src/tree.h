#ifndef TREE_H
#define TREE_H
#include "darknet.h"

//typedef struct{
//    int *leaf;
//    int n;
//    int *parent;
//	int *child;
//    int *group;
//    char **name;
//
//    int groups;
//    int *group_size;
//    int *group_offset;
//} tree;

#ifdef __cplusplus
extern "C" {
#endif
//tree *read_tree(char *filename);
void hierarchy_predictions(float *predictions, int n, tree *hier, int only_leaves);

#ifdef __cplusplus
}
#endif
#endif
