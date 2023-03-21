#include "data.h"
#include "utils.h"
#include "image.h"
#include "box.h"
#include "http_stream.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern int check_mistakes;

list *get_paths(char *filename)
{
    char *path;
    FILE *file = fopen(filename, "r");
    if(!file) file_error(filename);
    list *lines = make_list();
    while((path=fgetl(file))){
        list_insert(lines, path);
    }
    fclose(file);
    return lines;
}


char **get_labels_custom(char *filename, int *size)
{
    list *plist = get_paths(filename);
    if(size) *size = plist->size;
    char **labels = (char **)list_to_array(plist);
    free_list(plist);
    return labels;
}

char **get_labels(char *filename)
{
    return get_labels_custom(filename, NULL);
}
