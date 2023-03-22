#ifndef IMAGE_H
#define IMAGE_H
#include "darknet.h"

#include <stdlib.h>
#include <stdio.h>
#include <float.h>
#include <string.h>
#include <math.h>

//#include "image_opencv.h"

#include "box.h"
#ifdef __cplusplus
extern "C" {
#endif
/*
typedef struct {
    int w;
    int h;
    int c;
    float *data;
} image;
*/
void write_label(image a, int r, int c, image *characters, char *string, float *rgb);
void scale_image(image m, float s);

//LIB_API image make_image(int w, int h, int c);
image make_empty_image(int w, int h, int c);
image float_to_image(int w, int h, int c, float *data);
image copy_image(image p);
image load_image(char *filename, int w, int h, int c);
//LIB_API image load_image_color(char *filename, int w, int h);

//LIB_API void free_image(image m);
#ifdef __cplusplus
}
#endif

#endif
