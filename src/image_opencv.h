#ifndef IMAGE_OPENCV_H
#define IMAGE_OPENCV_H

#include "image.h"
#include "matrix.h"

#ifdef __cplusplus
extern "C" {
#endif


void show_opencv_info();
int wait_key_cv(int delay);
int wait_until_press_key_cv();
void destroy_all_windows_cv();
void resize_window_cv(char const* window_name, int width, int height);


#ifdef __cplusplus
}
#endif

#endif // IMAGE_OPENCV_H
