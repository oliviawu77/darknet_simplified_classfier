#include "image_opencv.h"
#include <iostream>

extern "C" void show_opencv_info()
{
    std::cerr << " OpenCV isn't used - data augmentation will be slow \n";
}
extern "C" int wait_key_cv(int delay) { return 0; }
extern "C" int wait_until_press_key_cv() { return 0; }
extern "C" void destroy_all_windows_cv() {}
extern "C" void resize_window_cv(char const* window_name, int width, int height) {}
