#define _XOPEN_SOURCE
#include "http_stream.h"

#include <chrono>
#include <iostream>

static std::chrono::steady_clock::time_point steady_start, steady_end;
static double total_time;

double get_time_point() {
    std::chrono::steady_clock::time_point current_time = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(current_time.time_since_epoch()).count();
}




