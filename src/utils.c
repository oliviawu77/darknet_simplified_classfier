#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#include <math.h>
#include <assert.h>
#include <float.h>
#include <limits.h>
#include "darkunistd.h"
#ifdef WIN32
#include "gettimeofday.h"
#else
#include <sys/time.h>
#include <sys/stat.h>
#endif


#ifndef USE_CMAKE_LIBS
#pragma warning(disable: 4996)
#endif

void *xmalloc_location(const size_t size, const char * const filename, const char * const funcname, const int line) {
    void *ptr=malloc(size);
    if(!ptr) {
        malloc_error(size, filename, funcname, line);
    }
    return ptr;
}

void *xcalloc_location(const size_t nmemb, const size_t size, const char * const filename, const char * const funcname, const int line) {
    void *ptr=calloc(nmemb, size);
    if(!ptr) {
        calloc_error(nmemb * size, filename, funcname, line);
    }
    return ptr;
}

void *xrealloc_location(void *ptr, const size_t size, const char * const filename, const char * const funcname, const int line) {
    ptr=realloc(ptr,size);
    if(!ptr) {
        realloc_error(size, filename, funcname, line);
    }
    return ptr;
}

void top_k(float *a, int n, int k, int *index)
{
    int i,j;
    for(j = 0; j < k; ++j) index[j] = -1;
    for(i = 0; i < n; ++i){
        int curr = i;
        for(j = 0; j < k; ++j){
            if((index[j] < 0) || a[curr] > a[index[j]]){
                int swap = curr;
                curr = index[j];
                index[j] = swap;
            }
        }
    }
}

void error(const char * const msg, const char * const filename, const char * const funcname, const int line)
{
    fprintf(stderr, "Darknet error location: %s, %s, line #%d\n", filename, funcname, line);
    perror(msg);
    exit(EXIT_FAILURE);
}

const char * size_to_IEC_string(const size_t size)
{
    const float bytes = (double)size;
    const float KiB = 1024;
    const float MiB = 1024 * KiB;
    const float GiB = 1024 * MiB;

    static char buffer[25];
    if (size < KiB)         sprintf(buffer, "%ld bytes", size);
    else if (size < MiB)    sprintf(buffer, "%1.1f KiB", bytes / KiB);
    else if (size < GiB)    sprintf(buffer, "%1.1f MiB", bytes / MiB);
    else                    sprintf(buffer, "%1.1f GiB", bytes / GiB);

    return buffer;
}

void malloc_error(const size_t size, const char * const filename, const char * const funcname, const int line)
{
    fprintf(stderr, "Failed to malloc %s\n", size_to_IEC_string(size));
    error("Malloc error - possibly out of CPU RAM", filename, funcname, line);
}

void calloc_error(const size_t size, const char * const filename, const char * const funcname, const int line)
{
    fprintf(stderr, "Failed to calloc %s\n", size_to_IEC_string(size));
    error("Calloc error - possibly out of CPU RAM", filename, funcname, line);
}

void realloc_error(const size_t size, const char * const filename, const char * const funcname, const int line)
{
    fprintf(stderr, "Failed to realloc %s\n", size_to_IEC_string(size));
    error("Realloc error - possibly out of CPU RAM", filename, funcname, line);
}

void file_error(const char * const s)
{
    fprintf(stderr, "Couldn't open file: %s\n", s);
    exit(EXIT_FAILURE);
}

void strip(char *s)
{
    size_t i;
    size_t len = strlen(s);
    size_t offset = 0;
    for(i = 0; i < len; ++i){
        char c = s[i];
        if(c==' '||c=='\t'||c=='\n'||c =='\r'||c==0x0d||c==0x0a) ++offset;
        else s[i-offset] = c;
    }
    s[len-offset] = '\0';
}

char *fgetl(FILE *fp)
{
    if(feof(fp)) return 0;
    size_t size = 512;
    char* line = (char*)xmalloc(size * sizeof(char));
    if(!fgets(line, size, fp)){
        free(line);
        return 0;
    }

    size_t curr = strlen(line);

    while((line[curr-1] != '\n') && !feof(fp)){
        if(curr == size-1){
            size *= 2;
            line = (char*)xrealloc(line, size * sizeof(char));
        }
        size_t readsize = size-curr;
        if(readsize > INT_MAX) readsize = INT_MAX-1;
        fgets(&line[curr], readsize, fp);
        curr = strlen(line);
    }
    if(curr >= 2)
        if(line[curr-2] == 0x0d) line[curr-2] = 0x00;

    if(curr >= 1)
        if(line[curr-1] == 0x0a) line[curr-1] = 0x00;

    return line;
}

float sum_array(float *a, int n)
{
    int i;
    float sum = 0;
    for(i = 0; i < n; ++i) sum += a[i];
    return sum;
}

int constrain_int(int a, int min, int max)
{
    if (a < min) return min;
    if (a > max) return max;
    return a;
}

float rand_uniform(float min, float max)
{
    if(max < min){
        float swap = min;
        min = max;
        max = swap;
    }

#if (RAND_MAX < 65536)
        int rnd = rand()*(RAND_MAX + 1) + rand();
        return ((float)rnd / (RAND_MAX*RAND_MAX) * (max - min)) + min;
#else
        return ((float)rand() / RAND_MAX * (max - min)) + min;
#endif
    //return (random_float() * (max - min)) + min;
}

static unsigned int x = 123456789, y = 362436069, z = 521288629;
