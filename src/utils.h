#ifndef UTILS_H
#define UTILS_H
#include "darknet.h"
#include "list.h"

#include <stdio.h>
#include <time.h>
#include <stdbool.h>

#ifndef M_PI
#define M_PI       3.14159265358979323846   // pi
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define DARKNET_LOC __FILE__, __func__, __LINE__

LIB_API void top_k(float *a, int n, int k, int *index);

/* The "location" is the file, function, and line as defined by the DARKNET_LOC macro.
 * This is then printed when error() is called to terminate the instance of darknet.
 */

void *xmalloc_location(const size_t size, const char * const filename, const char * const funcname, const int line);
void *xcalloc_location(const size_t nmemb, const size_t size, const char * const filename, const char * const funcname, const int line);
void *xrealloc_location(void *ptr, const size_t size, const char * const filename, const char * const funcname, const int line);

#define xmalloc(s)      xmalloc_location(s, DARKNET_LOC)
#define xcalloc(m, s)   xcalloc_location(m, s, DARKNET_LOC)
#define xrealloc(p, s)  xrealloc_location(p, s, DARKNET_LOC)

void error(const char * const msg, const char * const filename, const char * const funcname, const int line);

void malloc_error(const size_t size, const char * const filename, const char * const funcname, const int line);
void calloc_error(const size_t size, const char * const filename, const char * const funcname, const int line);
void realloc_error(const size_t size, const char * const filename, const char * const funcname, const int line);

void file_error(const char * const s);
void strip(char *s);
char *fgetl(FILE *fp);
int constrain_int(int a, int min, int max);
float rand_uniform(float min, float max);
float sum_array(float *a, int n);

#define max_val_cmp(a,b) (((a) > (b)) ? (a) : (b))
#define min_val_cmp(a,b) (((a) < (b)) ? (a) : (b))

#ifdef __cplusplus
}
#endif

#endif
