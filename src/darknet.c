#include "darknet.h"
#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#if defined(_MSC_VER) && defined(_DEBUG)
#include <crtdbg.h>
#endif

#include "parser.h"
#include "utils.h"
#include "blas.h"


extern void predict_classifier(char *datacfg, char *cfgfile, char *weightfile, char *filename, int top);

int main(int argc, char **argv)
{

    init_cpu();

    predict_classifier("cfg/imagenet1k.data", "cfg/darknet19.cfg", "weights/darknet19.weights", "data/dog.jpg", 5);

    return 0;
}
