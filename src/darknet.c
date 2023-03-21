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


	int i;
	for (i = 0; i < argc; ++i) {
		if (!argv[i]) continue;
		strip_args(argv[i]);
	}

    if(argc < 2){
        fprintf(stderr, "usage: %s <function>\n", argv[0]);
        return 0;
    }
    gpu_index = find_int_arg(argc, argv, "-i", 0);


    gpu_index = -1;
    printf(" GPU isn't used \n");
    init_cpu();


    show_opencv_info();

    if (0 == strcmp(argv[1], "classify")){
        predict_classifier("cfg/imagenet1k.data", argv[2], argv[3], argv[4], 5);
    }else {
        fprintf(stderr, "Not an option: %s\n", argv[1]);
    }
    return 0;
}
