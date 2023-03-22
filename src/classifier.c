#include "network.h"
#include "utils.h"
#include "parser.h"
#include "option_list.h"
#include "blas.h"
#include "assert.h"
#include "classifier.h"
#ifdef WIN32
#include <time.h>
#else
#include <sys/time.h>
#endif

void predict_classifier(char *datacfg, char *cfgfile, char *weightfile, char *filename, int top)
{
    network net = parse_network_cfg_custom(cfgfile, 1, 0);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    set_batch_network(&net, 1);
    srand(2222222);

    fuse_conv_batchnorm(net);
    calculate_binary_weights(net);

    list *options = read_data_cfg(datacfg);

    char *name_list = option_find_str(options, "names", 0);
    if(!name_list) name_list = option_find_str(options, "labels", "data/labels.list");
    int classes = option_find_int(options, "classes", 2);
    printf(" classes = %d, output in cfg = %d \n", classes, net.layers[net.n - 1].c);
    layer l = net.layers[net.n - 1];
    if (classes != l.outputs && (l.type == SOFTMAX || l.type == COST)) {
        printf("\n Error: num of filters = %d in the last conv-layer in cfg-file doesn't match to classes = %d in data-file \n",
            l.outputs, classes);
        getchar();
    }
    if (top == 0) top = option_find_int(options, "top", 1);
    if (top > classes) top = classes;

    int i = 0;
    char **names = get_labels(name_list);
    clock_t time;
    int* indexes = (int*)xcalloc(top, sizeof(int));
    char buff[256];
    char *input = buff;
    //int size = net.w;
    while(1){
        strncpy(input, filename, 256);
        
        image im = load_image_color(input, 0, 0);
        image resized = resize_min(im, net.w);
        image cropped = crop_image(resized, (resized.w - net.w)/2, (resized.h - net.h)/2, net.w, net.h);
        printf("%d %d\n", cropped.w, cropped.h);

        float *X = cropped.data;

        double time = get_time_point();
        float *predictions = network_predict(net, X);
        printf("%s: Predicted in %lf milli-seconds.\n", input, ((double)get_time_point() - time) / 1000);

        if(net.hierarchy) hierarchy_predictions(predictions, net.outputs, net.hierarchy, 0);
        top_k(predictions, net.outputs, top, indexes);

        for(i = 0; i < top; ++i){
            int index = indexes[i];
            if(net.hierarchy) printf("%d, %s: %f, parent: %s \n",index, names[index], predictions[index], (net.hierarchy->parent[index] >= 0) ? names[net.hierarchy->parent[index]] : "Root");
            else printf("%s: %f\n",names[index], predictions[index]);
        }

        free_image(cropped);
        if (resized.data != im.data) {
            free_image(resized);
        }
        free_image(im);

        if (filename) break;
    }
    free(indexes);
    free_network(net);
    free_list_contents_kvp(options);
    free_list(options);
}


