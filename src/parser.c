#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>

#include "activations.h"
#include "assert.h"
#include "avgpool_layer.h"
#include "blas.h"
#include "convolutional_layer.h"
#include "cost_layer.h"
#include "list.h"
#include "maxpool_layer.h"
#include "option_list.h"
#include "parser.h"
#include "softmax_layer.h"
#include "utils.h"

typedef struct{
    char *type;
    list *options;
}section;

list *read_cfg(char *filename);

LAYER_TYPE string_to_layer_type(char * type)
{
    if (strcmp(type, "[cost]")==0) return COST;
    if (strcmp(type, "[conv]")==0
            || strcmp(type, "[convolutional]")==0) return CONVOLUTIONAL;
    if (strcmp(type, "[net]")==0
            || strcmp(type, "[network]")==0) return NETWORK;
    if (strcmp(type, "[max]")==0
            || strcmp(type, "[maxpool]")==0) return MAXPOOL;
    if (strcmp(type, "[avg]")==0
            || strcmp(type, "[avgpool]")==0) return AVGPOOL;
    if (strcmp(type, "[soft]")==0
            || strcmp(type, "[softmax]")==0) return SOFTMAX;
    return BLANK;
}

void free_section(section *s)
{
    free(s->type);
    node *n = s->options->front;
    while(n){
        kvp *pair = (kvp *)n->val;
        free(pair->key);
        free(pair);
        node *next = n->next;
        free(n);
        n = next;
    }
    free(s->options);
    free(s);
}

typedef struct size_params{
    int batch;
    int inputs;
    int h;
    int w;
    int c;
    int index;
    int time_steps;
    int train;
    network net;
} size_params;


convolutional_layer parse_convolutional(list *options, size_params params)
{
    int n = option_find_int(options, "filters",1);
    int groups = option_find_int_quiet(options, "groups", 1);
    int size = option_find_int(options, "size",1);
    int stride = -1;
    //int stride = option_find_int(options, "stride",1);
    int stride_x = option_find_int_quiet(options, "stride_x", -1);
    int stride_y = option_find_int_quiet(options, "stride_y", -1);
    if (stride_x < 1 || stride_y < 1) {
        stride = option_find_int(options, "stride", 1);
        if (stride_x < 1) stride_x = stride;
        if (stride_y < 1) stride_y = stride;
    }
    else {
        stride = option_find_int_quiet(options, "stride", 1);
    }
    int dilation = option_find_int_quiet(options, "dilation", 1);
    int antialiasing = option_find_int_quiet(options, "antialiasing", 0);
    if (size == 1) dilation = 1;
    int pad = option_find_int_quiet(options, "pad",0);
    int padding = option_find_int_quiet(options, "padding",0);
    if(pad) padding = size/2;

    char *activation_s = option_find_str(options, "activation", "logistic");
    ACTIVATION activation = get_activation(activation_s);

    int assisted_excitation = option_find_float_quiet(options, "assisted_excitation", 0);

    int share_index = option_find_int_quiet(options, "share_index", -1000000000);
    convolutional_layer *share_layer = NULL;
    if(share_index >= 0) share_layer = &params.net.layers[share_index];
    else if(share_index != -1000000000) share_layer = &params.net.layers[params.index + share_index];

    int batch,h,w,c;
    h = params.h;
    w = params.w;
    c = params.c;
    batch=params.batch;
    if(!(h && w && c)) error("Layer before convolutional layer must output image.", DARKNET_LOC);
    int batch_normalize = option_find_int_quiet(options, "batch_normalize", 0);
    int cbn = option_find_int_quiet(options, "cbn", 0);
    if (cbn) batch_normalize = 2;
    int binary = option_find_int_quiet(options, "binary", 0);
    int xnor = option_find_int_quiet(options, "xnor", 0);
    int use_bin_output = option_find_int_quiet(options, "bin_output", 0);
    int sway = option_find_int_quiet(options, "sway", 0);
    int rotate = option_find_int_quiet(options, "rotate", 0);
    int stretch = option_find_int_quiet(options, "stretch", 0);
    int stretch_sway = option_find_int_quiet(options, "stretch_sway", 0);
    if ((sway + rotate + stretch + stretch_sway) > 1) {
        printf(" Error: should be used only 1 param: sway=1, rotate=1 or stretch=1 in the [convolutional] layer \n");
        exit(0);
    }
    int deform = sway || rotate || stretch || stretch_sway;
    if (deform && size == 1) {
        printf(" Error: params (sway=1, rotate=1 or stretch=1) should be used only with size >=3 in the [convolutional] layer \n");
        exit(0);
    }

    convolutional_layer layer = make_convolutional_layer(batch,1,h,w,c,n,groups,size,stride_x,stride_y,dilation,padding,activation, batch_normalize, binary, xnor, params.net.adam, use_bin_output, params.index, antialiasing, share_layer, assisted_excitation, deform, params.train);
    layer.flipped = option_find_int_quiet(options, "flipped", 0);
    layer.dot = option_find_float_quiet(options, "dot", 0);
    layer.sway = sway;
    layer.rotate = rotate;
    layer.stretch = stretch;
    layer.stretch_sway = stretch_sway;
    layer.angle = option_find_float_quiet(options, "angle", 15);
    layer.grad_centr = option_find_int_quiet(options, "grad_centr", 0);
    layer.reverse = option_find_float_quiet(options, "reverse", 0);
    layer.coordconv = option_find_int_quiet(options, "coordconv", 0);

    layer.stream = option_find_int_quiet(options, "stream", -1);
    layer.wait_stream_id = option_find_int_quiet(options, "wait_stream", -1);

    if(params.net.adam){
        layer.B1 = params.net.B1;
        layer.B2 = params.net.B2;
        layer.eps = params.net.eps;
    }

    return layer;
}

softmax_layer parse_softmax(list *options, size_params params)
{
	int groups = option_find_int_quiet(options, "groups", 1);
	softmax_layer layer = make_softmax_layer(params.batch, params.inputs, groups);
	layer.temperature = option_find_float_quiet(options, "temperature", 1);
	char *tree_file = option_find_str(options, "tree", 0);
	if (tree_file) layer.softmax_tree = read_tree(tree_file);
	layer.w = params.w;
	layer.h = params.h;
	layer.c = params.c;
	layer.spatial = option_find_float_quiet(options, "spatial", 0);
	layer.noloss = option_find_int_quiet(options, "noloss", 0);
	return layer;
}

cost_layer parse_cost(list *options, size_params params)
{
    char *type_s = option_find_str(options, "type", "sse");
    COST_TYPE type = get_cost_type(type_s);
    float scale = option_find_float_quiet(options, "scale",1);
    cost_layer layer = make_cost_layer(params.batch, params.inputs, type, scale);
    layer.ratio =  option_find_float_quiet(options, "ratio",0);
    return layer;
}


maxpool_layer parse_maxpool(list *options, size_params params)
{
    int stride = option_find_int(options, "stride",1);
    int stride_x = option_find_int_quiet(options, "stride_x", stride);
    int stride_y = option_find_int_quiet(options, "stride_y", stride);
    int size = option_find_int(options, "size",stride);
    int padding = option_find_int_quiet(options, "padding", size-1);
    int maxpool_depth = option_find_int_quiet(options, "maxpool_depth", 0);
    int out_channels = option_find_int_quiet(options, "out_channels", 1);
    int antialiasing = option_find_int_quiet(options, "antialiasing", 0);
    const int avgpool = 0;

    int batch,h,w,c;
    h = params.h;
    w = params.w;
    c = params.c;
    batch=params.batch;
    if(!(h && w && c)) error("Layer before [maxpool] layer must output image.", DARKNET_LOC);

    maxpool_layer layer = make_maxpool_layer(batch, h, w, c, size, stride_x, stride_y, padding, maxpool_depth, out_channels, antialiasing, avgpool, params.train);
    layer.maxpool_zero_nonmax = option_find_int_quiet(options, "maxpool_zero_nonmax", 0);
    return layer;
}

avgpool_layer parse_avgpool(list *options, size_params params)
{
    int batch,w,h,c;
    w = params.w;
    h = params.h;
    c = params.c;
    batch=params.batch;
    if(!(h && w && c)) error("Layer before avgpool layer must output image.", DARKNET_LOC);

    avgpool_layer layer = make_avgpool_layer(batch,w,h,c);
    return layer;
}

learning_rate_policy get_policy(char *s)
{
    if (strcmp(s, "random")==0) return RANDOM;
    if (strcmp(s, "poly")==0) return POLY;
    if (strcmp(s, "constant")==0) return CONSTANT;
    if (strcmp(s, "step")==0) return STEP;
    if (strcmp(s, "exp")==0) return EXP;
    if (strcmp(s, "sigmoid")==0) return SIG;
    if (strcmp(s, "steps")==0) return STEPS;
    if (strcmp(s, "sgdr")==0) return SGDR;
    fprintf(stderr, "Couldn't find policy %s, going with constant\n", s);
    return CONSTANT;
}

void parse_net_options(list *options, network *net)
{
    net->max_batches = option_find_int(options, "max_batches", 0);
    net->batch = option_find_int(options, "batch",1);
    net->learning_rate = option_find_float(options, "learning_rate", .001);
    net->learning_rate_min = option_find_float_quiet(options, "learning_rate_min", .00001);
    net->batches_per_cycle = option_find_int_quiet(options, "sgdr_cycle", net->max_batches);
    net->batches_cycle_mult = option_find_int_quiet(options, "sgdr_mult", 2);
    net->momentum = option_find_float(options, "momentum", .9);
    net->decay = option_find_float(options, "decay", .0001);
    int subdivs = option_find_int(options, "subdivisions",1);
    net->time_steps = option_find_int_quiet(options, "time_steps",1);
    net->track = option_find_int_quiet(options, "track", 0);
    net->augment_speed = option_find_int_quiet(options, "augment_speed", 2);
    net->init_sequential_subdivisions = net->sequential_subdivisions = option_find_int_quiet(options, "sequential_subdivisions", subdivs);
    if (net->sequential_subdivisions > subdivs) net->init_sequential_subdivisions = net->sequential_subdivisions = subdivs;
    net->try_fix_nan = option_find_int_quiet(options, "try_fix_nan", 0);
    net->batch /= subdivs;          // mini_batch
    const int mini_batch = net->batch;
    net->batch *= net->time_steps;  // mini_batch * time_steps
    net->subdivisions = subdivs;    // number of mini_batches

    net->weights_reject_freq = option_find_int_quiet(options, "weights_reject_freq", 0);
    net->equidistant_point = option_find_int_quiet(options, "equidistant_point", 0);
    net->badlabels_rejection_percentage = option_find_float_quiet(options, "badlabels_rejection_percentage", 0);
    net->num_sigmas_reject_badlabels = option_find_float_quiet(options, "num_sigmas_reject_badlabels", 0);
    net->ema_alpha = option_find_float_quiet(options, "ema_alpha", 0);
    *net->badlabels_reject_threshold = 0;
    *net->delta_rolling_max = 0;
    *net->delta_rolling_avg = 0;
    *net->delta_rolling_std = 0;
    *net->seen = 0;
    *net->cur_iteration = 0;
    *net->cuda_graph_ready = 0;
    net->use_cuda_graph = option_find_int_quiet(options, "use_cuda_graph", 0);
    net->loss_scale = option_find_float_quiet(options, "loss_scale", 1);
    net->dynamic_minibatch = option_find_int_quiet(options, "dynamic_minibatch", 0);
    net->optimized_memory = option_find_int_quiet(options, "optimized_memory", 0);
    net->workspace_size_limit = (size_t)1024*1024 * option_find_float_quiet(options, "workspace_size_limit_MB", 1024);  // 1024 MB by default


    net->adam = option_find_int_quiet(options, "adam", 0);
    if(net->adam){
        net->B1 = option_find_float(options, "B1", .9);
        net->B2 = option_find_float(options, "B2", .999);
        net->eps = option_find_float(options, "eps", .000001);
    }

    net->h = option_find_int_quiet(options, "height",0);
    net->w = option_find_int_quiet(options, "width",0);
    net->c = option_find_int_quiet(options, "channels",0);
    net->inputs = option_find_int_quiet(options, "inputs", net->h * net->w * net->c);
    net->max_crop = option_find_int_quiet(options, "max_crop",net->w*2);
    net->min_crop = option_find_int_quiet(options, "min_crop",net->w);
    net->flip = option_find_int_quiet(options, "flip", 1);
    net->blur = option_find_int_quiet(options, "blur", 0);
    net->gaussian_noise = option_find_int_quiet(options, "gaussian_noise", 0);
    net->mixup = option_find_int_quiet(options, "mixup", 0);
    int cutmix = option_find_int_quiet(options, "cutmix", 0);
    int mosaic = option_find_int_quiet(options, "mosaic", 0);
    if (mosaic && cutmix) net->mixup = 4;
    else if (cutmix) net->mixup = 2;
    else if (mosaic) net->mixup = 3;
    net->letter_box = option_find_int_quiet(options, "letter_box", 0);
    net->mosaic_bound = option_find_int_quiet(options, "mosaic_bound", 0);
    net->contrastive = option_find_int_quiet(options, "contrastive", 0);
    net->contrastive_jit_flip = option_find_int_quiet(options, "contrastive_jit_flip", 0);
    net->contrastive_color = option_find_int_quiet(options, "contrastive_color", 0);
    net->unsupervised = option_find_int_quiet(options, "unsupervised", 0);
    if (net->contrastive && mini_batch < 2) {
        printf(" Error: mini_batch size (batch/subdivisions) should be higher than 1 for Contrastive loss \n");
        exit(0);
    }
    net->label_smooth_eps = option_find_float_quiet(options, "label_smooth_eps", 0.0f);
    net->resize_step = option_find_float_quiet(options, "resize_step", 32);
    net->attention = option_find_int_quiet(options, "attention", 0);
    net->adversarial_lr = option_find_float_quiet(options, "adversarial_lr", 0);
    net->max_chart_loss = option_find_float_quiet(options, "max_chart_loss", 20.0);

    net->angle = option_find_float_quiet(options, "angle", 0);
    net->aspect = option_find_float_quiet(options, "aspect", 1);
    net->saturation = option_find_float_quiet(options, "saturation", 1);
    net->exposure = option_find_float_quiet(options, "exposure", 1);
    net->hue = option_find_float_quiet(options, "hue", 0);
    net->power = option_find_float_quiet(options, "power", 4);

    if(!net->inputs && !(net->h && net->w && net->c)) error("No input parameters supplied", DARKNET_LOC);

    char *policy_s = option_find_str(options, "policy", "constant");
    net->policy = get_policy(policy_s);
    net->burn_in = option_find_int_quiet(options, "burn_in", 0);

    if(net->policy == STEP){
        net->step = option_find_int(options, "step", 1);
        net->scale = option_find_float(options, "scale", 1);
    } else if (net->policy == STEPS || net->policy == SGDR){
        char *l = option_find(options, "steps");
        char *p = option_find(options, "scales");
        char *s = option_find(options, "seq_scales");
        if(net->policy == STEPS && (!l || !p)) error("STEPS policy must have steps and scales in cfg file", DARKNET_LOC);

        if (l) {
            int len = strlen(l);
            int n = 1;
            int i;
            for (i = 0; i < len; ++i) {
                if (l[i] == '#') break;
                if (l[i] == ',') ++n;
            }
            int* steps = (int*)xcalloc(n, sizeof(int));
            float* scales = (float*)xcalloc(n, sizeof(float));
            float* seq_scales = (float*)xcalloc(n, sizeof(float));
            for (i = 0; i < n; ++i) {
                float scale = 1.0;
                if (p) {
                    scale = atof(p);
                    p = strchr(p, ',') + 1;
                }
                float sequence_scale = 1.0;
                if (s) {
                    sequence_scale = atof(s);
                    s = strchr(s, ',') + 1;
                }
                int step = atoi(l);
                l = strchr(l, ',') + 1;
                steps[i] = step;
                scales[i] = scale;
                seq_scales[i] = sequence_scale;
            }
            net->scales = scales;
            net->steps = steps;
            net->seq_scales = seq_scales;
            net->num_steps = n;
        }
    } else if (net->policy == EXP){
        net->gamma = option_find_float(options, "gamma", 1);
    } else if (net->policy == SIG){
        net->gamma = option_find_float(options, "gamma", 1);
        net->step = option_find_int(options, "step", 1);
    } else if (net->policy == POLY || net->policy == RANDOM){
        //net->power = option_find_float(options, "power", 1);
    }

}

int is_network(section *s)
{
    return (strcmp(s->type, "[net]")==0
            || strcmp(s->type, "[network]")==0);
}


network parse_network_cfg(char *filename)
{
    return parse_network_cfg_custom(filename, 0, 0);
}

network parse_network_cfg_custom(char *filename, int batch, int time_steps)
{
    list *sections = read_cfg(filename);
    node *n = sections->front;
    if(!n) error("Config file has no sections", DARKNET_LOC);
    network net = make_network(sections->size - 1);
    net.gpu_index = -1;
    size_params params;

    if (batch > 0) params.train = 0;    // allocates memory for Inference only
    else params.train = 1;              // allocates memory for Inference & Training

    section *s = (section *)n->val;
    list *options = s->options;
    if(!is_network(s)) error("First section must be [net] or [network]", DARKNET_LOC);
    parse_net_options(options, &net);

    params.h = net.h;
    params.w = net.w;
    params.c = net.c;
    params.inputs = net.inputs;
    if (batch > 0) net.batch = batch;
    if (time_steps > 0) net.time_steps = time_steps;
    if (net.batch < 1) net.batch = 1;
    if (net.time_steps < 1) net.time_steps = 1;
    if (net.batch < net.time_steps) net.batch = net.time_steps;
    params.batch = net.batch;
    params.time_steps = net.time_steps;
    params.net = net;
    printf("mini_batch = %d, batch = %d, time_steps = %d, train = %d \n", net.batch, net.batch * net.subdivisions, net.time_steps, params.train);

    int last_stop_backward = -1;
    int avg_outputs = 0;
    int avg_counter = 0;
    float bflops = 0;
    size_t workspace_size = 0;
    size_t max_inputs = 0;
    size_t max_outputs = 0;
    int receptive_w = 1, receptive_h = 1;
    int receptive_w_scale = 1, receptive_h_scale = 1;
    const int show_receptive_field = option_find_float_quiet(options, "show_receptive_field", 0);

    n = n->next;
    int count = 0;
    free_section(s);

    // find l.stopbackward = option_find_int_quiet(options, "stopbackward", 0);
    node *n_tmp = n;
    int count_tmp = 0;
    if (params.train == 1) {
        while (n_tmp) {
            s = (section *)n_tmp->val;
            options = s->options;
            int stopbackward = option_find_int_quiet(options, "stopbackward", 0);
            if (stopbackward == 1) {
                last_stop_backward = count_tmp;
                printf("last_stop_backward = %d \n", last_stop_backward);
            }
            n_tmp = n_tmp->next;
            ++count_tmp;
        }
    }

    int old_params_train = params.train;

    fprintf(stderr, "   layer   filters  size/strd(dil)      input                output\n");
    while(n){

        params.train = old_params_train;
        if (count < last_stop_backward) params.train = 0;

        params.index = count;
        fprintf(stderr, "%4d ", count);
        s = (section *)n->val;
        options = s->options;
        layer l = { (LAYER_TYPE)0 };
        LAYER_TYPE lt = string_to_layer_type(s->type);
        if(lt == CONVOLUTIONAL){
            l = parse_convolutional(options, params);
        }else if(lt == COST){
            l = parse_cost(options, params);
            l.keep_delta_gpu = 1;
        }else if(lt == SOFTMAX){
            l = parse_softmax(options, params);
            net.hierarchy = l.softmax_tree;
            l.keep_delta_gpu = 1;
        }else if(lt == MAXPOOL){
            l = parse_maxpool(options, params);
        }else if(lt == AVGPOOL){
            l = parse_avgpool(options, params);
        }else{
            fprintf(stderr, "Type not recognized: %s\n", s->type);
        }

        // calculate receptive field
        if(show_receptive_field)
        {
            int dilation = max_val_cmp(1, l.dilation);
            int stride = max_val_cmp(1, l.stride);
            int size = max_val_cmp(1, l.size);

            int increase_receptive = size + (dilation - 1) * 2 - 1;// stride;
            increase_receptive = max_val_cmp(0, increase_receptive);

            receptive_w += increase_receptive * receptive_w_scale;
            receptive_h += increase_receptive * receptive_h_scale;
            receptive_w_scale *= stride;
            receptive_h_scale *= stride;

            l.receptive_w = receptive_w;
            l.receptive_h = receptive_h;
            l.receptive_w_scale = receptive_w_scale;
            l.receptive_h_scale = receptive_h_scale;
        
            //printf(" size = %d, dilation = %d, stride = %d, receptive_w = %d, receptive_w_scale = %d - ", size, dilation, stride, receptive_w, receptive_w_scale);

            int cur_receptive_w = receptive_w;
            int cur_receptive_h = receptive_h;

            fprintf(stderr, "%4d - receptive field: %d x %d \n", count, cur_receptive_w, cur_receptive_h);
        }


        l.clip = option_find_float_quiet(options, "clip", 0);
        l.dynamic_minibatch = net.dynamic_minibatch;
        l.onlyforward = option_find_int_quiet(options, "onlyforward", 0);
        l.dont_update = option_find_int_quiet(options, "dont_update", 0);
        l.burnin_update = option_find_int_quiet(options, "burnin_update", 0);
        l.stopbackward = option_find_int_quiet(options, "stopbackward", 0);
        l.train_only_bn = option_find_int_quiet(options, "train_only_bn", 0);
        l.dontload = option_find_int_quiet(options, "dontload", 0);
        l.dontloadscales = option_find_int_quiet(options, "dontloadscales", 0);
        l.learning_rate_scale = option_find_float_quiet(options, "learning_rate", 1);
        option_unused(options);

        if (l.stopbackward == 1) printf(" ------- previous layers are frozen ------- \n");

        net.layers[count] = l;
        if (l.workspace_size > workspace_size) workspace_size = l.workspace_size;
        if (l.inputs > max_inputs) max_inputs = l.inputs;
        if (l.outputs > max_outputs) max_outputs = l.outputs;
        free_section(s);
        n = n->next;
        ++count;
        if(n){
            if (l.antialiasing) {
                params.h = l.input_layer->out_h;
                params.w = l.input_layer->out_w;
                params.c = l.input_layer->out_c;
                params.inputs = l.input_layer->outputs;
            }
            else {
                params.h = l.out_h;
                params.w = l.out_w;
                params.c = l.out_c;
                params.inputs = l.outputs;
            }
        }
        if (l.bflops > 0) bflops += l.bflops;

        if (l.w > 1 && l.h > 1) {
            avg_outputs += l.outputs;
            avg_counter++;
        }
    }

    if (last_stop_backward > -1) {
        int k;
        for (k = 0; k < last_stop_backward; ++k) {
            layer l = net.layers[k];
            if (l.keep_delta_gpu) {
                if (!l.delta) {
                    net.layers[k].delta = (float*)xcalloc(l.outputs*l.batch, sizeof(float));
                }
            }

            net.layers[k].onlyforward = 1;
            net.layers[k].train = 0;
        }
    }

    free_list(sections);

    net.outputs = get_network_output_size(net);
    net.output = get_network_output(net);
    avg_outputs = avg_outputs / avg_counter;
    fprintf(stderr, "Total BFLOPS %5.3f \n", bflops);
    fprintf(stderr, "avg_outputs = %d \n", avg_outputs);
    if (workspace_size) {
        net.workspace = (float*)xcalloc(1, workspace_size);
    }


    LAYER_TYPE lt = net.layers[net.n - 1].type;
    if ((net.w % 32 != 0 || net.h % 32 != 0) && (lt == YOLO || lt == REGION || lt == DETECTION)) {
        printf("\n Warning: width=%d and height=%d in cfg-file must be divisible by 32 for default networks Yolo v1/v2/v3!!! \n\n",
            net.w, net.h);
    }
    return net;
}



list *read_cfg(char *filename)
{
    FILE *file = fopen(filename, "r");
    if(file == 0) file_error(filename);
    char *line;
    int nu = 0;
    list *sections = make_list();
    section *current = 0;
    while((line=fgetl(file)) != 0){
        ++ nu;
        strip(line);
        switch(line[0]){
            case '[':
                current = (section*)xmalloc(sizeof(section));
                list_insert(sections, current);
                current->options = make_list();
                current->type = line;
                break;
            case '\0':
            case '#':
            case ';':
                free(line);
                break;
            default:
                if(!read_option(line, current->options)){
                    fprintf(stderr, "Config file error line %d, could parse: %s\n", nu, line);
                    free(line);
                }
                break;
        }
    }
    fclose(file);
    return sections;
}


void transpose_matrix(float *a, int rows, int cols)
{
    float* transpose = (float*)xcalloc(rows * cols, sizeof(float));
    int x, y;
    for(x = 0; x < rows; ++x){
        for(y = 0; y < cols; ++y){
            transpose[y*rows + x] = a[x*cols + y];
        }
    }
    memcpy(a, transpose, rows*cols*sizeof(float));
    free(transpose);
}

void load_convolutional_weights(layer l, FILE *fp)
{
    if(l.binary){
        //load_convolutional_weights_binary(l, fp);
        //return;
    }
    int num = l.nweights;
    int read_bytes;
    read_bytes = fread(l.biases, sizeof(float), l.n, fp);
    if (read_bytes > 0 && read_bytes < l.n) printf("\n Warning: Unexpected end of wights-file! l.biases - l.index = %d \n", l.index);
    //fread(l.weights, sizeof(float), num, fp); // as in connected layer
    if (l.batch_normalize && (!l.dontloadscales)){
        read_bytes = fread(l.scales, sizeof(float), l.n, fp);
        if (read_bytes > 0 && read_bytes < l.n) printf("\n Warning: Unexpected end of wights-file! l.scales - l.index = %d \n", l.index);
        read_bytes = fread(l.rolling_mean, sizeof(float), l.n, fp);
        if (read_bytes > 0 && read_bytes < l.n) printf("\n Warning: Unexpected end of wights-file! l.rolling_mean - l.index = %d \n", l.index);
        read_bytes = fread(l.rolling_variance, sizeof(float), l.n, fp);
        if (read_bytes > 0 && read_bytes < l.n) printf("\n Warning: Unexpected end of wights-file! l.rolling_variance - l.index = %d \n", l.index);
        if(0){
            int i;
            for(i = 0; i < l.n; ++i){
                printf("%g, ", l.rolling_mean[i]);
            }
            printf("\n");
            for(i = 0; i < l.n; ++i){
                printf("%g, ", l.rolling_variance[i]);
            }
            printf("\n");
        }
        if(0){
            fill_cpu(l.n, 0, l.rolling_mean, 1);
            fill_cpu(l.n, 0, l.rolling_variance, 1);
        }
    }
    read_bytes = fread(l.weights, sizeof(float), num, fp);
    if (read_bytes > 0 && read_bytes < l.n) printf("\n Warning: Unexpected end of wights-file! l.weights - l.index = %d \n", l.index);
    //if(l.adam){
    //    fread(l.m, sizeof(float), num, fp);
    //    fread(l.v, sizeof(float), num, fp);
    //}
    //if(l.c == 3) scal_cpu(num, 1./256, l.weights, 1);
    if (l.flipped) {
        transpose_matrix(l.weights, (l.c/l.groups)*l.size*l.size, l.n);
    }
    //if (l.binary) binarize_weights(l.weights, l.n, (l.c/l.groups)*l.size*l.size, l.weights);
#ifdef GPU
    if(gpu_index >= 0){
        push_convolutional_layer(l);
    }
#endif
}



void load_weights_upto(network *net, char *filename, int cutoff)
{

    fprintf(stderr, "Loading weights from %s...", filename);
    fflush(stdout);
    FILE *fp = fopen(filename, "rb");
    if(!fp) file_error(filename);

    int major;
    int minor;
    int revision;
    fread(&major, sizeof(int), 1, fp);
    fread(&minor, sizeof(int), 1, fp);
    fread(&revision, sizeof(int), 1, fp);
    if ((major * 10 + minor) >= 2) {
        printf("\n seen 64");
        uint64_t iseen = 0;
        fread(&iseen, sizeof(uint64_t), 1, fp);
        *net->seen = iseen;
    }
    else {
        printf("\n seen 32");
        uint32_t iseen = 0;
        fread(&iseen, sizeof(uint32_t), 1, fp);
        *net->seen = iseen;
    }
    *net->cur_iteration = get_current_batch(*net);
    printf(", trained: %.0f K-images (%.0f Kilo-batches_64) \n", (float)(*net->seen / 1000), (float)(*net->seen / 64000));
    int transpose = (major > 1000) || (minor > 1000);

    int i;
    for(i = 0; i < net->n && i < cutoff; ++i){
        layer l = net->layers[i];
        if (l.dontload) continue;
        if(l.type == CONVOLUTIONAL && l.share_layer == NULL){
            load_convolutional_weights(l, fp);
        }

        if (feof(fp)) break;
    }
    fprintf(stderr, "Done! Loaded %d layers from weights-file \n", i);
    fclose(fp);
}

void load_weights(network *net, char *filename)
{
    load_weights_upto(net, filename, net->n);
}

