#include "darknet.h"

#include <stdio.h>
#include <time.h>
#include <assert.h>

#include "network.h"
#include "image.h"
#include "data.h"
#include "utils.h"
#include "blas.h"

#include "convolutional_layer.h"
#include "activation_layer.h"
#include "normalization_layer.h"
#include "batchnorm_layer.h"
#include "maxpool_layer.h"
#include "avgpool_layer.h"
#include "cost_layer.h"
#include "softmax_layer.h"
#include "parser.h"

network make_network(int n)
{
    network net = {0};
    net.n = n;
    net.layers = (layer*)xcalloc(net.n, sizeof(layer));
    net.seen = (uint64_t*)xcalloc(1, sizeof(uint64_t));
    net.cuda_graph_ready = (int*)xcalloc(1, sizeof(int));
    net.badlabels_reject_threshold = (float*)xcalloc(1, sizeof(float));
    net.delta_rolling_max = (float*)xcalloc(1, sizeof(float));
    net.delta_rolling_avg = (float*)xcalloc(1, sizeof(float));
    net.delta_rolling_std = (float*)xcalloc(1, sizeof(float));
    net.cur_iteration = (int*)xcalloc(1, sizeof(int));
    net.total_bbox = (int*)xcalloc(1, sizeof(int));
    net.rewritten_bbox = (int*)xcalloc(1, sizeof(int));
    *net.rewritten_bbox = *net.total_bbox = 0;
#ifdef GPU
    net.input_gpu = (float**)xcalloc(1, sizeof(float*));
    net.truth_gpu = (float**)xcalloc(1, sizeof(float*));

    net.input16_gpu = (float**)xcalloc(1, sizeof(float*));
    net.output16_gpu = (float**)xcalloc(1, sizeof(float*));
    net.max_input16_size = (size_t*)xcalloc(1, sizeof(size_t));
    net.max_output16_size = (size_t*)xcalloc(1, sizeof(size_t));
#endif
    return net;
}

void forward_network(network net, network_state state)
{
    state.workspace = net.workspace;
    int i;
    for(i = 0; i < net.n; ++i){
        state.index = i;
        layer l = net.layers[i];
        if(l.delta && state.train && l.train){
            scal_cpu(l.outputs * l.batch, 0, l.delta, 1);
        }
        //double time = get_time_point();
        l.forward(l, state);
        //printf("%d - Predicted in %lf milli-seconds.\n", i, ((double)get_time_point() - time) / 1000);
        state.input = l.output;

        /*
        float avg_val = 0;
        int k;
        for (k = 0; k < l.outputs; ++k) avg_val += l.output[k];
        printf(" i: %d - avg_val = %f \n", i, avg_val / l.outputs);
        */
    }
}

float *get_network_output(network net)
{
#ifdef GPU
    if (gpu_index >= 0) return get_network_output_gpu(net);
#endif
    int i;
    for(i = net.n-1; i > 0; --i) if(net.layers[i].type != COST) break;
    return net.layers[i].output;
}



int recalculate_workspace_size(network *net)
{
#ifdef GPU
    cuda_set_device(net->gpu_index);
    if (gpu_index >= 0) cuda_free(net->workspace);
#endif
    int i;
    size_t workspace_size = 0;
    for (i = 0; i < net->n; ++i) {
        layer l = net->layers[i];
        //printf(" %d: layer = %d,", i, l.type);
        if (l.type == CONVOLUTIONAL) {
            l.workspace_size = get_convolutional_workspace_size(l);
        }
        else if (l.type == CONNECTED) {
            l.workspace_size = get_connected_workspace_size(l);
        }
        if (l.workspace_size > workspace_size) workspace_size = l.workspace_size;
        net->layers[i] = l;
    }

#ifdef GPU
    if (gpu_index >= 0) {
        printf("\n try to allocate additional workspace_size = %1.2f MB \n", (float)workspace_size / 1000000);
        net->workspace = cuda_make_array(0, workspace_size / sizeof(float) + 1);
        printf(" CUDA allocate done! \n");
    }
    else {
        free(net->workspace);
        net->workspace = (float*)xcalloc(1, workspace_size);
    }
#else
    free(net->workspace);
    net->workspace = (float*)xcalloc(1, workspace_size);
#endif
    //fprintf(stderr, " Done!\n");
    return 0;
}

void set_batch_network(network *net, int b)
{
    net->batch = b;
    int i;
    for(i = 0; i < net->n; ++i){
        net->layers[i].batch = b;

#ifdef CUDNN
        if(net->layers[i].type == CONVOLUTIONAL){
            cudnn_convolutional_setup(net->layers + i, cudnn_fastest, 0);
        }
        else if (net->layers[i].type == MAXPOOL) {
            cudnn_maxpool_setup(net->layers + i);
        }
#endif

    }
    recalculate_workspace_size(net); // recalculate workspace size
}


int get_network_output_size(network net)
{
    int i;
    for(i = net.n-1; i > 0; --i) if(net.layers[i].type != COST) break;
    return net.layers[i].outputs;
}



float *network_predict(network net, float *input)
{
#ifdef GPU
    if(gpu_index >= 0)  return network_predict_gpu(net, input);
#endif

    network_state state = {0};
    state.net = net;
    state.index = 0;
    state.input = input;
    state.truth = 0;
    state.train = 0;
    state.delta = 0;
    forward_network(net, state);
    float *out = get_network_output(net);
    return out;
}



void free_network(network net)
{
    int i;
    for (i = 0; i < net.n; ++i) {
        free_layer(net.layers[i]);
    }
    free(net.layers);

    free(net.seq_scales);
    free(net.scales);
    free(net.steps);
    free(net.seen);
    free(net.cuda_graph_ready);
    free(net.badlabels_reject_threshold);
    free(net.delta_rolling_max);
    free(net.delta_rolling_avg);
    free(net.delta_rolling_std);
    free(net.cur_iteration);
    free(net.total_bbox);
    free(net.rewritten_bbox);

#ifdef GPU
    if (gpu_index >= 0) cuda_free(net.workspace);
    else free(net.workspace);
    free_pinned_memory();
    if (net.input_state_gpu) cuda_free(net.input_state_gpu);
    if (net.input_pinned_cpu) {   // CPU
        if (net.input_pinned_cpu_flag) cudaFreeHost(net.input_pinned_cpu);
        else free(net.input_pinned_cpu);
    }
    if (*net.input_gpu) cuda_free(*net.input_gpu);
    if (*net.truth_gpu) cuda_free(*net.truth_gpu);
    if (net.input_gpu) free(net.input_gpu);
    if (net.truth_gpu) free(net.truth_gpu);

    if (*net.input16_gpu) cuda_free(*net.input16_gpu);
    if (*net.output16_gpu) cuda_free(*net.output16_gpu);
    if (net.input16_gpu) free(net.input16_gpu);
    if (net.output16_gpu) free(net.output16_gpu);
    if (net.max_input16_size) free(net.max_input16_size);
    if (net.max_output16_size) free(net.max_output16_size);
#else
    free(net.workspace);
#endif
}

static float relu(float src) {
    if (src > 0) return src;
    return 0;
}

static float lrelu(float src) {
    const float eps = 0.001;
    if (src > eps) return src;
    return eps;
}

void fuse_conv_batchnorm(network net)
{
    int j;
    for (j = 0; j < net.n; ++j) {
        layer *l = &net.layers[j];

        if (l->type == CONVOLUTIONAL) {
            //printf(" Merges Convolutional-%d and batch_norm \n", j);

            if (l->share_layer != NULL) {
                l->batch_normalize = 0;
            }

            if (l->batch_normalize) {
                int f;
                for (f = 0; f < l->n; ++f)
                {
                    l->biases[f] = l->biases[f] - (double)l->scales[f] * l->rolling_mean[f] / (sqrt((double)l->rolling_variance[f] + .00001));

                    double precomputed = l->scales[f] / (sqrt((double)l->rolling_variance[f] + .00001));

                    const size_t filter_size = l->size*l->size*l->c / l->groups;
                    int i;
                    for (i = 0; i < filter_size; ++i) {
                        int w_index = f*filter_size + i;

                        l->weights[w_index] *= precomputed;
                    }
                }

                free_convolutional_batchnorm(l);
                l->batch_normalize = 0;
#ifdef GPU
                if (gpu_index >= 0) {
                    push_convolutional_layer(*l);
                }
#endif
            }
        }
        else  if (l->type == SHORTCUT && l->weights && l->weights_normalization)
        {
            if (l->nweights > 0) {
                //cuda_pull_array(l.weights_gpu, l.weights, l.nweights);
                int i;
                for (i = 0; i < l->nweights; ++i) printf(" w = %f,", l->weights[i]);
                printf(" l->nweights = %d, j = %d \n", l->nweights, j);
            }

            // nweights - l.n or l.n*l.c or (l.n*l.c*l.h*l.w)
            const int layer_step = l->nweights / (l->n + 1);    // 1 or l.c or (l.c * l.h * l.w)

            int chan, i;
            for (chan = 0; chan < layer_step; ++chan)
            {
                float sum = 1, max_val = -FLT_MAX;

                if (l->weights_normalization == SOFTMAX_NORMALIZATION) {
                    for (i = 0; i < (l->n + 1); ++i) {
                        int w_index = chan + i * layer_step;
                        float w = l->weights[w_index];
                        if (max_val < w) max_val = w;
                    }
                }

                const float eps = 0.0001;
                sum = eps;

                for (i = 0; i < (l->n + 1); ++i) {
                    int w_index = chan + i * layer_step;
                    float w = l->weights[w_index];
                    if (l->weights_normalization == RELU_NORMALIZATION) sum += lrelu(w);
                    else if (l->weights_normalization == SOFTMAX_NORMALIZATION) sum += expf(w - max_val);
                }

                for (i = 0; i < (l->n + 1); ++i) {
                    int w_index = chan + i * layer_step;
                    float w = l->weights[w_index];
                    if (l->weights_normalization == RELU_NORMALIZATION) w = lrelu(w) / sum;
                    else if (l->weights_normalization == SOFTMAX_NORMALIZATION) w = expf(w - max_val) / sum;
                    l->weights[w_index] = w;
                }
            }

            l->weights_normalization = NO_NORMALIZATION;

#ifdef GPU
            if (gpu_index >= 0) {
                push_shortcut_layer(*l);
            }
#endif
        }
        else {
            //printf(" Fusion skip layer type: %d \n", l->type);
        }
    }
}

void forward_blank_layer(layer l, network_state state) {}

void calculate_binary_weights(network net)
{
    int j;
    for (j = 0; j < net.n; ++j) {
        layer *l = &net.layers[j];

        if (l->type == CONVOLUTIONAL) {
            //printf(" Merges Convolutional-%d and batch_norm \n", j);

            if (l->xnor) {
                //printf("\n %d \n", j);
                //l->lda_align = 256; // 256bit for AVX2    // set in make_convolutional_layer()
                //if (l->size*l->size*l->c >= 2048) l->lda_align = 512;

                binary_align_weights(l);

                if (net.layers[j].use_bin_output) {
                    l->activation = LINEAR;
                }

#ifdef GPU
                // fuse conv_xnor + shortcut -> conv_xnor
                if ((j + 1) < net.n && net.layers[j].type == CONVOLUTIONAL) {
                    layer *sc = &net.layers[j + 1];
                    if (sc->type == SHORTCUT && sc->w == sc->out_w && sc->h == sc->out_h && sc->c == sc->out_c)
                    {
                        l->bin_conv_shortcut_in_gpu = net.layers[net.layers[j + 1].index].output_gpu;
                        l->bin_conv_shortcut_out_gpu = net.layers[j + 1].output_gpu;

                        net.layers[j + 1].type = BLANK;
                        net.layers[j + 1].forward_gpu = forward_blank_layer;
                    }
                }
#endif  // GPU
            }
        }
    }
    //printf("\n calculate_binary_weights Done! \n");

}

void copy_cudnn_descriptors(layer src, layer *dst)
{
#ifdef CUDNN
    dst->normTensorDesc = src.normTensorDesc;
    dst->normDstTensorDesc = src.normDstTensorDesc;
    dst->normDstTensorDescF16 = src.normDstTensorDescF16;

    dst->srcTensorDesc = src.srcTensorDesc;
    dst->dstTensorDesc = src.dstTensorDesc;

    dst->srcTensorDesc16 = src.srcTensorDesc16;
    dst->dstTensorDesc16 = src.dstTensorDesc16;
#endif // CUDNN
}

void copy_weights_net(network net_train, network *net_map)
{
    int k;
    for (k = 0; k < net_train.n; ++k) {
        layer *l = &(net_train.layers[k]);
        layer tmp_layer;
        copy_cudnn_descriptors(net_map->layers[k], &tmp_layer);
        net_map->layers[k] = net_train.layers[k];
        copy_cudnn_descriptors(tmp_layer, &net_map->layers[k]);

        if (l->type == CRNN) {
            layer tmp_input_layer, tmp_self_layer, tmp_output_layer;
            copy_cudnn_descriptors(*net_map->layers[k].input_layer, &tmp_input_layer);
            copy_cudnn_descriptors(*net_map->layers[k].self_layer, &tmp_self_layer);
            copy_cudnn_descriptors(*net_map->layers[k].output_layer, &tmp_output_layer);
            net_map->layers[k].input_layer = net_train.layers[k].input_layer;
            net_map->layers[k].self_layer = net_train.layers[k].self_layer;
            net_map->layers[k].output_layer = net_train.layers[k].output_layer;
            //net_map->layers[k].output_gpu = net_map->layers[k].output_layer->output_gpu;  // already copied out of if()

            copy_cudnn_descriptors(tmp_input_layer, net_map->layers[k].input_layer);
            copy_cudnn_descriptors(tmp_self_layer, net_map->layers[k].self_layer);
            copy_cudnn_descriptors(tmp_output_layer, net_map->layers[k].output_layer);
        }
        else if(l->input_layer) // for AntiAliasing
        {
            layer tmp_input_layer;
            copy_cudnn_descriptors(*net_map->layers[k].input_layer, &tmp_input_layer);
            net_map->layers[k].input_layer = net_train.layers[k].input_layer;
            copy_cudnn_descriptors(tmp_input_layer, net_map->layers[k].input_layer);
        }
        net_map->layers[k].batch = 1;
        net_map->layers[k].steps = 1;
        net_map->layers[k].train = 0;
    }
}


// combine Training and Validation networks
network combine_train_valid_networks(network net_train, network net_map)
{
    network net_combined = make_network(net_train.n);
    layer *old_layers = net_combined.layers;
    net_combined = net_train;
    net_combined.layers = old_layers;
    net_combined.batch = 1;

    int k;
    for (k = 0; k < net_train.n; ++k) {
        layer *l = &(net_train.layers[k]);
        net_combined.layers[k] = net_train.layers[k];
        net_combined.layers[k].batch = 1;

        if (l->type == CONVOLUTIONAL) {
#ifdef CUDNN
            net_combined.layers[k].normTensorDesc = net_map.layers[k].normTensorDesc;
            net_combined.layers[k].normDstTensorDesc = net_map.layers[k].normDstTensorDesc;
            net_combined.layers[k].normDstTensorDescF16 = net_map.layers[k].normDstTensorDescF16;

            net_combined.layers[k].srcTensorDesc = net_map.layers[k].srcTensorDesc;
            net_combined.layers[k].dstTensorDesc = net_map.layers[k].dstTensorDesc;

            net_combined.layers[k].srcTensorDesc16 = net_map.layers[k].srcTensorDesc16;
            net_combined.layers[k].dstTensorDesc16 = net_map.layers[k].dstTensorDesc16;
#endif // CUDNN
        }
    }
    return net_combined;
}

void free_network_recurrent_state(network net)
{
    int k;
    for (k = 0; k < net.n; ++k) {
        if (net.layers[k].type == CONV_LSTM) free_state_conv_lstm(net.layers[k]);
        if (net.layers[k].type == CRNN) free_state_crnn(net.layers[k]);
    }
}

void randomize_network_recurrent_state(network net)
{
    int k;
    for (k = 0; k < net.n; ++k) {
        if (net.layers[k].type == CONV_LSTM) randomize_state_conv_lstm(net.layers[k]);
        if (net.layers[k].type == CRNN) free_state_crnn(net.layers[k]);
    }
}


void remember_network_recurrent_state(network net)
{
    int k;
    for (k = 0; k < net.n; ++k) {
        if (net.layers[k].type == CONV_LSTM) remember_state_conv_lstm(net.layers[k]);
        //if (net.layers[k].type == CRNN) free_state_crnn(net.layers[k]);
    }
}

void restore_network_recurrent_state(network net)
{
    int k;
    for (k = 0; k < net.n; ++k) {
        if (net.layers[k].type == CONV_LSTM) restore_state_conv_lstm(net.layers[k]);
        if (net.layers[k].type == CRNN) free_state_crnn(net.layers[k]);
    }
}


int is_ema_initialized(network net)
{
    int i;
    for (i = 0; i < net.n; ++i) {
        layer l = net.layers[i];
        if (l.type == CONVOLUTIONAL) {
            int k;
            if (l.weights_ema) {
                for (k = 0; k < l.nweights; ++k) {
                    if (l.weights_ema[k] != 0) return 1;
                }
            }
        }
    }

    return 0;
}

void ema_update(network net, float ema_alpha)
{
    int i;
    for (i = 0; i < net.n; ++i) {
        layer l = net.layers[i];
        if (l.type == CONVOLUTIONAL) {
#ifdef GPU
            if (gpu_index >= 0) {
                pull_convolutional_layer(l);
            }
#endif
            int k;
            if (l.weights_ema) {
                for (k = 0; k < l.nweights; ++k) {
                    l.weights_ema[k] = ema_alpha * l.weights_ema[k] + (1 - ema_alpha) * l.weights[k];
                }
            }

            for (k = 0; k < l.n; ++k) {
                if (l.biases_ema) l.biases_ema[k] = ema_alpha * l.biases_ema[k] + (1 - ema_alpha) * l.biases[k];
                if (l.scales_ema) l.scales_ema[k] = ema_alpha * l.scales_ema[k] + (1 - ema_alpha) * l.scales[k];
            }
        }
    }
}


void ema_apply(network net)
{
    int i;
    for (i = 0; i < net.n; ++i) {
        layer l = net.layers[i];
        if (l.type == CONVOLUTIONAL) {
            int k;
            if (l.weights_ema) {
                for (k = 0; k < l.nweights; ++k) {
                    l.weights[k] = l.weights_ema[k];
                }
            }

            for (k = 0; k < l.n; ++k) {
                if (l.biases_ema) l.biases[k] = l.biases_ema[k];
                if (l.scales_ema) l.scales[k] = l.scales_ema[k];
            }

#ifdef GPU
            if (gpu_index >= 0) {
                push_convolutional_layer(l);
            }
#endif
        }
    }
}



void reject_similar_weights(network net, float sim_threshold)
{
    int i;
    for (i = 0; i < net.n; ++i) {
        layer l = net.layers[i];
        if (i == 0) continue;
        if (net.n > i + 1) if (net.layers[i + 1].type == YOLO) continue;
        if (net.n > i + 2) if (net.layers[i + 2].type == YOLO) continue;
        if (net.n > i + 3) if (net.layers[i + 3].type == YOLO) continue;

        if (l.type == CONVOLUTIONAL && l.activation != LINEAR) {
#ifdef GPU
            if (gpu_index >= 0) {
                pull_convolutional_layer(l);
            }
#endif
            int k, j;
            float max_sim = -1000;
            int max_sim_index = 0;
            int max_sim_index2 = 0;
            int filter_size = l.size*l.size*l.c;
            for (k = 0; k < l.n; ++k)
            {
                for (j = k+1; j < l.n; ++j)
                {
                    int w1 = k;
                    int w2 = j;

                    float sim = cosine_similarity(&l.weights[filter_size*w1], &l.weights[filter_size*w2], filter_size);
                    if (sim > max_sim) {
                        max_sim = sim;
                        max_sim_index = w1;
                        max_sim_index2 = w2;
                    }
                }
            }

            printf(" reject_similar_weights: i = %d, l.n = %d, w1 = %d, w2 = %d, sim = %f, thresh = %f \n",
                i, l.n, max_sim_index, max_sim_index2, max_sim, sim_threshold);

            if (max_sim > sim_threshold) {
                printf(" rejecting... \n");
                float scale = sqrt(2. / (l.size*l.size*l.c / l.groups));

                for (k = 0; k < filter_size; ++k) {
                    l.weights[max_sim_index*filter_size + k] = scale*rand_uniform(-1, 1);
                }
                if (l.biases) l.biases[max_sim_index] = 0.0f;
                if (l.scales) l.scales[max_sim_index] = 1.0f;
            }

#ifdef GPU
            if (gpu_index >= 0) {
                push_convolutional_layer(l);
            }
#endif
        }
    }
}
