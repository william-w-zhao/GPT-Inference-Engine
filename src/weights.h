#ifndef WEIGHTS_H
#define WEIGHTS_H

#include <vector>

struct TransformerLayer
{
    std::vector<float> ln1_weights; // [768]
    std::vector<float> ln1_bias;    // [768]

    std::vector<float> q_weights; // [768, 768]
    std::vector<float> k_weights; // [768, 768]
    std::vector<float> v_weights; // [768, 768]

    std::vector<float> q_bias; // [768]
    std::vector<float> k_bias; // [768]
    std::vector<float> v_bias; // [768]

    std::vector<float> attn_proj_weights; // [768, 768]
    std::vector<float> attn_proj_bias;    // [768]

    std::vector<float> ln2_weights; // [768]
    std::vector<float> ln2_bias;    // [768]

    std::vector<float> mlp_fc_weights; // [768, 3072]
    std::vector<float> mlp_fc_bias;    // [3072]

    std::vector<float> mlp_proj_weights; // [3072, 768]
    std::vector<float> mlp_proj_bias;    // [768]
};

struct GPTConfig
{
    int vocab_size;
    int context_window;
    int n_layer;
    int n_head;
    int n_embd;
};

struct GPTWeights
{
    GPTConfig config;

    std::vector<float> wte_weights; // [50257, 768]
    std::vector<float> wpe_weights; // [1024, 768]

    std::vector<TransformerLayer> layers;

    std::vector<float> lnf_weights; // [768]
    std::vector<float> lnf_bias;    // [768]
};

GPTWeights load_weights(char *path);

#endif