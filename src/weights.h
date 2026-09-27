#ifndef WEIGHTS_H
#define WEIGHTS_H

#include <vector>

struct TransformerLayer
{
    std::vector<float> ln1_weights;
    std::vector<float> ln1_bias;

    std::vector<float> q_weights;
    std::vector<float> k_weights;
    std::vector<float> v_weights;

    std::vector<float> q_bias;
    std::vector<float> k_bias;
    std::vector<float> v_bias;

    std::vector<float> attn_proj_weights;
    std::vector<float> attn_proj_bias;

    std::vector<float> ln2_weights;
    std::vector<float> ln2_bias;

    std::vector<float> mlp_fc_weights;
    std::vector<float> mlp_fc_bias;

    std::vector<float> mlp_proj_weights;
    std::vector<float> mlp_proj_bias;
};

struct GPT2Weights
{
    int vocab_size = 50257;
    int context_window = 1024;
    int n_layer = 12;
    int n_head = 12;
    int n_embd = 3;

    std::vector<float> wte_weights;
    std::vector<float> wpe_weights;

    std::vector<TransformerLayer> layers;

    std::vector<float> lnf_weights;
    std::vector<float> lnf_bias;
};

#endif