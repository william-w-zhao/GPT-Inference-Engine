#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

#include <vector>

#include "weights.h"

struct Activations
{
    // residual stream
    std::vector<float> x;
    // layer normalization vectors (mean 0, variance 1)
    std::vector<float> ln;
    // query
    std::vector<float> q;
    // KV caching
    std::vector<float> k_cache;
    std::vector<float> v_cache;
    // query, value dot products
    std::vector<float> attn_weights;
    // attention output
    std::vector<float> attn;
    // mlp hidden layer for non-linear functions
    std::vector<float> mlp_hidden;
    // mlp output
    std::vector<float> mlp;
    // logits
    std::vector<float> logits;

    Activations(GPTConfig &config) : x(config.context_window * config.n_embd),
                                     ln(config.context_window * config.n_embd),
                                     q(config.context_window * config.n_embd),
                                     k_cache((size_t)config.n_layer * config.context_window * config.n_embd),
                                     v_cache((size_t)config.n_layer * config.context_window * config.n_embd),
                                     attn_weights(config.context_window),
                                     attn(config.context_window * config.n_embd),
                                     mlp_hidden(config.context_window * 4 * config.n_embd),
                                     mlp(config.context_window * config.n_embd),
                                     logits(config.vocab_size)
    {
    }
};

#endif