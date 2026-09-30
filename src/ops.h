#ifndef OPS_H
#define OPS_H

#include <cstdint>

// input: out, token ids, wte, wpe, input size in tokens, embedding size
// output: token represented by token embedding and position embedding
void embedding(float *out, const int32_t *tok_ids, const float *wte, const float *wpe, int tok_size, int emb_size, int pos);

// input: out, x, weights, bias, input size in tokens, embedding size
// output: normalized x, with mean 0 and variance 1, combined with affine transformation
void layer_norm(float *out, const float *x, const float *weights, const float *bias, int tok_size, int emb_size);

// input: out, dot-product buffer, q, k, v, input size in tokens, embedding size, number of attention heads
// output: weighted sum of values, weights are softmaxed dot-products of query and key values
void causal_attention(float *out, float *attn_weights, const float *q, const float *k_cache, const float *v_cache, int new_tok_size, int emb_size, int n_head, int pos);

// input: x, input size
// output: GELU-applied to x, in-place
void GELU(float *x, int n);

// input: x, input size
// output: softmax-applied to x, in-place
void softmax(float *x, int n);

// input: out, x, weights, bias, input size in tokens, dimension of input matrix, dimension of output matrix
// output: xW + b
void linear(float *out, const float *x, const float *weights, const float *bias, int tok_size, int in_dim, int out_dim);

// input: out, matrix a, matrix b, dimension of input matrix, common dimension, dimension of output matrix
// output = [M x P] = a x b = [M x N] x [N x P]
void matmul(float *out, const float *a, const float *b, int M, int N, int P);

// input: x, y, input size
// output: x += y in-place
void add(float *x, const float *y, int n);

// input: x [rows x cols]
// output: out [cols x rows]
void transpose(float *out, const float *x, int rows, int cols);

#endif