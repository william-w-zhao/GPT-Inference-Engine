#ifndef OPS_H
#define OPS_H

#include <cstdint>

void embedding(float *out, int32_t *tok_ids, float *wte, float *wpe, int tok_size, int emb_size);

void layer_norm(float *out, const float *x, const float *weights, const float *bias, int tok_size, int emb_size);

void linear(float *out, const float *x, const float *weights, const float *bias, int tok_size, int emb_size);

void GELU(float *x, int n);

void softmax(float *x, int n);

void matmul(float *out, const float *a, const float *b, int M, int N, int P);

#endif