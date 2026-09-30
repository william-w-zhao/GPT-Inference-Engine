#ifndef MLP_H
#define MLP_H

#include "weights.h"

void mlp_forward(float *out, float *hidden, float *x, TransformerLayer &l, int tok_size, int emb_size);

#endif