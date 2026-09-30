#ifndef ATTENTION_H
#define ATTENTION_H

#include "activations.h"
#include "weights.h"

void attention_forward(float *out, Activations &act, float *k_cache, float *v_cache, float *x, TransformerLayer &l, int new_tok_size, int emb_size, int n_head, int pos);

#endif