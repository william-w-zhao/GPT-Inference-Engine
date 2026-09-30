#ifndef ATTENTION_H
#define ATTENTION_H

#include "activations.h"
#include "weights.h"

void attention_forward(float *out, Activations &act, float *x, TransformerLayer &l, int tok_size, int emb_size, int n_head);

#endif