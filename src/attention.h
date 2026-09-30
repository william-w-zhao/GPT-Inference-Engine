#ifndef ATTENTION_H
#define ATTENTION_H

#include "weights.h"
#include "activations.h"
#include "ops.h"

void attention_forward(float *out, Activations &act, float *x, TransformerLayer &l, int tok_size, int emb_size, int n_head);

#endif