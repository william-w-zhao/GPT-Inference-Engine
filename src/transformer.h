#ifndef TRANSFORMER_H
#define TRANSFORMER_H

#include "attention.h"
#include "mlp.h"
#include "weights.h"
#include "ops.h"

void forward(GPTWeights &gpt, Activations &act, const int32_t *tok_ids, int tok_size);

#endif