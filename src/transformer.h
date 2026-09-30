#ifndef TRANSFORMER_H
#define TRANSFORMER_H

#include <cstdint>

#include "activations.h"
#include "weights.h"

void forward(GPTWeights &gpt, Activations &act, const int32_t *tok_ids, int tok_size);

#endif