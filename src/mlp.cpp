#include "mlp.h"

#include "ops.h"

void mlp_forward(float *out, float *hidden, float *x, TransformerLayer &l, int tok_size, int emb_size)
{
    const size_t feed_forward = 4 * emb_size;
    linear(hidden, x, l.mlp_fc_weights.data(), l.mlp_fc_bias.data(), tok_size, emb_size, feed_forward);
    GELU(hidden, tok_size * feed_forward);
    linear(out, hidden, l.mlp_proj_weights.data(), l.mlp_proj_bias.data(), tok_size, feed_forward, emb_size);
}