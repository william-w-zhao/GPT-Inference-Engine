#include "attention.h"

#include "ops.h"

void attention_forward(float *out, Activations &act, float *x, TransformerLayer &l, int tok_size, int emb_size, int n_head)
{
    linear(act.q.data(), x, l.q_weights.data(), l.q_bias.data(), tok_size, emb_size, emb_size);
    linear(act.k.data(), x, l.k_weights.data(), l.k_bias.data(), tok_size, emb_size, emb_size);
    linear(act.v.data(), x, l.v_weights.data(), l.v_bias.data(), tok_size, emb_size, emb_size);

    causal_attention(act.attn.data(), act.attn_weights.data(), act.q.data(), act.k.data(), act.v.data(), tok_size, emb_size, n_head);

    linear(out, act.attn.data(), l.attn_proj_weights.data(), l.attn_proj_bias.data(), tok_size, emb_size, emb_size);
}