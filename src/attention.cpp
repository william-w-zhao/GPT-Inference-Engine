#include "attention.h"

#include "ops.h"

void attention_forward(float *out, Activations &act, float *k_cache, float *v_cache, float *x, TransformerLayer &l, int new_tok_size, int emb_size, int n_head, int pos)
{
    float *k_new = k_cache + (size_t)pos * emb_size;
    float *v_new = v_cache + (size_t)pos * emb_size;

    linear(act.q.data(), x, l.q_weights.data(), l.q_bias.data(), new_tok_size, emb_size, emb_size);
    linear(k_new, x, l.k_weights.data(), l.k_bias.data(), new_tok_size, emb_size, emb_size);
    linear(v_new, x, l.v_weights.data(), l.v_bias.data(), new_tok_size, emb_size, emb_size);

    causal_attention(act.attn.data(), act.attn_weights.data(), act.q.data(), k_cache, v_cache, new_tok_size, emb_size, n_head, pos);

    linear(out, act.attn.data(), l.attn_proj_weights.data(), l.attn_proj_bias.data(), new_tok_size, emb_size, emb_size);
}