#include "transformer.h"

#include <stdexcept>

#include "attention.h"
#include "mlp.h"
#include "ops.h"

// tok_size only refers to uncached tokens
void forward(GPTWeights &gpt, Activations &act, const int32_t *tok_ids, int new_tok_size, int pos)
{
    GPTConfig &config = gpt.config;

    if (pos + new_tok_size > config.context_window)
    {
        throw std::runtime_error("sequence too long");
    }

    float *x = act.x.data();
    float *ln = act.ln.data();

    int vocab_size = config.vocab_size;
    int context_window = config.context_window;
    int emb_size = config.n_embd;
    int n_head = config.n_head;

    embedding(x, tok_ids, gpt.wte_weights.data(), gpt.wpe_weights.data(), new_tok_size, emb_size, pos);

    for (int i = 0; i < gpt.layers.size(); i++)
    {
        TransformerLayer &l = gpt.layers[i];

        float *k_cache = act.k_cache.data() + (size_t)i * context_window * emb_size;
        float *v_cache = act.v_cache.data() + (size_t)i * context_window * emb_size;

        layer_norm(ln, x, l.ln1_weights.data(), l.ln1_bias.data(), new_tok_size, emb_size);
        attention_forward(act.mlp.data(), act, k_cache, v_cache, ln, l, new_tok_size, emb_size, n_head, pos);
        add(x, act.mlp.data(), new_tok_size * emb_size);

        layer_norm(ln, x, l.ln2_weights.data(), l.ln2_bias.data(), new_tok_size, emb_size);
        mlp_forward(act.mlp.data(), act.mlp_hidden.data(), ln, l, new_tok_size, emb_size);
        add(x, act.mlp.data(), new_tok_size * emb_size);
    }

    layer_norm(ln, x, gpt.lnf_weights.data(), gpt.lnf_bias.data(), new_tok_size, emb_size);

    // retrieve last token tensor for prediction
    const float *last = ln + (size_t)(new_tok_size - 1) * emb_size;
    matmul(act.logits.data(), last, gpt.wte_weights_transposed.data(), 1, emb_size, vocab_size);
}