#include <stdexcept>

#include "transformer.h"

void forward(GPTWeights &gpt, Activations &act, const int32_t *tok_ids, int tok_size)
{
    GPTConfig &config = gpt.config;

    if (tok_size > config.context_window)
    {
        throw std::runtime_error("sequence too long");
    }

    float *x = act.x.data();
    float *ln = act.ln.data();

    int vocab_size = config.vocab_size;
    int emb_size = config.n_embd;
    int n_head = config.n_head;

    embedding(x, tok_ids, gpt.wte_weights.data(), gpt.wpe_weights.data(), tok_size, emb_size);

    for (TransformerLayer &l : gpt.layers)
    {
        layer_norm(ln, x, l.ln1_weights.data(), l.ln1_bias.data(), tok_size, emb_size);
        attention_forward(act.mlp.data(), act, ln, l, tok_size, emb_size, n_head);
        add(x, act.mlp.data(), tok_size * emb_size);

        layer_norm(ln, x, l.ln2_weights.data(), l.ln2_bias.data(), tok_size, emb_size);
        mlp_forward(act.mlp.data(), act.mlp_hidden.data(), ln, l, tok_size, emb_size);
        add(x, act.mlp.data(), tok_size * emb_size);
    }

    layer_norm(ln, x, gpt.lnf_weights.data(), gpt.lnf_bias.data(), tok_size, emb_size);

    // retrieve last token tensor for prediction
    const float *last = ln + (size_t)(tok_size - 1) * emb_size;
    matmul(act.logits.data(), last, gpt.wte_weights_transposed.data(), 1, emb_size, vocab_size);
}