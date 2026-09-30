#include "weights.h"

#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "ops.h"

// input: file
GPTConfig read_header(std::ifstream &f)
{
    int32_t header[256];
    f.read(reinterpret_cast<char *>(header), sizeof(header));

    if (!f)
    {
        throw std::runtime_error("unable to read header");
    }

    GPTConfig config;
    config.vocab_size = header[0];
    config.context_window = header[1];
    config.n_layer = header[2];
    config.n_head = header[3];
    config.n_embd = header[4];

    return config;
}

// input: file, output vector, tensor size
void read_tensor(std::ifstream &f, std::vector<float> &v, size_t n)
{
    v.resize(n);
    f.read(reinterpret_cast<char *>(v.data()), n * sizeof(float));
    if (!f)
    {
        throw std::runtime_error("unable to read weights file");
    }
}

// input: path to weights file
GPTWeights load_weights(std::string &path)
{
    std::ifstream f(path, std::ifstream::binary);
    if (!f)
    {
        throw std::runtime_error("unable to open weights file");
    }

    GPTWeights gpt;
    gpt.config = read_header(f);
    const GPTConfig &config = gpt.config;

    const size_t vocab_size = config.vocab_size;
    const size_t context_window = config.context_window;
    const size_t emb_size = config.n_embd;
    const size_t feed_forward = 4 * config.n_embd;

    gpt.layers.resize(config.n_layer);

    read_tensor(f, gpt.wte_weights, vocab_size * emb_size);
    read_tensor(f, gpt.wpe_weights, context_window * emb_size);

    gpt.wte_weights_transposed.resize(vocab_size * emb_size);
    transpose(gpt.wte_weights_transposed.data(), gpt.wte_weights.data(), vocab_size, emb_size);

    for (TransformerLayer &l : gpt.layers)
    {
        read_tensor(f, l.ln1_weights, emb_size);
        read_tensor(f, l.ln1_bias, emb_size);

        read_tensor(f, l.q_weights, emb_size * emb_size);
        read_tensor(f, l.k_weights, emb_size * emb_size);
        read_tensor(f, l.v_weights, emb_size * emb_size);

        read_tensor(f, l.q_bias, emb_size);
        read_tensor(f, l.k_bias, emb_size);
        read_tensor(f, l.v_bias, emb_size);

        read_tensor(f, l.attn_proj_weights, emb_size * emb_size);
        read_tensor(f, l.attn_proj_bias, emb_size);

        read_tensor(f, l.ln2_weights, emb_size);
        read_tensor(f, l.ln2_bias, emb_size);

        read_tensor(f, l.mlp_fc_weights, feed_forward * emb_size);
        read_tensor(f, l.mlp_fc_bias, feed_forward);

        read_tensor(f, l.mlp_proj_weights, feed_forward * emb_size);
        read_tensor(f, l.mlp_proj_bias, emb_size);
    }

    read_tensor(f, gpt.lnf_weights, emb_size);
    read_tensor(f, gpt.lnf_bias, emb_size);

    if (f.peek() != std::ifstream::traits_type::eof())
    {
        throw std::runtime_error("weights file has remaining data");
    }

    return gpt;
}