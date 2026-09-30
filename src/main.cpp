#include <algorithm>
#include <cstdio>
#include <iostream>
#include <string>

#include "activations.h"
#include "tokenizer.h"
#include "transformer.h"
#include "weights.h"

int main()
{
    std::vector<int32_t> tok_ids = {464, 3797, 3332, 319, 262};
    std::string model_path = "models/weights.bin";
    GPTWeights gpt = load_weights(model_path);
    GPTConfig &config = gpt.config;

    Activations act = Activations(config);

    std::string vocab_path = "models/vocab.bin";
    std::vector<std::string> vocab = load_vocab(vocab_path);

    const int max_steps = 10;
    for (int step = 0; step < max_steps; step++)
    {

        if (tok_ids.size() > config.context_window)
        {
            break;
        }

        forward(gpt, act, tok_ids.data(), (int)tok_ids.size());

        // argmax
        int next_tok = std::max_element(act.logits.begin(), act.logits.end()) - act.logits.begin();

        tok_ids.push_back(next_tok);
        std::cout << vocab[next_tok] << std::flush;
    }

    return 0;
}