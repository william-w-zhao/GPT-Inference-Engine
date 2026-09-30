#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "activations.h"
#include "transformer.h"
#include "weights.h"

using Clock = std::chrono::steady_clock;

double ms_since(Clock::time_point t0)
{
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

double median(std::vector<double> v)
{
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

void measure_prefill(GPTWeights &gpt, Activations &act, int tok_size, int k)
{
    std::vector<int32_t> tok_ids(tok_size);
    for (int i = 0; i < tok_size; i++)
        tok_ids[i] = i;

    forward(gpt, act, tok_ids.data(), tok_size);

    std::vector<double> times;
    for (int i = 0; i < k; i++)
    {
        auto t0 = Clock::now();
        forward(gpt, act, tok_ids.data(), tok_size);
        times.push_back(ms_since(t0));
    }

    double ms = median(times);
    std::cout << "prefill " << tok_size << ":  " << ms << " ms   ("
              << tok_size / (ms / 1000.0) << " tok/s)\n";
}

void measure_decode(GPTWeights &gpt, Activations &act, int steps)
{
    std::vector<int32_t> tok_ids = {464, 3797, 3332, 319, 262};

    std::vector<double> times;
    for (int step = 0; step < steps; step++)
    {
        auto t0 = Clock::now();
        forward(gpt, act, tok_ids.data(), (int)tok_ids.size());
        int next = std::max_element(act.logits.begin(), act.logits.end()) - act.logits.begin();
        tok_ids.push_back(next);
        times.push_back(ms_since(t0));
    }

    double total = std::accumulate(times.begin(), times.end(), 0.0);
    double avg = total / steps;
    std::cout << "decode " << steps << ":  " << avg << " ms/tok   ("
              << 1000.0 / avg << " tok/s)   first " << times.front()
              << " ms, last " << times.back() << " ms\n";
}

int main()
{
    std::string model_path = "models/weights.bin";

    auto t0 = Clock::now();
    GPTWeights gpt = load_weights(model_path);
    std::cout << "load:  " << ms_since(t0) << " ms\n";

    Activations act(gpt.config);

    measure_prefill(gpt, act, 128, 10);
    measure_decode(gpt, act, 32);

    return 0;
}