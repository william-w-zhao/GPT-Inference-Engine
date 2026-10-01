## GPT Inference Engine

This is a C++ implementation of a GPT-2 inference engine, with ~124M weights loaded from HuggingFace. No libraries are used
in this implementation, but all transformer processes, attention, MLP, KV-caching, and tokenizer decoding are all implemented
through editing registers of floats.

## Features

- Weight and vocab initialization through reading of .bin files
- Token and positional embedding
- Layer normalization
- 12 attention and multilayer perceptron blocks (linear, multi-head causal attention, GELU)
- **KV caching**
- **Cache friendly matrix multiplication with NEON SIMD**, reading and writing into 4 consecutive floats in memory
- Benchmarking code for prefill and decode

## Performance

GPT-2 124M, single CPU thread

| Version                    | Prefill (128 tokens) | Decode                   |
| -------------------------- | -------------------- | ------------------------ |
| Naive                      | 8605 ms (14.9 tok/s) | 1450 ms/tok (0.7 tok/s)  |
| + KV cache                 | 8694 ms (14.7 tok/s) | 101 ms/tok (9.9 tok/s)   |
| + Matmul SIMD optimization | 753 ms (170.0 tok/s) | 9.5 ms/tok (105.2 tok/s) |

See [BENCHMARK.md]

## Project structure

```
src/
  main.cpp           greedy text generation
  transformer.cpp    forward pass, embedding
  attention.cpp      QKV projections, KV cache writes, attention
  mlp.cpp            feed-forward block
  ops.cpp            matrix multiplication, layer normalization, softmax, GELU, causal attention
  activations.h      preallocated buffers, KV cache
  weights.cpp        weight loading
  weights.h          weight layout
  tokenizer.cpp      vocab loading
benchmark/
  benchmark.cpp      load / prefill / decode timings
tools/
  model_loader.py    exports weights.bin and vocab.bin
  reference.py       reference logits from HuggingFace for correctness
```

## Setup

Requires a C++20 compiler and Python 3 with `torch`, `transformers` and `numpy`.
