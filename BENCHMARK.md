## Basic Implementation (without KV caching or matmul optimization)

load: 356.579 ms
prefill (128 tokens, averaged over 10 samples): 8605.42 ms (14.8744 tok/s)
decode (32 tokens): 1450.13 ms/tok (0.689595 tok/s) first 368.235 ms, last 2596.23 ms

## KV Caching Optimized Implementation

load: 207.115 ms
prefill (128 tokens, averaged over 10 samples): 8694.01 ms (14.7228 tok/s)
decode (32 tokens): 101.493 ms/tok (9.85292 tok/s) first 101.632 ms, last 99.8701 ms

## MatMul Optimized Implementation (SIMD friendly)

load: 174.083 ms
prefill (128 tokens, averaged over 10 samples): 752.815 ms (170.029 tok/s)
decode (32 tokens): 9.49983 ms/tok (105.265 tok/s) first 9.41296 ms, last 9.68429 ms
