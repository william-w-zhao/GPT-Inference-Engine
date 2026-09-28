#include <cmath>
#include <cstdint>
#include <vector>
#include <numbers>

#include "weights.h"

// input: out, token ids, wte, wpe, input size in tokens, embedding size
void embedding(float *out, int32_t *tok_ids, float *wte, float *wpe, int tok_size, int emb_size)
{
    for (int t = 0; t < tok_size; t++)
    {
        const float *tok = wte + (size_t)tok_ids[t] * emb_size;
        const float *pos = wpe + (size_t)t * emb_size;
        float *ot = out + (size_t)t * emb_size;
        for (int i = 0; i < emb_size; i++)
        {
            ot[i] = tok[i] + pos[i];
        }
    }
}

// input: out, x, weights, bias, input size in tokens, embedding size
void layer_norm(float *out, const float *x, const float *weights, const float *bias, int tok_size, int emb_size)
{
    const float eps = 1e-5f;

    for (int t = 0; t < tok_size; t++)
    {
        const float *xt = x + (size_t)t * emb_size;

        // mean
        float mean = 0.0f;
        for (int i = 0; i < emb_size; i++)
        {
            mean += xt[i];
        }
        mean /= emb_size;

        // variance
        float variance = 0.0f;
        for (int i = 0; i < emb_size; i++)
        {
            float d = xt[i] - mean;
            variance += d * d;
        }
        variance /= emb_size;

        // compute normalization and affine transformation
        float *ot = out + (size_t)t * emb_size;
        for (int i = 0; i < emb_size; i++)
        {
            ot[i] = ((xt[i] - mean) / std::sqrt(variance + eps)) * weights[i] + bias[i];
        }
    }
}

// xW + b
// input: out, x, weights, bias, input size in tokens, embedding size
void linear(float *out, const float *x, const float *weights, const float *bias, int tok_size, int emb_size)
{
    matmul(out, x, weights, tok_size, emb_size, emb_size);
    for (int t = 0; t < tok_size; t++)
    {
        float *ot = out + (size_t)t * emb_size;
        for (int i = 0; i < emb_size; i++)
        {
            ot[i] += bias[i];
        }
    }
}

// input: out, x, weights, bias, input size in tokens, embedding size
void GELU(float *x, int n)
{
    for (int i = 0; i < n; i++)
    {
        const float sqrt = std::sqrt(2.0f / 3.14159265358979f);
        const float hyp_tan = std::tanh(sqrt * (x[i] + 0.044715f * x[i] * x[i] * x[i]));
        x[i] = 0.5f * x[i] * (1.0f + hyp_tan);
    }
}

// input: out, x, weights, bias, input size in tokens, embedding size
void softmax(float *x, int n)
{
    const float e = std::exp(1.0);

    float sum = 0.0f;
    for (int i = 0; i < n; i++)
    {
        x[i] = std::pow(e, x[i]);
        sum += x[i];
    }

    for (int i = 0; i < n; i++)
    {
        x[i] /= sum;
    }
}

// input: out, x, weights, bias, input size in tokens, embedding size
// output = [M x P] = a x b = [M x N] x [N x P]
void matmul(float *out, const float *a, const float *b, int M, int N, int P)
{
    // compute dot product for each row of A, column of B
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < P; j++)
        {
            float sum = 0.0f;
            for (int k = 0; k < N; k++)
            {
                sum += a[(size_t)i * N + k] * b[(size_t)k * P + j];
            }
            out[(size_t)i * P + j] = sum;
        }
    }
}