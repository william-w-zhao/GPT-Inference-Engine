#include <cmath>
#include <cstdint>
#include <vector>
#include <numbers>

#include "weights.h"
#include "ops.h"

void embedding(float *out, const int32_t *tok_ids, const float *wte, const float *wpe, int tok_size, int emb_size)
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

void causal_attention(float *out, float *attn_weights, const float *q, const float *k, const float *v, int tok_size, int emb_size, int n_head)
{
    const int head_size = emb_size / n_head;
    const float scale = 1.0f / std::sqrt((float)head_size);

    // for each attention head
    for (int h = 0; h < n_head; h++)
    {
        // for each token
        for (int t = 0; t < tok_size; t++)
        {
            const float *qt = q + (size_t)t * emb_size + h * head_size;
            // causal mask
            for (int i = 0; i <= t; i++)
            {
                const float *kt = k + (size_t)i * emb_size + h * head_size;
                float dot = 0.0f;
                for (int d = 0; d < head_size; d++)
                {
                    dot += qt[d] * kt[d];
                }
                attn_weights[i] = dot * scale;
            }
            softmax(attn_weights, t + 1);

            float *ot = out + (size_t)t * emb_size + h * head_size;
            for (int d = 0; d < head_size; d++)
            {
                float sum = 0.0f;
                for (int i = 0; i <= t; i++)
                    sum += attn_weights[i] * v[(size_t)i * emb_size + h * head_size + d];
                ot[d] = sum;
            }
        }
    }
}

void GELU(float *x, int n)
{
    const float sqrt = std::sqrt(2.0f / 3.14159265358979f);
    for (int i = 0; i < n; i++)
    {
        const float hyp_tan = std::tanh(sqrt * (x[i] + 0.044715f * x[i] * x[i] * x[i]));
        x[i] = 0.5f * x[i] * (1.0f + hyp_tan);
    }
}

void softmax(float *x, int n)
{
    const float e = std::exp(1.0);

    float max = x[0];

    // prevent infinity overflow
    for (int i = 1; i < n; i++)
    {
        max = std::max(max, x[i]);
    }

    float sum = 0.0f;
    for (int i = 0; i < n; i++)
    {
        x[i] = std::exp(x[i] - max);
        sum += x[i];
    }

    for (int i = 0; i < n; i++)
    {
        x[i] /= sum;
    }
}

void linear(float *out, const float *x, const float *weights, const float *bias, int tok_size, int in_dim, int out_dim)
{
    matmul(out, x, weights, tok_size, in_dim, out_dim);
    for (int t = 0; t < tok_size; t++)
    {
        float *ot = out + (size_t)t * out_dim;
        for (int i = 0; i < out_dim; i++)
        {
            ot[i] += bias[i];
        }
    }
}

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

void add(float *x, const float *y, int n)
{
    for (int i = 0; i < n; i++)
        x[i] += y[i];
}

void transpose(float *out, const float *x, int rows, int cols)
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            out[(size_t)j * rows + i] = x[(size_t)i * cols + j];
        }
    }
}