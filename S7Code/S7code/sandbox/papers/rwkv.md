# RWKV: Reinventing RNNs for the Transformer Era

**Authors:** Bo Peng, Eric Alcaide, Quentin Anthony, Alon Albalak, Samuel Arcadinho, Huanqi Cao, Xin Cheng, Michael Chung, Matteo Grella, Kranthi Kiran GV, Xuzheng He, Haowen Hou, Przemyslaw Kazienko, Jan Kocon, Jiaming Kong, Bartłomiej Koptyra, Hayden Lau, Krishna Sri Ipsit Mantri, Ferdinand Mom, Atsushi Saito, Xiangru Tang, Bolun Wang, Johan S. Wind, Stansilaw Wozniak, Ruichong Zhang, Zhenyuan Zhang, Qihang Zhao, Peng Zhou, Jian Zhu, Rui-Jie Zhu  
**Year:** 2023  
**Paper:** arXiv:2305.13048  
**Repository:** github.com/BlinkDL/RWKV-LM

---

## Overview

RWKV (pronounced "RWaKuV") is a novel architecture that combines the training parallelism of transformers with the O(1) inference complexity of recurrent neural networks. It achieves this by reformulating attention as a linear operation that can be computed either as a parallel scan (during training, like a transformer) or as a recurrent update (during inference, like an RNN). The result is a model that scales to billions of parameters, trains efficiently on modern GPU clusters, yet generates tokens with constant-time and constant-memory per step — eliminating the KV-cache bottleneck that makes transformer inference expensive.

RWKV represents a fundamental rethinking of the sequence modeling problem: rather than accepting the quadratic cost of full attention as necessary for quality, it asks whether a carefully designed linear recurrence can match transformer quality while recovering the efficiency benefits of RNNs.

---

## The Core Problem with Transformers and RNNs

**Transformers:** Full self-attention is O(n²) in sequence length at both training and inference. At inference, the KV-cache grows linearly with sequence length — for a 13B parameter model generating at 4096 tokens, the KV-cache alone requires ~16GB of memory. Long-context inference is bottlenecked by memory bandwidth.

**Standard RNNs (LSTM, GRU):** O(1) inference (fixed-size hidden state, one recurrent update per token). But training is sequential — backpropagation through time (BPTT) cannot be parallelized across timesteps, making training on modern GPU hardware inefficient. Additionally, RNNs struggle with long-range dependencies due to vanishing/exploding gradients.

**RWKV's position:** Train like a transformer (parallel, on GPU), infer like an RNN (sequential, constant memory). Achieves transformer-quality language modeling with RNN-class inference efficiency.

---

## Architecture

RWKV is an autoregressive language model (decoder-only) with a stack of RWKV blocks. Each block contains two components: **the Time-mixing block** (analogous to attention) and **the Channel-mixing block** (analogous to feed-forward). Both use only element-wise operations and simple linear projections — no softmax attention.

### Time-Mixing (The "WKV" Mechanism)

The core of RWKV is the WKV operator, which computes a weighted sum of past values:

For each token position t and channel d:

```
wkv_t = (Σₛ≤ₜ exp(-(t-1-s)·w + k_s) · v_s) / (Σₛ≤ₜ exp(-(t-1-s)·w + k_s))
```

Where:
- **w** is a channel-wise learned decay vector (negative, so exp(-(t-s)w) decays with distance)
- **k_s** is the "key" at position s (a linear projection of the input)
- **v_s** is the "value" at position s

This is essentially exponentially weighted attention where past tokens are down-weighted by a learned decay factor w. The formula resembles the attention formula but with a fixed (not query-dependent) weighting by recency.

**R (receptance):** A sigmoid-gated output: `output = σ(r_t) ⊙ wkv_t` where r_t is a linear projection of the input.

**Time-shift:** Input to each sub-layer is a linear interpolation between the current token's embedding and the previous token's embedding: `x_shifted = lerp(x_{t-1}, x_t, μ)` where μ is a learned channel-wise interpolation parameter. This gives the model access to both the current token and recent history even before the recurrent computation.

### Recurrent Formulation for Inference

The WKV computation can be rewritten as an RNN state update:

```
a_t = exp(-w + k_t) · v_t + exp(-w) · a_{t-1}
b_t = exp(-w + k_t) + exp(-w) · b_{t-1}
wkv_t = a_t / b_t
```

The state (a_t, b_t) is constant-size (proportional to the number of channels × layers), regardless of sequence length. Generating each new token requires only updating this state — no growing KV-cache, no quadratic computation.

### Parallel Formulation for Training

During training, the entire WKV sequence can be computed as a parallel prefix scan (analogous to parallel scan algorithms in functional programming). This allows full GPU parallelism across the time dimension, achieving training throughput comparable to transformers.

### Channel-Mixing Block

Analogous to the feed-forward network in a transformer, but using a similar time-shift and gating mechanism. The channel-mixing block uses:
```
output = σ(r_t) ⊙ (W_v · (σ(W_k · x_shifted)²))
```
A squared ReLU gate (σ(·)²) is used as the nonlinearity.

---

## Training Details

RWKV uses standard autoregressive next-token prediction with cross-entropy loss. Model sizes range from 169M to 14B parameters. Training uses:
- Adam optimizer with learning rate warmup and decay
- Mixed precision (bfloat16)
- Pile dataset (300B tokens), consistent with other comparable open-source models

The RWKV-4 series (the primary model in the paper) was trained at scales of 169M, 430M, 1.5B, 3B, 7B, and 14B parameters. The 14B model was trained on approximately 330B tokens.

---

## Results

RWKV is evaluated against transformers (GPT-3, OPT, Pythia, BLOOM) at comparable parameter counts on standard language modeling benchmarks:

- **Pile perplexity:** RWKV achieves perplexity competitive with transformer models of equivalent size, closing to within 1-2 perplexity points of comparably-trained transformers.
- **Zero-shot tasks (Lambada, HellaSwag, PIQA, ARC, WinoGrande):** RWKV matches Pythia and OPT at comparable sizes.
- **Long-context:** RWKV maintains constant memory usage regardless of sequence length, enabling context lengths impractical for transformers without special modifications.

RWKV does not quite match the best transformers at each parameter count, but the gap is small and closing with architectural improvements (RWKV-5, RWKV-6 incorporate attention-like improvements).

---

## RWKV-5 and RWKV-6 Improvements

- **RWKV-5** introduced multi-head WKV (analogous to multi-head attention), improving model expressivity.
- **RWKV-6** introduced data-dependent decay (w is now input-dependent, not just channel-dependent), making the temporal weighting dynamic rather than fixed — a key step toward closing the quality gap with full attention.

---

## Significance

RWKV demonstrates that the transformer's quadratic attention is not a necessary condition for language model quality — it is an architectural choice with alternatives. By recovering the linear-time inference of RNNs without sacrificing training efficiency or model quality, RWKV opens practical paths to:
- Streaming inference without KV-cache memory explosion
- Deployment on memory-constrained edge devices
- Infinite-context generation (no hard sequence length limit)
- Real-time applications where latency per token must be constant

RWKV is one of the most important developments in the "post-transformer" architectural research thread, alongside Mamba (state space models), Hyena, and RetNet — all seeking the same goal of linear-time sequence modeling without sacrificing quality.
