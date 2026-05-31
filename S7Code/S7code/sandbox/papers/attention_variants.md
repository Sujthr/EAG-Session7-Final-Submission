# Attention Mechanism Variants: MHA, GQA, MQA, Sparse, Linear, and Cross-Attention

**Key Papers:**
- *Attention Is All You Need* — Vaswani et al., Google, 2017 (Multi-Head Attention)
- *Fast Transformer Decoding: One Write-Head is All You Need* — Shazeer, 2019 (MQA)
- *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints* — Ainslie et al., Google, 2023
- *Generating Long Sequences with Sparse Transformers* — Child et al., OpenAI, 2019
- *Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention* — Katharopoulos et al., 2020

---

## Overview

The self-attention mechanism introduced in "Attention Is All You Need" computes relationships between all pairs of tokens, enabling rich contextual representations. However, its quadratic complexity and growing KV cache during autoregressive generation have motivated a family of variants that trade between expressiveness, speed, memory efficiency, and implementation simplicity. Understanding these variants is essential for evaluating and deploying modern LLMs.

## Multi-Head Attention (MHA)

Standard multi-head attention splits the model dimension d_model into h parallel attention heads, each of dimension d_k = d_model / h. For each head, separate linear projections produce queries (Q), keys (K), and values (V):

```
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) · V
```

The outputs of all heads are concatenated and projected back to d_model. Multiple heads allow the model to simultaneously attend to different representation subspaces — some heads may focus on syntax, others on coreference, others on positional proximity.

**KV Cache Cost**: During autoregressive generation, each token generated requires storing K and V vectors for all previous tokens (the "KV cache"). With H heads and sequence length N, the cache size is 2 · N · H · d_k per layer, which becomes a primary memory bottleneck for long sequences and large batch sizes.

## Multi-Query Attention (MQA)

Shazeer (2019) proposed Multi-Query Attention as a memory-efficient alternative: all attention heads **share a single set of K and V projections**, while each head retains its own Q projection. This reduces the KV cache size by a factor of H (number of heads), dramatically improving throughput for long-sequence generation.

The tradeoff: model quality degrades slightly because the shared K/V representation has lower effective capacity than per-head K/V. MQA is used in PaLM, Falcon-7B, and early versions of several production LLMs. Training with MQA from scratch generally preserves most of the quality while achieving significantly faster inference.

## Grouped-Query Attention (GQA)

GQA (Ainslie et al., 2023) interpolates between MHA and MQA by grouping the H query heads into G groups, where each group shares one K/V head. If G = H, this is standard MHA; if G = 1, this is MQA. Intermediate values of G (e.g., G = 8 for H = 32 heads) provide a Pareto-optimal tradeoff between quality and memory efficiency.

GQA also introduced **uptraining** — a technique to convert existing MHA checkpoints to GQA by mean-pooling K/V heads within each group and continuing training for a small fraction of original training compute (~5%). This allows leveraging large pretrained MHA models (e.g., LLaMA 2 70B was converted from MHA to GQA this way) without training from scratch.

GQA is now the standard in most frontier open-source models: LLaMA 2 (70B), LLaMA 3, Mistral 7B, Gemma, and Qwen2 all use GQA. It achieves near-MHA quality with near-MQA inference speed for reasonable group sizes.

## Sparse Attention

Sparse Transformers (Child et al., 2019) decompose full attention into structured sparse patterns to reduce cost from O(N²) to O(N√N) or O(N log N). Two primary patterns:

- **Strided attention**: Each token attends to every `stride`-th token (capturing long-range global patterns) plus a local window.
- **Fixed attention**: Tokens attend to a fixed set of positions at regular intervals plus all tokens in the current local window.

Combined across layers (alternating pattern types), sparse Transformers can model dependencies across very long sequences. Sparse Transformers achieved state-of-the-art results on image generation (treating pixels as sequences) and audio generation for sequences up to 12,288 tokens in 2019, well beyond what dense attention could handle.

## Linear Attention

Linear attention (Katharopoulos et al., 2020) reformulates the attention computation using a kernel trick. By replacing the softmax with a kernel function φ(·) applied to Q and K, the associativity of matrix multiplication allows computing attention in O(N) time:

```
Attention(Q, K, V) = φ(Q)(φ(K)^T V) / (φ(Q) φ(K)^T 1)
```

The inner product `φ(K)^T V` can be maintained as a running matrix during inference, making linear attention equivalent to an RNN-style recurrence at inference time (O(1) per step). This eliminates the KV cache entirely.

The tradeoff is expressiveness: the softmax normalization in standard attention is crucial for focusing sharply on relevant tokens, and linear attention approximations often perform significantly worse on tasks requiring precise selection. Methods like RWKV, RetNet, and GLA (Gated Linear Attention) have improved linear attention quality through gating mechanisms and improved kernel designs.

## Cross-Attention

Cross-attention allows one sequence (the query source) to attend to a different sequence (the key/value source). In encoder-decoder Transformers (T5, BART, Whisper), the decoder uses cross-attention to query encoder representations at each decoder layer. In diffusion models, cross-attention enables text conditioning — the image latent queries the text embedding.

Cross-attention is structurally identical to self-attention, but Q comes from one sequence and K, V come from another. It is the mechanism by which multi-modal models fuse information across modalities.

## Significance

The progression from MHA to MQA to GQA reflects a systematic optimization of the inference bottleneck introduced by the KV cache. Sparse and linear attention address the training-time quadratic complexity. These variants collectively enable practical deployment of LLMs at scale — without them, 7B+ parameter models generating thousands of tokens would be impractical. GQA in particular has become a near-universal design choice in 2023–2024 model architectures.
