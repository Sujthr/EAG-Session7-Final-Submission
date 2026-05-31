# Mistral 7B: Efficient Language Modeling with Sliding Window and Grouped-Query Attention

**Authors:** Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, Lélio Renard Lavaud, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timothée Lacroix, William El Sayed
**Institution:** Mistral AI
**Year:** 2023

---

## Overview

Mistral 7B is a 7-billion parameter language model released by Mistral AI in September 2023 that outperforms LLaMA 2 13B on most benchmarks and approaches LLaMA 2 34B on several tasks, despite having significantly fewer parameters. Its efficiency stems from two architectural innovations — **Sliding Window Attention (SWA)** and **Grouped-Query Attention (GQA)** — combined with high-quality training data and careful hyperparameter choices. Mistral 7B was released with open weights under the Apache 2.0 license, making it one of the most capable freely available models at the time.

## Architecture

Mistral 7B is a decoder-only Transformer using the LLaMA architecture as a base, with several modifications:

| Hyperparameter | Value |
|----------------|-------|
| Hidden dimension | 4096 |
| Number of layers | 32 |
| Attention heads (Q) | 32 |
| KV heads (GQA) | 8 |
| Sliding window size | 4096 |
| Vocabulary size | 32,000 |
| Max sequence length | 8192 |
| Activation | SwiGLU |
| Position embedding | RoPE |
| Normalization | RMSNorm |

The use of SwiGLU activation, RMSNorm (pre-norm), and RoPE follows the LLaMA design, placing Mistral firmly in the dominant architectural lineage of 2023 open-source LLMs.

## Sliding Window Attention (SWA)

Standard self-attention attends to all previous tokens (up to the context limit), giving O(N²) complexity and a KV cache that grows linearly with sequence length. Sliding Window Attention restricts each token to attending only to the W most recent tokens (W = 4096 in Mistral 7B), giving O(N·W) attention computation.

**Receptive field growth across layers**: Although each layer has a window of W tokens, information can propagate across longer distances through multiple layers. After k layers, a token can have an effective receptive field of k·W tokens. With 32 layers and W = 4096, the theoretical receptive field is 131,072 tokens — far beyond the nominal 8K context window.

**Rolling KV cache**: SWA enables a fixed-size KV cache of W key-value pairs per layer per head, regardless of sequence length. This is a major advantage for long-sequence inference: the KV cache memory cost is constant at O(W · H · d_k) rather than growing with sequence length. For very long sequences, this prevents GPU out-of-memory errors that would occur with a full attention KV cache.

**Flash Attention integration**: Mistral uses Flash Attention 2 for efficient implementation of SWA. The sliding window pattern maps naturally to Flash Attention's block-based computation, enabling hardware-efficient attention without custom CUDA kernels.

## Grouped-Query Attention (GQA)

Mistral 7B uses GQA with 32 query heads grouped into 8 key-value head groups (4 queries per KV head). This reduces the KV cache size by 4x compared to standard multi-head attention, with minimal quality degradation (GQA quality at this group ratio is empirically near-identical to full MHA for models in this size range).

Combined with the fixed-size rolling cache from SWA, GQA makes Mistral 7B's inference memory requirements very predictable and compact. During generation, the KV cache for a 4096-token context requires approximately:

```
2 (K and V) × 4096 (window) × 8 (KV heads) × 128 (head_dim) × 32 (layers) × 2 bytes (fp16)
≈ 536 MB
```

This compares favorably to LLaMA 2 7B's full attention KV cache at the same context length (~1.1 GB for 32 heads).

## Training

Mistral AI has not disclosed full details of their training data or procedure, but the model was trained on a large corpus of internet text with a focus on data quality. The tokenizer uses a byte-level BPE tokenizer with a 32K vocabulary (same as LLaMA). The model uses RoPE for position encoding with a base frequency of 10,000. No technical report on the training data mixture or compute budget has been published.

## Results

Mistral 7B outperforms LLaMA 2 13B on all reported benchmarks and outperforms LLaMA 1 34B on most:

- **HellaSwag**: 81.3% (vs. LLaMA 2 13B: 80.7%)
- **ARC Challenge**: 59.98% (vs. LLaMA 2 13B: 59.4%)
- **MMLU**: 64.16% (vs. LLaMA 2 13B: 55.77%)
- **HumanEval**: 30.5% (vs. LLaMA 2 13B: 18.3%)
- **GSM8K**: 52.2% (vs. LLaMA 2 13B: 28.7%)

Mistral-7B-Instruct (the instruction-tuned version using supervised fine-tuning and DPO) outperforms LLaMA 2 13B Chat on human preference evaluations.

## Mistral Mixture of Experts: Mixtral 8x7B

A follow-up model, Mixtral 8x7B, applies a Mixture of Experts architecture with 8 expert FFN layers per token position, of which 2 are active per forward pass. This achieves the inference cost of a ~12B model while having the parameter count of a 46.7B model. Mixtral 8x7B outperforms LLaMA 2 70B and approaches GPT-3.5 on several benchmarks, demonstrating that sparse expert models scale very favorably.

## Significance

Mistral 7B demonstrated that careful architectural choices — specifically SWA and GQA — can substantially improve the capability-to-compute ratio compared to a naive scaling of prior architectures. It set a new efficiency frontier for 7B-scale open models and its weights and architecture have become a popular starting point for fine-tuning, continued pretraining, and research. The Mistral codebase and model card established a template for responsible open model releases that subsequent labs (Qwen, Gemma, OLMo) followed.
