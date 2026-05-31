# Gemma: Open Models Based on Gemini Research and Technology

**Authors:** Gemma Team, Google DeepMind  
**Institution:** Google DeepMind  
**Year:** 2024  
**Paper:** arXiv:2403.08295

---

## Overview

Gemma is a family of lightweight open-weight language models released by Google DeepMind in February 2024. The initial release includes two sizes — **Gemma 2B** and **Gemma 7B** — along with instruction-tuned variants of each. Gemma is derived from the same research and infrastructure used to train Gemini, Google's flagship multimodal model family, but targeted at open deployment: the weights are publicly available for research and commercial use under Google's custom Gemma Terms of Use.

The release of Gemma marked Google DeepMind's serious entry into the competitive open-weight model space (alongside Meta's LLaMA, Mistral AI's Mistral, and others), with a specific focus on safety, responsibility, and deployment on consumer hardware.

---

## Key Ideas and Design Philosophy

**Gemini-derived architecture and data:** Gemma is trained using many of the techniques developed for Gemini, including the same tokenizer (SentencePiece BPE with 256,128-token vocabulary — significantly larger than LLaMA's 32,000). The training data is primarily web text, mathematics, and code from the same distribution used for Gemini pretraining, though the exact dataset is not fully disclosed.

**Size-performance efficiency:** Gemma 7B was designed to be competitive with models significantly larger than itself. The Gemma team emphasizes that training on the right data mix and for sufficient token counts — following Chinchilla scaling law insights — matters more than raw parameter count. Gemma 7B outperforms LLaMA 2 7B and 13B on most benchmarks, and is competitive with models up to 34B parameters on certain tasks.

**Responsible AI focus:** Each Gemma release includes a detailed model card covering known limitations, demographic parity evaluations, and safety benchmark performance. Instruction-tuned Gemma models undergo reinforcement learning from human feedback (RLHF) with specific attention to reducing harmful outputs.

---

## Architecture Details

Gemma largely follows the transformer decoder architecture established by LLaMA, with several distinct choices:

**Multi-Head Attention (MHA) vs. Multi-Query Attention (MQA):**
- Gemma 2B uses **Multi-Query Attention** (MQA), where all attention heads share a single key-value head. This reduces the KV-cache memory footprint significantly, critical for efficient inference on consumer hardware.
- Gemma 7B uses **Multi-Head Attention** (MHA), the standard formulation with separate K and V projections per head.

**RMSNorm:** Pre-normalization using Root Mean Square Layer Normalization, consistent with LLaMA. RMSNorm applies before each attention and feed-forward sub-layer.

**GeGLU Activations:** Gemma uses the GeGLU (Gated Exponential Linear Unit) variant in feed-forward layers: `FFN(x) = (xW₁ ⊙ sigmoid(xW₂)) × W₃`. This is similar to SwiGLU (used in LLaMA) and has been shown empirically to produce better language model perplexity than standard GELU or ReLU.

**Rotary Positional Embeddings (RoPE):** Gemma encodes positional information using RoPE applied to query and key vectors at each attention layer, consistent with the broader trend in modern LLMs.

**Context Length:** Both Gemma 2B and 7B support context lengths of 8,192 tokens — double the original LLaMA's 4,096 — which enables handling longer documents and more complex multi-turn conversations.

**Vocabulary:** The 256K-token vocabulary is one of Gemma's distinguishing features. The large vocabulary provides better tokenization efficiency for multilingual text, code, and mathematical notation. For comparison, LLaMA uses 32K tokens and GPT-4 uses ~100K tokens (tiktoken cl100k_base).

**No bias terms:** Linear layers use no bias terms, consistent with modern LLM practice.

---

## Training

**Pretraining data:** Gemma is trained primarily on English web documents, with mathematical text (selected from web and curated math datasets) and code (from public repositories). The exact token counts are not fully disclosed, but Gemma 7B is trained on approximately 6 trillion tokens and Gemma 2B on approximately 2-3 trillion tokens.

**Training infrastructure:** Trained on Google TPUv5e clusters using JAX and the MaxText infrastructure. The use of JAX/XLA allows aggressive compiler optimization and efficient memory use across thousands of chips.

**Instruction tuning (Gemma-IT):** Instruction-tuned variants are trained with a mixture of supervised fine-tuning on curated conversational datasets and RLHF using human preference data. The instruction format uses structured tags (`<start_of_turn>user\n`, `<start_of_turn>model\n`, `<end_of_turn>`) to delineate conversational turns.

---

## Results

Gemma was evaluated against contemporary open-weight models on a standard suite of benchmarks:

| Benchmark | Gemma 7B | LLaMA 2 13B | Mistral 7B |
|---|---|---|---|
| MMLU | 64.3 | 54.8 | 62.5 |
| HellaSwag | 81.2 | 80.7 | 81.3 |
| ARC-C | 53.2 | 49.4 | 55.5 |
| GSM8K | 46.4 | 28.7 | 35.4 |
| HumanEval | 32.3 | 18.3 | 26.2 |
| BBH | 55.1 | 45.6 | 56.7 |

Gemma 7B demonstrates particularly strong performance on mathematical reasoning (GSM8K) and code generation (HumanEval) relative to comparably-sized models, reflecting the quality of its training data mix.

Gemma 2B, despite its small size, outperforms several 7B models from 2023 on key reasoning benchmarks — making it practical for edge deployment.

---

## Gemma 2 (June 2024)

A follow-up release — **Gemma 2** — introduced 9B and 27B parameter models with architectural improvements including **sliding window attention** alternating with full attention, **logit soft-capping** to prevent logit explosion, and **knowledge distillation** from larger Gemini models. Gemma 2 9B competitive with models 2–3× its size, and 27B approaches Llama 3 70B performance.

---

## Significance

Gemma represents Google DeepMind's strategy of building open-weight models that serve as research platforms and developer tools while maintaining safety standards. The large vocabulary, long context, and instruction-tuning quality make Gemma models strong baselines for fine-tuning research. The responsible release practices — including detailed model cards, safety evaluations, and usage terms that permit commercial use — set a standard for responsible open-weight releases. Gemma's strong performance at the 2B scale in particular has made it a popular choice for on-device inference and edge AI applications, where VRAM and compute are severely constrained.
