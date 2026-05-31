# Mixture of Experts in Large Language Models

## Overview and Key Papers

- **Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer** — Shazeer et al., Google Brain, 2017
- **Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity** — Fedus, Zoph, Shazeer, Google, 2022
- **Mixtral of Experts** — Mistral AI, 2023

---

## Core Idea

Mixture of Experts (MoE) is a conditional computation paradigm where only a subset of model parameters is activated for any given input token. Instead of routing every token through every parameter of the network, a learned gating mechanism selects a small number of "expert" subnetworks to process each token. This allows models to scale total parameter counts dramatically without a proportional increase in compute per forward pass.

The key insight is that **parameter count and compute are decoupled**. A model with 100 billion parameters can require roughly the same FLOPs per token as a dense 10-billion parameter model if only 10% of experts are active at any time.

---

## Shazeer et al. 2017 — The Foundational Work

The 2017 paper introduced the Sparsely-Gated MoE layer and demonstrated it at scale in the context of recurrent language models. The architecture replaced certain feed-forward layers with an MoE layer consisting of up to thousands of expert networks.

**Gating mechanism:** A trainable gating network takes the token representation as input and outputs a sparse probability distribution over experts. Only the top-k experts (typically k=1 or k=2) receive nonzero weight. The output of the MoE layer is the weighted sum of selected expert outputs.

**Key challenges addressed:**
- **Load balancing:** Without explicit regularization, the gating network collapses to always selecting the same few experts. Shazeer introduced an auxiliary load-balancing loss to encourage uniform expert utilization.
- **Communication overhead:** In distributed training, experts may reside on different devices, requiring all-to-all communication. The paper established the token-routing infrastructure that all subsequent MoE work builds on.

Results showed 1000x improvements in model capacity with only a 2–4x increase in compute, achieving lower perplexity than dense baselines on language modeling benchmarks.

---

## Switch Transformer (2022) — Simplification and Trillion-Scale

The Switch Transformer paper from Google simplified Shazeer's design by routing each token to exactly **one expert** (k=1, called "switching"), rather than combining outputs from multiple experts. This simplified the routing logic and reduced communication costs significantly.

**Architecture:** MoE layers replace every other feed-forward network (FFN) block in a standard Transformer. The switch router is a single linear projection followed by a softmax, producing a routing probability. Each token is dispatched to the single highest-probability expert.

**Load balancing loss:** A differentiable auxiliary loss is added to encourage balanced expert load. The loss penalizes configurations where some experts receive many tokens while others receive none — a phenomenon called "expert collapse."

**Expert capacity:** Each expert has a fixed capacity buffer (capacity factor × tokens_per_expert). Tokens routed to a full expert are dropped, which introduces some information loss but keeps memory usage bounded.

**Results:**
- Achieved 7x faster pre-training than T5-XXL at comparable quality
- Scaled to 1.6 trillion parameters while maintaining tractable training
- Demonstrated strong multilingual performance with 101-language models

**Training stability** was a known challenge: MoE models showed higher instability than dense models, and the paper explored selective precision (bfloat16 for routing, float32 for expert computation) as a mitigation.

---

## Mixtral 8x7B (2023) — MoE Comes to Open-Source LLMs

Mistral AI released Mixtral 8x7B, the first widely adopted open-weight sparse MoE model competitive with the best dense models of similar compute cost. Despite having 46.7B total parameters, Mixtral uses only ~12.9B parameters per token (two experts active per token out of eight).

**Architecture:** Mixtral follows the standard LLaMA-style decoder-only Transformer, but replaces each FFN block with 8 expert FFNs and a router. The router selects the top-2 experts per token, and outputs are combined as a weighted sum.

**Performance:** Mixtral 8x7B matched or exceeded LLaMA 2 70B and GPT-3.5 on most benchmarks including MMLU, HumanEval, and GSM8K, while requiring only a fraction of the active compute. On math and code tasks it showed particularly strong performance.

**Inference efficiency:** Because only 2 of 8 experts are active per token, inference can be parallelized across GPUs by assigning experts to different devices, or run efficiently on consumer hardware using quantized GGUF formats.

---

## Significance and Impact

MoE architectures represent one of the most practical paths to scaling model capability without proportionally scaling inference cost. The architectural pattern has been adopted in GPT-4 (reportedly), Gemini 1.5, and numerous open models. The key trade-off — more total parameters, same active compute — directly addresses the fundamental bottleneck of large-scale AI deployment: inference cost.

Ongoing research focuses on expert routing strategies, reducing token dropping, improving load balancing, and applying MoE to other modalities including vision and multimodal systems.
