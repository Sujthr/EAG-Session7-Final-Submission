# Fast Inference from Transformers via Speculative Decoding

## Paper Details

- **Title:** Fast Inference from Transformers via Speculative Decoding
- **Authors:** Yaniv Leviathan, Matan Kalman, Yossi Matias
- **Affiliation:** Google
- **Year:** 2022
- **Venue:** ICML 2023

---

## The Inference Bottleneck

Autoregressive language models generate text one token at a time. Each token generation requires a full forward pass through the model — for a 70B parameter model, this is an expensive matrix multiplication cascade. The key inefficiency is that this process is fundamentally **memory-bandwidth bound** rather than compute-bound: the GPU spends most of its time loading model weights from HBM (high-bandwidth memory) into compute units, not in the arithmetic operations themselves.

As a result, generating a single token and generating a batch of tokens require similar wall-clock time (up to the memory capacity limit). The sequential dependency — each token depends on all previous tokens — prevents parallelism across the sequence dimension.

**Speculative decoding** exploits this observation: if we could somehow generate and verify multiple tokens in a single forward pass of the large model, we could dramatically improve throughput without changing the model or degrading output quality.

---

## The Core Idea: Draft Then Verify

Speculative decoding uses two models:

1. **Draft model (small/fast):** A smaller, faster model that generates a sequence of K candidate tokens autoregressively. This is cheap — the draft model is typically 7–10x smaller than the target model.

2. **Target model (large/slow):** The original large model that defines the desired token distribution. It is used to verify the draft tokens.

The key algorithmic insight is that **the large model can score all K draft tokens in a single forward pass**, because the attention mechanism can process all K positions in parallel (given that all draft tokens are already available as context). This single forward pass is nearly as fast as generating one token, due to memory-bandwidth bottleneck behavior.

---

## Algorithm Details

**Notation:** Let p(x | context) be the target model's token probability distribution, q(x | context) be the draft model's distribution, and let x̃₁, x̃₂, ..., x̃ₖ be the K draft tokens generated autoregressively by the draft model.

**Step 1 — Draft generation:**
Run the draft model autoregressively to produce K tokens: x̃₁ ~ q(·|ctx), x̃₂ ~ q(·|ctx, x̃₁), ..., x̃ₖ ~ q(·|ctx, x̃₁,...,x̃ₖ₋₁).

**Step 2 — Parallel scoring:**
Run the target model on the full sequence [ctx, x̃₁, ..., x̃ₖ] in a single forward pass. This yields target probabilities p(·|ctx), p(·|ctx, x̃₁), ..., p(·|ctx, x̃₁,...,x̃ₖ) — one distribution per position.

**Step 3 — Token acceptance/rejection:**
For each draft token x̃ᵢ, apply a rejection sampling criterion:
- Accept x̃ᵢ with probability min(1, p(x̃ᵢ|·) / q(x̃ᵢ|·))
- If accepted, move to x̃ᵢ₊₁
- If rejected, sample a correction token from an adjusted distribution: (p(·) - q(·))₊ / Z (the positive part, normalized), then stop

**Step 4 — Output:**
All accepted tokens plus one correction token are appended to the context. If all K tokens are accepted, also sample one additional token from the target's distribution at position K+1.

**Mathematical guarantee:** The accepted tokens follow exactly the target model's distribution p — not an approximation. This is proven via the correctness of the rejection sampling scheme. Speculative decoding produces **identical output distributions** to running the target model alone, just faster.

---

## Efficiency Analysis

**Expected tokens per step:** If the draft model is well-aligned with the target (high acceptance rate α), each speculative step accepts ~K·α + 1 tokens while performing the equivalent compute of one target model step plus K draft model steps. Since the draft model is much cheaper, the effective speedup is:

Speedup ≈ (K·α + 1) / (1 + K·cost_draft/cost_target)

For a 7B draft model used with a 70B target model (10:1 cost ratio), with α=0.8 and K=5:
- Tokens per step: 5 × 0.8 + 1 = 5
- Relative cost: 1 + 5 × 0.1 = 1.5
- Speedup: 5/1.5 ≈ 3.3x

**Empirical results:** The paper demonstrated 2–3x wall-clock speedup on T5-XXL (11B parameters) with a T5-Small draft model, with no change in output quality on translation and summarization benchmarks.

---

## Practical Considerations

**Draft model selection:** The draft model must be "compatible" with the target model — producing similar distributions so that acceptance rates are high. Common approaches:
- Use a smaller model from the same family (e.g., LLaMA-7B as draft for LLaMA-70B)
- Train a purpose-built draft model from the target via distillation
- Use earlier layers of the target model (Medusa, EAGLE variants)

**Hardware implications:** Speculative decoding is most beneficial in memory-bandwidth-bound regimes (single-user inference on large models). In high-throughput batch serving, continuous batching may already saturate compute units, reducing the benefit.

**Self-speculative variants:** Subsequent work (Medusa, EAGLE) avoids the separate draft model by adding lightweight draft heads to the target model itself, further reducing complexity.

---

## Significance

Speculative decoding was one of the first inference optimization techniques to offer **lossless** speedups for autoregressive generation — meaning no degradation in output quality whatsoever. This distinguishes it from quantization (small quality loss) or pruning (variable quality loss).

It has been adopted in production inference systems at Google, Meta, and Hugging Face, and is integrated into frameworks like vLLM, TensorRT-LLM, and Transformers. The technique elegantly turns the parallel scoring capability of transformers (which was previously only exploited during training) into an inference-time advantage, addressing the core bottleneck of sequential autoregressive generation.
