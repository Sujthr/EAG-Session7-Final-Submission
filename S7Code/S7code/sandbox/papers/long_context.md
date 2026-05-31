# Long-Context Transformers: Longformer, BigBird, RoPE, ALiBi, and Positional Encoding for Long Sequences

**Key Papers:**
- *Longformer: The Long-Document Transformer* — Beltagy et al., Allen Institute for AI, 2020
- *Big Bird: Transformers for Longer Sequences* — Zaheer et al., Google Research, 2020
- *RoFormer: Enhanced Transformer with Rotary Position Embedding* — Su et al., 2021
- *Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation* — Press et al., Meta AI, 2021 (ALiBi)

---

## Overview

Standard Transformer self-attention has O(N²) complexity in sequence length N — both in computation and memory — which makes it prohibitive for sequences beyond roughly 1024–4096 tokens at training time. Long-context modeling is a central challenge in NLP, required for tasks like document summarization, multi-hop reasoning, code understanding, and long-form generation. Researchers have pursued two broad strategies: **sparse attention patterns** that reduce the quadratic cost, and **improved positional encodings** that allow models to extrapolate to lengths longer than those seen during training.

## Longformer

Longformer (Beltagy et al., 2020) replaces full self-attention with a combination of **local windowed attention** and **global attention** on a small number of selected tokens.

**Local windowed attention**: Each token attends only to a window of w tokens to its left and right. This is O(N·w) rather than O(N²), and windows can be dilated (like dilated convolutions) to increase effective receptive field without increasing cost.

**Global attention**: Certain task-specific tokens (e.g., the [CLS] token for classification, question tokens for QA) attend to all other tokens and are attended to by all other tokens. This allows information to flow across the entire sequence while keeping most attention local.

Longformer was pretrained on long documents (up to 4096 tokens) starting from RoBERTa weights, using gradient checkpointing and a custom CUDA kernel for efficient windowed attention. On tasks like WikiHop, TriviaQA, and long document summarization, Longformer substantially outperformed models that truncated inputs to 512 tokens.

## BigBird

BigBird (Zaheer et al., 2020) proves theoretically that sparse attention with three components can be a universal approximator of full attention: **(1) random attention** (each token attends to r random tokens), **(2) local windowed attention** (w tokens on each side), and **(3) global tokens** (g tokens that attend to everything). This is a O(N) attention mechanism.

The theoretical contribution is significant: BigBird proves that any function computable by a full Transformer is also computable by a sparse Transformer with this structure, under mild conditions. In practice, BigBird extends BERT/T5-style models to 4096 tokens and achieves strong results on genomics tasks (long DNA sequences), document QA, and summarization. BigBird-Pegasus combines this sparse attention with the Pegasus pretraining objective for summarization.

## Rotary Position Embedding (RoPE)

RoPE (Su et al., 2021) is a **relative position encoding** that encodes position information by rotating query and key vectors in embedding space. Rather than adding position embeddings to token embeddings (absolute position encoding as in the original Transformer), RoPE modifies the attention score computation directly.

For a query vector q at position m and key vector k at position n, RoPE ensures the dot product `q^T · k` depends only on the content of q and k and on their **relative position** (m - n). This is achieved by multiplying q and k by rotation matrices parameterized by the positions.

Key properties of RoPE:
- **Relative position dependence**: The model naturally handles relative distances without learning explicit position embeddings.
- **Extrapolation**: RoPE enables some length generalization, though it degrades for positions much larger than those seen in training.
- **Efficiency**: No additional parameters; computation is fused into the attention QK operation.
- **Widespread adoption**: RoPE is used in LLaMA, Mistral, Falcon, Qwen, and most modern open-source LLMs.

Extensions like **YaRN** (Yet another RoPE extension) and **LongRoPE** apply scaling and interpolation tricks to RoPE to extend context length significantly (e.g., from 4K to 128K tokens) with minimal additional fine-tuning.

## ALiBi (Attention with Linear Biases)

ALiBi (Press et al., 2021) takes a different approach: instead of modifying position embeddings, it **adds a position-dependent bias to attention logits** before the softmax. The bias is a linear penalty proportional to the distance between query and key positions:

```
Attention score(q_i, k_j) = q_i · k_j - m · |i - j|
```

where m is a head-specific slope (different for each attention head). Closer tokens get smaller penalties; distant tokens get larger penalties that discourage attending to them.

**Extrapolation advantage**: ALiBi's key insight is that the linear penalty naturally extrapolates to sequence lengths not seen during training. A model trained on 1024 tokens can still process 2048 tokens coherently, because the bias function extends smoothly. Empirically, ALiBi models lose less perplexity when tested on longer sequences than models using sinusoidal or learned absolute position embeddings.

ALiBi is used in BLOOM and MPT, and influenced subsequent positional encoding designs.

## Comparison and Significance

| Method | Approach | Complexity | Extrapolation | Notable Users |
|--------|----------|-----------|--------------|--------------|
| Longformer | Sparse attention | O(N·w) | Limited | AllenAI models |
| BigBird | Sparse + random + global | O(N) | Limited | Google genomics |
| RoPE | Relative rotation | O(N²) standard | Moderate | LLaMA, Mistral |
| ALiBi | Logit bias | O(N²) standard | Good | BLOOM, MPT |

The field has largely converged on RoPE with interpolation/scaling techniques as the dominant approach for modern long-context LLMs, with sparse attention being used selectively for extreme lengths. The challenge of reliable extrapolation — handling sequences longer than those seen in training — remains an active research area.
