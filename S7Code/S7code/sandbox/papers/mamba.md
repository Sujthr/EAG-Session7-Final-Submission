# Mamba: Linear-Time Sequence Modeling with Selective State Spaces

**Authors:** Albert Gu, Tri Dao
**Institutions:** Carnegie Mellon University, Together AI
**Year:** 2023

---

## Overview

Mamba is a sequence modeling architecture that achieves linear-time complexity in sequence length, directly challenging the quadratic complexity of standard Transformer self-attention. It is built on State Space Models (SSMs), specifically extending the S4 (Structured State Space) model with a novel **selection mechanism** that allows the model to selectively retain or ignore information based on the input content. Mamba achieves Transformer-quality performance on language modeling while being significantly faster at inference, particularly for long sequences.

## Background: State Space Models

State Space Models are a class of sequence-to-sequence maps derived from continuous-time dynamical systems. A linear SSM can be described by:

```
h'(t) = Ah(t) + Bx(t)
y(t)  = Ch(t) + Dx(t)
```

where `x(t)` is the input, `h(t)` is a latent state, and `y(t)` is the output. The matrices A, B, C, D govern the dynamics. When discretized for sequences, this becomes a recurrence that can also be computed efficiently in parallel using convolutions. Prior SSMs like S4 used fixed (input-independent) A, B, C matrices, which limited their ability to focus on relevant tokens selectively.

## The Core Innovation: Selective State Spaces

The fundamental problem with prior SSMs is that their transition matrices are **time-invariant** — they apply the same dynamics to every token regardless of content. This means the model cannot selectively remember or forget based on what it reads, a capability that attention mechanisms handle naturally.

Mamba introduces **input-dependent parameterization**: the matrices B, C, and the discretization step size Δ are now functions of the input `x`. Specifically:

- B and C are computed via linear projections of the current input.
- Δ (delta) controls how much the model focuses on the current input vs. prior state; larger Δ means the model "focuses" more on the new input.

This selectivity allows Mamba to behave like a content-aware filter — it can flush irrelevant context and retain only task-relevant information, mimicking the selective attention of Transformers without the quadratic cost.

## Hardware-Aware Algorithm

A key engineering contribution is Mamba's **hardware-aware parallel scan algorithm**. Naively, input-dependent SSMs cannot use the FFT-based convolution trick that made S4 fast, because the convolution kernel changes with each input. Mamba addresses this with a custom CUDA kernel that:

1. Performs the recurrence in **SRAM (on-chip memory)** rather than HBM (off-chip GPU RAM), avoiding memory bandwidth bottlenecks.
2. Uses a **parallel prefix scan** (similar to parallel cumulative sum) to compute the recurrence efficiently across sequence positions.
3. Fuses operations to minimize memory reads/writes.

This makes Mamba faster than attention in practice starting at sequence lengths of around 2K tokens, with the gap widening dramatically for longer sequences.

## Architecture

The Mamba model uses a simple block structure that replaces the Transformer's attention + MLP sublayers. Each Mamba block contains:

- An input projection that expands the dimension (similar to SwiGLU gating).
- A 1D depthwise convolution (short kernel, captures local patterns).
- The selective SSM layer.
- An output projection.

There is no separate MLP block — the SSM and gating together serve both roles. Layer normalization and residual connections are used as in standard Transformers.

## Results

On the Pile language modeling benchmark, Mamba-3B outperforms Transformer models of equivalent parameter count (such as GPT-3-style models) at all evaluated sequence lengths. It matches or exceeds performance of Transformer++ (an optimized Transformer baseline with rotary embeddings and SwiGLU) while being **5x faster** at inference for sequences of length 16K.

On downstream tasks (zero-shot and few-shot), Mamba is competitive with same-size Transformers on standard NLP benchmarks including HellaSwag, LAMBADA, ARC, and WinoGrande. On long-range tasks from the Long Range Arena, Mamba also performs well.

Mamba generates text at **~5x the throughput** of a comparable Transformer at sequence length 2048 on a single A100 GPU, due to its O(N) recurrent inference (no KV cache growth).

## Significance

Mamba represents one of the most credible alternatives to Transformers for sequence modeling since the original attention paper. Its success suggests that the quadratic attention mechanism is not strictly necessary for high-quality language modeling. Subsequent work has combined Mamba with attention in hybrid architectures (e.g., Jamba, Zamba), finding complementary strengths. Mamba's approach has also been applied to vision (Vision Mamba), genomics (Caduceus), and audio modeling. The selective state space concept has opened a new research direction exploring what forms of "memory" and "selectivity" are truly necessary for capable sequence models.
