# FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness

**Authors:** Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Re (Stanford University)
**Year:** 2022 (NeurIPS 2022)
**Follow-up:** FlashAttention-2 (Dao, 2023); FlashAttention-3 (Shah et al., 2024)

---

## Overview

FlashAttention is an exact attention algorithm that dramatically reduces the memory footprint and wall-clock time of transformer attention by restructuring the computation to minimize slow reads and writes to GPU high-bandwidth memory (HBM). Unlike approximate attention methods that sacrifice accuracy for efficiency, FlashAttention computes standard softmax attention exactly while achieving 2-4x speedups and O(N) memory scaling in sequence length — enabling training on sequences 5-20x longer than previously practical.

## The Memory Bottleneck in Standard Attention

Standard multi-head self-attention for a sequence of length N with model dimension d requires:

1. Compute query, key, value matrices: Q, K, V ∈ R^(N×d) — O(Nd) memory
2. Compute attention score matrix S = QK^T ∈ R^(N×N) — **O(N²) memory**
3. Apply softmax: P = softmax(S) — O(N²) memory
4. Compute output: O = PV ∈ R^(N×d) — O(Nd) memory

The N×N attention matrix is the critical bottleneck. For N=4096, this matrix alone requires 64GB at float32 — far exceeding GPU SRAM capacity. Every training step requires writing this matrix to HBM and reading it back multiple times.

## Key Insight: IO-Awareness

Modern GPU memory is hierarchical:
- **SRAM (on-chip):** ~20MB, ~19TB/s bandwidth — very fast, very small
- **HBM (GPU DRAM):** ~40-80GB, ~2TB/s bandwidth — much slower, much larger

GPU compute throughput has scaled far faster than HBM bandwidth, creating a bandwidth-bound regime where attention is limited not by computation but by the number of HBM reads/writes (IO operations). Standard attention performs O(N²) HBM accesses; FlashAttention reduces this to O(N²/M) where M is SRAM size.

## Algorithm: Tiled Computation with Online Softmax

FlashAttention fuses the attention computation into a single GPU kernel using two key techniques:

### Tiling
Rather than computing the full N×N attention matrix at once, FlashAttention splits Q, K, V into blocks that fit in SRAM. It processes one block at a time, loading each block from HBM to SRAM, computing attention locally, and accumulating the output — never materializing the full attention matrix in HBM.

### Online Softmax (Numerically Stable)
Standard softmax requires two passes over the full sequence (one for the max, one for the exponentials and normalization). FlashAttention uses the **online normalization trick** (Milakov & Giammichele, 2018) to compute softmax incrementally as new blocks arrive:

For each new block, maintain running statistics:
- `m_i`: running maximum of attention logits seen so far
- `l_i`: running sum of exponentials (normalization factor)
- `O_i`: running weighted output accumulation

When a new block arrives with maximum logit `m_new`:
```
m_updated = max(m_i, m_new)
l_updated = exp(m_i - m_updated) * l_i + exp(m_new - m_updated) * l_new
O_updated = (exp(m_i - m_updated) * l_i * O_i + exp(m_new - m_updated) * O_new) / l_updated
```

This allows exact softmax computation without ever storing the full attention matrix.

### Recomputation in Backward Pass
Rather than storing the N×N attention matrix for backpropagation (which would negate memory savings), FlashAttention recomputes attention during the backward pass from the stored Q, K, V blocks and the output/softmax statistics. This trades minimal extra computation for dramatic memory savings.

## Complexity Analysis

| Method | HBM Reads/Writes | Memory |
|--------|-----------------|--------|
| Standard Attention | O(N²d) | O(N²) |
| FlashAttention | O(N²d²/M) | O(N) |

For typical values (N=4096, d=64, M=100KB), FlashAttention achieves ~9x fewer HBM accesses.

## Results

Benchmarked on A100 GPU (80GB HBM):

- **Speed:** 2-4x faster than standard PyTorch attention for sequence lengths 512-4096; speedup grows with sequence length
- **Memory:** 5-20x memory reduction, enabling sequences up to 64k tokens on a single A100
- **Exact:** Output is mathematically identical to standard attention (verified empirically)
- **GPT-2 training (1.7B tokens):** 3x end-to-end speedup over Hugging Face baseline
- **Long-range Arena benchmark:** Enables models that previously ran OOM on 4K sequences
- **BERT pre-training:** 15% end-to-end speedup on MLPerf training benchmark

## FlashAttention-2 (2023)

FlashAttention-2 improved work partitioning across thread blocks and warps, achieving:
- Additional 2x speedup over FlashAttention-1
- Better utilization of tensor cores
- Support for head dimensions up to 256
- Up to 73% of A100 theoretical FLOPs utilization

## Significance

FlashAttention has been adopted universally across the frontier model ecosystem:

1. **Enabled long-context models:** 32K, 128K, and 1M+ context windows became practical
2. **Adopted by major frameworks:** Integrated into PyTorch (as `F.scaled_dot_product_attention`), HuggingFace, Megatron-LM, JAX/XLA
3. **Training efficiency:** Estimated to save millions of dollars in training costs across the field
4. **Inspired IO-aware algorithm design:** Demonstrated that hardware-aware algorithm design is as important as architectural innovation

FlashAttention is a landmark example of systems research that had immediate and large-scale practical impact on deep learning — enabling the long-context language models that power modern AI applications without sacrificing any model quality.
