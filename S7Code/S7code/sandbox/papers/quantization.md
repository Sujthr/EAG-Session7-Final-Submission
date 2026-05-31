# Neural Network Quantization: GPTQ, AWQ, and GGUF

## Key Papers and Formats

- **GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers** — Frantar, Ashkboos, Hoefler, Alistarh — ETH Zurich, 2022 (ICLR 2023)
- **AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration** — Lin, Tang, Tang, Yang, Dang, Han — MIT, 2023
- **GGML/GGUF:** Community-developed quantization formats for efficient CPU/GPU inference (Georgi Gerganov, 2023)

---

## Why Quantization Matters

Modern LLMs contain billions of parameters stored in 32-bit or 16-bit floating point. A 70B parameter model in float16 requires ~140GB of GPU memory — far exceeding consumer hardware and even many professional GPUs. Quantization reduces the numerical precision of weights (and optionally activations) to use fewer bits, dramatically reducing memory footprint and inference cost.

The key challenge is doing this **without significant accuracy loss**: compressing a 16-bit float to 4 bits discards 75% of the information in the numerical representation, yet a well-designed quantization scheme can preserve model quality to within 1–2 perplexity points.

---

## GPTQ (2022) — Post-Training Quantization via Second-Order Information

### Problem Setup

Post-training quantization (PTQ) quantizes a trained model without any further gradient-based optimization. Prior PTQ methods for large models (ZeroQuant, LLM.int8()) struggled at 4-bit precision, showing unacceptable perplexity degradation. GPTQ achieves 4-bit (and even 3-bit) quantization with minimal quality loss by leveraging second-order information about weight importance.

### Method: Optimal Brain Quantization (OBQ) at Scale

GPTQ is based on the Optimal Brain Quantization framework (itself derived from the classic Optimal Brain Surgeon from the 1990s). The key idea:

1. Quantize weights layer by layer, processing each weight matrix independently
2. For each weight being quantized, compute the increase in layer output error caused by the quantization
3. **Compensate** for this error by adjusting the remaining (not-yet-quantized) weights in the same row, using the inverse Hessian of the layer's reconstruction loss

The Hessian captures second-order sensitivity: weights with high curvature (high second derivatives) have large impact when perturbed, and should be compensated more aggressively.

**Key algorithmic innovation:** Computing the full Hessian and its inverse is O(d³) for a d×d weight matrix — infeasible for large matrices. GPTQ uses a lazy batch update strategy and a Cholesky decomposition that reduces this to O(d²) and enables processing all 175B parameters of GPT-3 in approximately 4 GPU-hours.

### Results

- OPT-175B quantized to 4-bit with GPTQ: perplexity increase of ~0.5–1.0 points over float16
- BLOOM-176B: similar results
- 3-bit quantization achieves ~2–3 perplexity point degradation — acceptable for many use cases
- Inference speedup: 3–4x over float16 on A100 due to reduced memory bandwidth consumption
- The quantized models fit on a single A100 80GB GPU at 4-bit (vs. requiring 5 A100s at float16 for 175B parameter models)

---

## AWQ: Activation-Aware Weight Quantization (2023)

### Key Insight: Not All Weights Are Equal

AWQ observes that naive uniform quantization treats all weights as equally important. In practice, a small fraction of weights (~1%) have outsized influence on model outputs because they correspond to features with high activation magnitudes. Quantizing these "salient" weights to low precision causes disproportionate accuracy loss.

### Method

AWQ identifies salient weights by analyzing **activation statistics** on a small calibration dataset (not gradients — no backpropagation required). Specifically, weights are deemed salient if the corresponding input activation channels have large magnitudes across many samples.

Rather than keeping salient weights at higher precision (which complicates hardware implementation), AWQ uses a mathematically equivalent **per-channel scaling trick**:
- Scale up salient weight channels by a factor s before quantization (reducing relative quantization error)
- Scale down corresponding input activations by 1/s to preserve the layer's output

This scaling is absorbed into adjacent layers (LayerNorm scales, etc.) and has zero inference overhead. The optimal scaling factor s is found by minimizing the layer output reconstruction error on the calibration set.

**Result:** 4-bit AWQ models match or exceed GPTQ in quality while being simpler to implement and faster to apply (minutes vs. hours for calibration).

AWQ also enables efficient hardware kernels (AWQ-CUDA) that fuse dequantization with GEMM operations, achieving practical 3–4x speedups on consumer GPUs (RTX 3090/4090) where memory bandwidth is the bottleneck.

---

## GGML / GGUF — Community Quantization for Local Inference

### Background

GGML (Georgi Gerganov Machine Learning) is a C library for tensor operations designed for CPU inference. It was the foundation for `llama.cpp`, which enabled running LLaMA models on MacBooks and consumer CPUs. GGUF (GGML Unified Format) is the successor file format introduced in 2023 to address limitations of the original GGML format.

### Quantization Schemes in GGUF

GGUF supports a range of quantization levels, named by bits per weight:

- **Q8_0:** 8-bit quantization, ~8 bits/weight. Near-lossless, requires ~7.5GB for 7B models
- **Q4_K_M:** 4-bit quantization with k-quants (mixed precision). Attention and embedding layers may use slightly higher precision. ~4.5GB for 7B models
- **Q4_0 / Q4_1:** Simpler 4-bit schemes without mixed precision. Faster to run but slightly lower quality
- **Q2_K / Q3_K:** Aggressive quantization for severely memory-limited scenarios

**K-quants (introduced 2023):** A major improvement in GGUF, k-quants apply slightly higher precision (e.g., 6-bit) to layers identified as more sensitive (typically attention projections and output layers), while keeping less sensitive layers at lower precision. This asymmetric approach significantly improves quality at the same average bit rate.

### Hardware Targeting

GGUF models can run on:
- **CPU only:** Using BLAS-accelerated matrix operations (OpenBLAS, BLIS)
- **Metal (Apple Silicon):** Offloading layers to Apple GPU via Metal
- **CUDA/ROCm:** Partial or full GPU offload for Nvidia/AMD GPUs
- **Vulkan:** Cross-platform GPU backend

The ability to **split layers across CPU and GPU** is especially useful: a user with 8GB VRAM can offload 30 of 32 layers to the GPU and run the remaining 2 on CPU, achieving near-GPU speeds for a 13B model that wouldn't otherwise fit in VRAM.

---

## Comparative Summary

| Method | Precision | Quality Loss | Speed | Hardware |
|--------|-----------|-------------|-------|----------|
| GPTQ | 4-bit | Low (~1 PPL) | 3–4x vs fp16 | Nvidia GPU |
| AWQ | 4-bit | Very Low | 3–4x vs fp16 | Nvidia GPU |
| GGUF Q4_K_M | ~4.5 bit | Low | Variable | CPU/GPU |

---

## Significance

Quantization has democratized access to large language models. A 70B parameter model that required 5 A100 GPUs in float16 can run on two consumer RTX 4090s at 4-bit (GPTQ/AWQ) or on a CPU with adequate RAM (GGUF). This enabled the local LLM movement, privacy-preserving on-device inference, and substantially reduced the cost of LLM serving at scale — making billion-parameter models practical outside of large data center environments.
