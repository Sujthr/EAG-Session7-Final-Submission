# Self-Supervised Learning: From Masked Language Models to Contrastive Vision

**Key Works Covered:**
- **BERT** — Devlin et al., Google AI, 2018 (arXiv:1810.04805)
- **GPT** — Radford et al., OpenAI, 2018
- **SimCLR** — Chen et al., Google Brain, 2020 (arXiv:2002.05709)
- **MAE (Masked Autoencoders)** — He et al., Meta AI, 2021 (arXiv:2111.06377)

---

## What is Self-Supervised Learning?

Self-supervised learning (SSL) is a paradigm in which supervision signals are derived automatically from the structure of unlabeled data, rather than from human annotations. The model is trained on a pretext task — predicting masked tokens, the next frame in a video, or whether two image crops came from the same image — and the resulting representations transfer to downstream tasks. SSL bridges the gap between the vastness of unlabeled data and the scarcity of labeled data, and has become the dominant pretraining paradigm for both NLP and computer vision.

---

## BERT: Masked Language Modeling

**Bidirectional Encoder Representations from Transformers (BERT)** introduced two self-supervised pretext tasks for NLP:

### Masked Language Modeling (MLM)

15% of input tokens are randomly selected. Of these:
- 80% are replaced with a [MASK] token
- 10% are replaced with a random token
- 10% are left unchanged

The model must predict the original identity of the masked tokens. Unlike causal (left-to-right) language modeling used in GPT, MLM conditions on both left and right context simultaneously, producing bidirectional representations. This is critical for tasks like named entity recognition and question answering, where full-context understanding matters.

### Next Sentence Prediction (NSP)

The model is given two sentence segments and must predict whether the second follows the first in the original document. This was intended to train discourse-level understanding, though later work (RoBERTa, 2019) showed NSP provides marginal benefit and can be removed without hurting performance.

### Architecture and Impact

BERT uses a transformer encoder (not decoder). The pretrained [CLS] token representation is used for classification tasks; token representations are used for sequence labeling. BERT established the **pretrain-then-fine-tune** paradigm that dominated NLP from 2018–2022, achieving state-of-the-art on GLUE, SQuAD, and numerous other benchmarks. Its innovations — contextual embeddings, MLM pretraining, and transfer learning — became the foundation of virtually all subsequent NLP models.

---

## GPT: Autoregressive (Next-Token) Prediction

While BERT learns bidirectional representations, the original **GPT** (Generative Pre-trained Transformer) trains a left-to-right language model: given tokens t₁, t₂, ..., tₙ₋₁, predict tₙ. The pretext task is simply maximum likelihood estimation over the token sequence:

L = Σ log P(tᵢ | t₁, ..., tᵢ₋₁; θ)

This seemingly simple objective has profound consequences: a model that accurately predicts text must implicitly model syntax, semantics, world knowledge, and discourse structure. The autoregressive formulation naturally enables generation (sampling), which BERT's bidirectional architecture does not support. GPT-1 demonstrated that fine-tuning a pretrained autoregressive LM significantly outperformed task-specific models, setting the stage for GPT-2 and GPT-3's zero/few-shot capabilities.

The key architectural difference from BERT is **causal masking** in self-attention — each position can only attend to earlier positions, enforcing the autoregressive property.

---

## SimCLR: Contrastive Self-Supervised Learning for Vision

**A Simple Framework for Contrastive Learning of Visual Representations (SimCLR)** extended the SSL paradigm to computer vision using contrastive learning rather than reconstruction.

### Method

1. **Two views per image:** Each image is augmented twice using random cropping, color jitter, grayscale conversion, and Gaussian blur to create two "views" (xᵢ, xⱼ) of the same image.
2. **Shared encoder:** Both views are passed through the same ResNet encoder f(·) to produce representations hᵢ and hⱼ.
3. **Projection head:** A small MLP g(·) projects representations to a lower-dimensional space: zᵢ = g(hᵢ). This projection head is discarded after pretraining.
4. **NT-Xent loss (Normalized Temperature-scaled Cross Entropy):** For a mini-batch of N images (2N augmented views), the loss treats (zᵢ, zⱼ) as a positive pair and all other 2(N-1) views as negatives:

   L = -log [ exp(sim(zᵢ,zⱼ)/τ) / Σₖ≠ᵢ exp(sim(zᵢ,zₖ)/τ) ]

   where sim(u,v) = uᵀv/(‖u‖‖v‖) is cosine similarity and τ is a temperature parameter.

### Key Findings

- **Data augmentation composition is crucial:** Random cropping combined with color distortion is the most important augmentation pair. Without color distortion, the model learns shortcuts (background color correlations) rather than semantic structure.
- **Projection head matters:** Representations at the output of the projection head are worse for downstream tasks than representations at the encoder output. The projection head helps the encoder avoid collapsing to task-specific features during contrastive training.
- **Large batch size and training duration:** SimCLR requires large batches (4096–8192) for sufficient negatives and benefits from extended training (800–1000 epochs).

SimCLR matched supervised ResNet-50 performance on ImageNet linear evaluation with a ResNet-50 encoder, and closed much of the gap using larger encoders — a striking demonstration of SSL's potential for vision.

---

## MAE: Masked Autoencoders for Vision

**Masked Autoencoders (MAE)** revisited the MLM pretext task for vision using Vision Transformers (ViT), demonstrating that high masking ratios (75%) force richer representation learning than is possible with full-image or low-mask-rate inputs.

### Method

1. **Patch tokenization:** The image is divided into non-overlapping 16×16 patches, each linearly projected to an embedding vector.
2. **Random masking:** 75% of patches are masked (removed entirely from the encoder input). The encoder only processes the visible 25%.
3. **Asymmetric encoder-decoder:** A large ViT encoder processes visible patches with full self-attention. A small lightweight decoder (with mask tokens re-inserted) reconstructs the original pixel values of masked patches. The decoder is narrow and shallow compared to the encoder.
4. **Reconstruction target:** The model predicts the mean-normalized pixel values of each masked patch (no perceptual loss or VQ codebook — raw pixels suffice).

### Why 75% Masking?

Unlike text, where masking 15% of tokens is already a hard task (dense semantic information), image patches contain high spatial redundancy — a model can reconstruct a masked patch from its neighbors via simple interpolation without learning semantics. The 75% masking rate is high enough that reconstruction requires understanding the global structure of the image, not just local texture extrapolation.

### Results

MAE pretrained ViT-Large achieves 87.8% top-1 ImageNet accuracy with fine-tuning, outperforming prior SSL methods and supervised baselines. Critically, MAE training is faster than contrastive methods (no large batches needed) and simpler (no data augmentation engineering, negative pair sampling, or momentum encoders). MAE established that reconstruction-based SSL can match and exceed contrastive SSL in vision, unifying the pretraining philosophy across modalities.

---

## Unified Perspective

All four approaches share a common structure: define a pretext task from unlabeled data, train a model to solve it, and use the resulting representations for downstream tasks. The specific tasks differ — reconstruction (BERT, MAE), next-token prediction (GPT), or instance discrimination (SimCLR) — but the central insight is the same: structure inherent in data (token co-occurrence, spatial coherence, semantic invariance to augmentation) provides sufficient signal to learn general-purpose representations without labels.
