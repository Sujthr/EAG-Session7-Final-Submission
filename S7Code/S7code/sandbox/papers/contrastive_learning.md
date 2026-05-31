# Contrastive Learning: SimCLR, MoCo, ANCE, and the InfoNCE Loss

**Key Papers:**
- *A Simple Framework for Contrastive Learning of Visual Representations* — Chen et al., Google Research, 2020 (SimCLR)
- *Momentum Contrast for Unsupervised Visual Representation Learning* — He et al., FAIR, 2020 (MoCo)
- *Approximate Nearest Neighbor Negative Contrastive Estimation for Dense Text Retrieval* — Xiong et al., Microsoft, 2020 (ANCE)

---

## Overview

Contrastive learning is a self-supervised representation learning paradigm in which a model is trained to bring similar (positive) pairs of examples closer together in embedding space while pushing dissimilar (negative) pairs apart. It has become one of the dominant frameworks for learning representations without labeled data in both vision and language domains, and is the foundation of dense retrieval systems used in modern RAG pipelines.

## The InfoNCE Loss

The **InfoNCE loss** (Information Noise-Contrastive Estimation), introduced by van den Oord et al. (2018), is the mathematical backbone of most contrastive methods. Given a query embedding `q` and a set of keys `{k_0, k_1, ..., k_N}` where `k_0` is the positive key and the rest are negatives, the loss is:

```
L = -log( exp(q · k_0 / τ) / Σ_i exp(q · k_i / τ) )
```

where τ (tau) is a temperature hyperparameter. This is a cross-entropy loss over a softmax distribution — the model is trained to identify the positive key from a set of N+1 candidates. Lower temperature τ makes the distribution sharper and forces more discriminative representations, but can cause training instability if set too low.

The InfoNCE loss has a principled information-theoretic interpretation: minimizing it is equivalent to maximizing a lower bound on the mutual information between the query and its positive key.

## SimCLR

SimCLR (Simple Contrastive Learning of Representations) from Chen et al. at Google Brain is a clean, minimal framework that achieves strong visual representations through data augmentation. For each image in a batch, two random augmented views are generated (the positive pair). All other augmented views in the batch serve as negatives — this is called **in-batch negatives**.

Key components:
- **Augmentation pipeline**: Random cropping, color jitter, Gaussian blur, grayscale — the composition of random augmentations defines what "similar" means.
- **Projection head**: A two-layer MLP applied to ResNet representations before computing the contrastive loss. The projection head is discarded after training; the frozen backbone is used for downstream tasks. This was a non-obvious but critical finding: the head improves training but the raw backbone representations are better for transfer.
- **Large batch sizes**: SimCLR requires very large batches (4096–8192) to have enough in-batch negatives for effective training. This is computationally expensive.

SimCLR v2 scaled the method to larger encoders (ResNet-152, semi-supervised fine-tuning) and achieved near-supervised performance on ImageNet.

## MoCo (Momentum Contrast)

MoCo addresses SimCLR's need for large batches by maintaining a **dynamic queue** of negative keys from previous batches, decoupling the number of negatives from the batch size. A **momentum encoder** (a slowly-updated copy of the main encoder with exponential moving average: `θ_k = m·θ_k + (1-m)·θ_q`) encodes keys to ensure consistency across the queue despite the encoder updating.

This allows MoCo to maintain a large, consistent set of negatives (e.g., 65,536) with small batch sizes, making it more memory-efficient than SimCLR. MoCo v2 and v3 further refined the method by adopting SimCLR's projection head and augmentations, and extending to ViT backbones.

## ANCE: Contrastive Learning for Dense Retrieval

ANCE (Approximate Nearest Neighbor Negative Contrastive Estimation) adapts contrastive learning for text retrieval, where the goal is to learn bi-encoder embeddings such that a query vector is close to relevant passage vectors. The key challenge is **hard negative mining**: random negatives are trivially easy to distinguish from the positive, yielding little training signal.

ANCE addresses this by periodically refreshing negatives using the model's own approximate nearest neighbor index (ANN index built with FAISS). During training:
1. All passages are encoded and indexed with the current model.
2. For each query, top-ranked passages that are not relevant become hard negatives.
3. The model is trained on these hard negatives using InfoNCE.
4. The index is periodically rebuilt as the encoder improves.

This creates a self-improving feedback loop: better representations find harder negatives, which in turn push representations to become even more discriminative. ANCE significantly outperformed prior dense retrieval methods like DPR on the MSMARCO and Natural Questions benchmarks, demonstrating that negative sampling strategy is as important as architecture in dense retrieval.

## Applications in RAG

Contrastive learning underlies the bi-encoders used in modern Retrieval-Augmented Generation. Models like E5, BGE, GTE, and Sentence-BERT are trained with variants of InfoNCE loss. The bi-encoder produces separate query and document embeddings that can be compared with dot product or cosine similarity. ANCE-style hard negative mining, combined with large-scale supervised data (MS MARCO, Natural Questions) and knowledge distillation from cross-encoders, is the standard recipe for state-of-the-art dense retrievers.

## Significance

Contrastive learning reshaped self-supervised learning by showing that well-constructed positive pairs and sufficient negatives can substitute for labeled data. The InfoNCE loss, projection heads, hard negative mining, and momentum encoders are now standard tools. In NLP, contrastive training enabled dense retrieval to displace BM25 as the dominant first-stage retrieval method, which is the foundation of practical RAG systems.
