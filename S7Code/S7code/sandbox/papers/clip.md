# CLIP: Learning Transferable Visual Models From Natural Language Supervision

## Paper Details

- **Title:** Learning Transferable Visual Models From Natural Language Supervision
- **Authors:** Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, Ilya Sutskever
- **Affiliation:** OpenAI
- **Year:** 2021
- **Venue:** ICML 2021

---

## Motivation

Standard computer vision models are trained on fixed, manually labeled datasets (ImageNet, COCO) with predefined class taxonomies. This creates several limitations:
- **Brittle generalization:** Models trained on a fixed label set struggle on distribution shifts and novel categories
- **Expensive supervision:** Human annotation is costly and doesn't scale to the breadth of visual concepts in the world
- **Rigid evaluation:** Task-specific models must be retrained or fine-tuned for each new application

The central question motivating CLIP: **can visual representations learned from natural language supervision be as good or better than those learned from task-specific labels, while generalizing far more broadly?**

---

## Training Data

CLIP was trained on a newly constructed dataset called **WIT (WebImageText)**, containing 400 million (image, text) pairs scraped from the internet. Each pair consists of an image and its associated caption or alt-text from a web page.

This scale — 400M pairs vs. ImageNet's 1.2M labeled images — combined with the natural language signal makes WIT qualitatively different from prior vision datasets. Rather than a fixed vocabulary of class labels, the text supervision spans an essentially unlimited range of visual concepts, styles, and contexts.

---

## Architecture and Training Objective

CLIP trains two encoders jointly:

**Image Encoder:** Either a ResNet (ResNet-50 through ResNet-101×64) or a Vision Transformer (ViT-B/32, ViT-L/14). The encoder maps an image to a d-dimensional embedding vector.

**Text Encoder:** A Transformer (63M parameters, 12 layers, 512-wide) that encodes the tokenized caption into a d-dimensional embedding vector.

**Contrastive training objective:** Given a batch of N (image, text) pairs, CLIP learns to maximize the cosine similarity of the N correct (image, text) embeddings while minimizing the cosine similarity of the N² - N incorrect pairings within the batch.

This is implemented as a symmetric cross-entropy loss over the similarity matrix:
- For each image, classify which of the N texts is its pair
- For each text, classify which of the N images is its pair

The loss is averaged over both directions. This is a form of **contrastive language-image pre-training** — the name gives the method its acronym.

Large batch sizes (32,768 in the largest experiments) are crucial, as they provide many negative examples per step, making the discrimination task harder and the representations more informative.

---

## Zero-Shot Transfer

CLIP's most significant capability is **zero-shot classification**: classifying images into arbitrary categories without any task-specific training.

The procedure is:
1. For a target classification dataset with classes {c₁, c₂, ..., cₙ}, construct text prompts such as "a photo of a {cᵢ}" for each class
2. Encode all prompts with the text encoder, producing class embedding vectors
3. Encode the query image with the image encoder
4. Classify the image to the class whose text embedding has highest cosine similarity to the image embedding

The model is never shown labeled training examples from the target dataset — classification is performed purely by matching image and text representations in the shared embedding space.

**Prompt engineering** matters significantly. Using "a photo of a {class}" rather than just the class label, and ensembling across multiple prompt templates (e.g., "a photo of a big {class}", "a photo of a small {class}"), consistently improves zero-shot accuracy.

---

## Results

**ImageNet zero-shot:** CLIP's ViT-L/14 achieves 76.2% top-1 accuracy on ImageNet zero-shot — matching ResNet-50 trained with 1.28 million supervised examples. This is a remarkable result: natural language supervision alone, with no labeled ImageNet examples, achieves competitive performance.

**Robustness to distribution shift:** On ImageNet distribution shift benchmarks (ImageNet-V2, ImageNet-R, ImageNet-Sketch, ObjectNet), CLIP showed significantly better robustness than supervised models. When supervised models drop from 76% to 45% on ObjectNet, CLIP drops from 76% to ~70% — a much shallower degradation.

**Breadth of zero-shot transfer:** CLIP was evaluated on 27 diverse datasets including fine-grained classification (flowers, cars, food), medical imaging (chest X-rays), satellite imagery, and OCR. It outperformed prior zero-shot methods on most datasets and was competitive with task-specific linear probes (fine-tuned classifiers) on many.

**Linear probe performance:** When a linear classifier is trained on CLIP features (with the encoder frozen), performance equals or exceeds full fine-tuning of prior state-of-the-art models on many benchmarks.

---

## Limitations

- CLIP struggles on abstract tasks (counting, spatial relationships) where vision-language co-training has not provided sufficient signal
- Performance on fine-grained datasets (MNIST, specific satellite domains) is weaker, as these are underrepresented in internet text-image pairs
- Biases present in internet data are embedded in the learned representations

---

## Significance and Legacy

CLIP established **contrastive vision-language pre-training** as the dominant paradigm for learning visual representations. It enabled:
- DALL-E and Stable Diffusion (using CLIP embeddings for text-to-image generation guidance)
- Open-vocabulary detection and segmentation (GLIP, OWL-ViT)
- Multimodal LLMs (LLaVA, Flamingo use CLIP as the visual backbone)
- Semantic image search at scale

The paper demonstrated that the scale and richness of natural language supervision could substitute for — and in many ways surpass — expensive manual annotation, fundamentally shifting how visual representations are trained.
