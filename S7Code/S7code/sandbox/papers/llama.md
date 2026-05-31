# LLaMA: Open and Efficient Foundation Language Models

**Authors:** Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, Guillaume Lample  
**Institution:** Meta AI  
**Year:** 2023  
**Paper:** arXiv:2302.13971

---

## Overview

LLaMA (Large Language Model Meta AI) is a collection of foundation language models ranging from 7B to 65B parameters, introduced by Meta AI in February 2023. Its central thesis is that training smaller models on substantially more data than is typically used can yield models that outperform much larger counterparts — challenging the prevailing assumption that model scale alone drives performance. LLaMA became one of the most influential open-weight releases in the history of AI, spawning a vast ecosystem of fine-tuned variants and research derivatives.

## Key Ideas

The paper's core insight draws from the scaling laws work by Hoffmann et al. (Chinchilla, 2022), which showed that compute-optimal training requires far more tokens than had been used to train models like GPT-3. While Chinchilla focused on compute-optimality at training time, LLaMA takes a different angle: optimizing for **inference efficiency**. A 13B model trained on 1T tokens may be more practically useful than a 175B model trained on 300B tokens, because the smaller model is cheaper to serve at scale.

LLaMA models were trained exclusively on publicly available data, including:
- English CommonCrawl (67%)
- C4 (15%)
- GitHub code (4.5%)
- Wikipedia in 20 languages (4.5%)
- Books (Project Gutenberg + Books3): ~4.5%
- ArXiv scientific papers (2.5%)
- Stack Exchange (2%)

No proprietary or licensed data was used, making LLaMA reproducible and auditable in ways that GPT-3 and PaLM are not.

## Architecture

LLaMA adopts the standard autoregressive transformer decoder architecture with several key modifications drawn from recent advances:

**Pre-normalization (RMSNorm):** Rather than applying LayerNorm to the output of each sub-layer, LLaMA normalizes the *input* to each transformer sub-layer using RMSNorm. This improves training stability and was found to outperform post-normalization in practice.

**SwiGLU Activation:** The ReLU nonlinearity in the feed-forward layers is replaced with SwiGLU (a gated variant of the Swish activation), introduced by Noam Shazeer. SwiGLU has been empirically shown to improve performance on language modeling tasks. The FFN dimension is adjusted to 2/3 × 4d to keep parameter counts comparable.

**Rotary Positional Embeddings (RoPE):** Absolute positional embeddings are removed; instead, rotary position embeddings are added at each attention layer. RoPE encodes position information by rotating query and key vectors, which naturally handles relative positions and generalizes better to sequence lengths not seen during training.

**Grouped-Query Attention (in LLaMA 2):** The original LLaMA used standard multi-head attention, but this was upgraded in subsequent work.

**Context Length:** 2048 tokens for the original LLaMA models.

**Tokenizer:** Byte-pair encoding (BPE) using the SentencePiece implementation, with a vocabulary of 32,000 tokens. Numbers are split into individual digits to improve arithmetic generalization.

## Training Details

All models were trained using AdamW with β1 = 0.9, β2 = 0.95, and a cosine learning rate schedule with 2000 warmup steps. Gradient clipping at 1.0 was applied. Training used mixed precision (bfloat16) and was performed on 2048 A100 GPUs using tensor parallelism and pipeline parallelism via efficient distributed training infrastructure. The 65B model was trained on approximately 1.4T tokens; smaller models on 1T tokens.

## Results and Benchmarks

LLaMA's results were striking:

- **LLaMA-13B outperforms GPT-3 (175B)** on most benchmarks despite being more than 10x smaller.
- **LLaMA-65B is competitive with Chinchilla-70B and PaLM-540B** on reasoning, commonsense, and reading comprehension tasks.
- Strong performance on: BoolQ, PIQA, HellaSwag, WinoGrande, ARC (Easy/Challenge), OpenbookQA, NaturalQuestions, TriviaQA, HumanEval (code), and GSM8k (math reasoning).

The model showed notable weaknesses in tasks requiring multi-step instruction following, which motivated the development of instruction-tuned variants like Alpaca, Vicuna, and eventually LLaMA 2-Chat.

## Significance and Impact

LLaMA's release had an outsized impact on AI research and the open-source community:

1. **Democratization:** By releasing weights (under a research license), Meta enabled academic researchers and independent developers to study, fine-tune, and deploy competitive LLMs without access to massive compute infrastructure.
2. **Fine-tuning research explosion:** Models like Alpaca (Stanford, using self-instruct), Vicuna, WizardLM, and dozens of others emerged within weeks, demonstrating that instruction tuning on small curated datasets could rapidly improve base model capabilities.
3. **Quantization and edge deployment:** LLaMA's manageable size made it the primary target for quantization research (GPTQ, llama.cpp, GGUF), enabling inference on consumer GPUs and even CPUs.
4. **Architecture standard:** The architectural choices in LLaMA (RMSNorm, SwiGLU, RoPE) became a de facto standard for subsequent open models including Mistral, Falcon, and LLaMA 2/3.

The paper demonstrated that transparency about training data and model architecture, combined with rigorous evaluation, enables genuine scientific progress — in contrast to the trend of closed, proprietary model releases.
