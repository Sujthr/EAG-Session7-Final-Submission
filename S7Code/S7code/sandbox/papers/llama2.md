# Llama 2: Open Foundation and Fine-Tuned Chat Models

**Authors:** Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Dan Bikel, Lukas Blecher, Cristian Canton Ferrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony Hartshorn, Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez, Madian Khabsa, Isabel Kloumann, Artem Korenev, Punit Singh Koura, Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee, Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov, Pushkar Mishra, Igor Molybog, Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Rashi Rungta, Kalyan Saladi, Alan Schelten, Ruan Silva, Eric Michael Smith, Ranjan Subramanian, Xiaoqing Ellen Tan, Binh Tang, Ross Taylor, Adina Williams, Jian Xiang Kuan, Puxin Xu, Zheng Yan, Iliyan Zarov, Yuchen Zhang, Angela Fan, Melanie Kambadur, Sharan Narang, Aurelien Rodriguez, Robert Stojnic, Sergey Edunov, Thomas Scialom (Meta AI)
**Year:** 2023

---

## Overview

Llama 2 is Meta's release of open-weight pre-trained language models and instruction-tuned chat variants, covering 7B, 13B, 34B, and 70B parameter sizes. Released with a permissive license for research and commercial use (with restrictions on deployments over 700M monthly active users), Llama 2 represented a major democratization of large language model capability. The chat variants (Llama-2-Chat) are fine-tuned using supervised learning and reinforcement learning from human feedback, and were competitive with proprietary models like GPT-3.5 on several benchmarks at the time of release.

## Pre-training

### Data
Llama 2 was pre-trained on 2 trillion tokens of publicly available text data — 40% more data than Llama 1. The data mixture was curated to include web text, books, code, and scientific literature, with heavy deduplication and quality filtering. Importantly, no Meta user data or proprietary sources were used. The pre-training corpus was upsampled on higher-quality sources (books, academic papers) relative to raw web text.

### Architecture
Llama 2 uses a standard autoregressive transformer decoder with several architectural choices from recent research:

- **Pre-normalization with RMSNorm:** Applied before each transformer sublayer rather than after, stabilizing training
- **SwiGLU activation function:** Used in the feed-forward network instead of ReLU, improving performance
- **Rotary Positional Embeddings (RoPE):** Instead of absolute positional embeddings, enabling better length generalization
- **Grouped Query Attention (GQA):** The 34B and 70B models use GQA, where multiple query heads share a single key-value head. This reduces the KV cache memory footprint during inference, enabling larger batch sizes and faster generation without significant quality loss.

Context length was extended to **4096 tokens** from Llama 1's 2048 tokens. Trained using AdamW optimizer with cosine learning rate schedule, gradient clipping, and weight decay.

### Scale and Compute
| Model | Parameters | Training Tokens | GPU-hours |
|-------|-----------|-----------------|-----------|
| Llama-2-7B | 6.7B | 2T | 184,320 |
| Llama-2-13B | 13B | 2T | 368,640 |
| Llama-2-70B | 68.9B | 2T | 1,720,320 |

Training was conducted on Meta's A100 GPU cluster using 2000 GPU nodes.

## Fine-tuning: Llama-2-Chat

The chat variants were produced through a multi-stage alignment pipeline:

### Stage 1: Supervised Fine-Tuning (SFT)
Starting from the pre-trained model, supervised fine-tuning was performed on approximately 27,540 high-quality annotation samples in a conversational format. Meta prioritized quality over quantity, finding that a small number of well-crafted demonstrations outperformed large amounts of lower-quality data. Annotators were instructed to write both the prompt and ideal response to ensure alignment.

### Stage 2: Reward Modeling
Two separate reward models were trained:
- **Helpfulness reward model:** Trained to predict which response is more helpful
- **Safety reward model:** Trained to predict which response is safer

This dual reward approach allows independent optimization signals for helpfulness and safety, preventing the common failure mode where safety training degrades helpfulness (and vice versa). Over 1 million human comparison annotations were collected across multiple rounds of iteration.

### Stage 3: Iterative RLHF with PPO and Rejection Sampling

Two complementary fine-tuning techniques were used in alternation:

**Rejection Sampling Fine-tuning:** Generate K responses per prompt, rank them with the reward model, and fine-tune on the highest-ranked response only. This is simpler than PPO and worked well for helpfulness.

**PPO (Proximal Policy Optimization):** Standard RLHF with KL penalty against the SFT model, using the combined reward signal. Multiple rounds of PPO were applied with updated reward models trained on each iteration's data — a strategy called **iterative RLHF**.

### Ghost Attention (GAtt)
A novel technique introduced to maintain system prompt adherence across multi-turn conversations. During synthetic data generation, the system prompt is synthetically inserted into all conversation turns, training the model to condition on initial instructions throughout a long dialogue.

## Safety Approach

Llama 2 incorporated extensive safety analysis:

- **Red-teaming:** Hundreds of adversarial prompts tested against the model iteratively
- **Safety-specific RLHF:** The safety reward model penalizes harmful outputs across 13 risk categories
- **Context distillation:** Safety system prompts are used to generate safe responses which are then fine-tuned into the model weights
- **Evaluation:** Published detailed model cards with performance across standard safety benchmarks (TruthfulQA, ToxiGen)

## Results

**Pre-trained model benchmarks (70B):**
- MMLU (5-shot): 68.9 (comparable to GPT-3.5)
- HumanEval (code): 29.9
- GSM8K (math): 56.8
- HellaSwag: 85.9

**Chat model human evaluations:**
- Llama-2-Chat-70B was preferred over ChatGPT (GPT-3.5) in approximately 36% of comparisons and tied in ~27% (effectively competitive)
- Substantially better safety ratings than open-source alternatives
- On helpful and harmless criteria, competitive with Claude at the 70B scale

## Significance

Llama 2 had an outsized impact on the AI ecosystem:

1. **Open-weight model availability:** Researchers worldwide gained access to production-quality models for study, fine-tuning, and deployment
2. **Fine-tuning democratization:** Enabled a wave of specialized fine-tunes (CodeLlama, Orca, WizardLM, Mistral-based models) on consumer hardware
3. **Reproducibility baseline:** Provided a common foundation for alignment and safety research
4. **Commercial deployment:** The permissive license enabled startups and enterprises to build products without API dependency
5. **Safety research platform:** The detailed methodology for iterative RLHF and dual reward modeling influenced subsequent open alignment work

Llama 2 established the pattern of open-weight releases with full technical transparency that subsequent models (Mistral, Llama 3, Phi-3, Gemma) followed, fundamentally reshaping the landscape of accessible large language model research.
