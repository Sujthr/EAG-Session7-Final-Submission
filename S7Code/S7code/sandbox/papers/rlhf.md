# Reinforcement Learning from Human Feedback (RLHF)

**Primary Papers:**
- *Learning to Summarize from Human Feedback* — Nisan Stiennon, Long Ouyang, Jeff Wu, Daniel M. Ziegler, Ryan Lowe, Chelsea Voss, Alec Radford, Dario Amodei, Paul Christiano (OpenAI) — 2020
- *Training Language Models to Follow Instructions with Human Feedback (InstructGPT)* — Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Christiano, Jan Leike, Ryan Lowe (OpenAI) — 2022

---

## Overview

Reinforcement Learning from Human Feedback (RLHF) is a training methodology that aligns language model behavior with human preferences by using human comparison judgments as a reward signal. Rather than relying solely on next-token prediction loss — which optimizes for plausible text, not helpful or accurate text — RLHF directly optimizes for what humans find good. These two papers together established RLHF as the dominant technique for building instruction-following and helpful AI assistants.

## Motivation

Pre-trained language models optimized with standard supervised learning exhibit several alignment problems:

- **Sycophancy over truthfulness:** Models learn to produce fluent text that mirrors training data patterns rather than grounded facts
- **Harmful content generation:** Models may produce toxic, biased, or dangerous outputs present in web-scale training data
- **Failure to follow instructions:** A model trained on internet text is not trained to obey user intents expressed as instructions
- **Specification gaming:** Metrics like ROUGE or perplexity correlate poorly with human judgments of quality

Human feedback provides a richer signal that naturally captures nuanced quality dimensions — helpfulness, harmlessness, honesty — that are hard to formalize programmatically.

## The RLHF Pipeline

RLHF operates in three stages:

### Stage 1: Supervised Fine-Tuning (SFT)
A pre-trained language model is fine-tuned on a dataset of human-written demonstrations. For InstructGPT, labelers wrote high-quality responses to sampled prompts from the OpenAI API. This produces an initial policy that can follow instructions at a basic level.

### Stage 2: Reward Model Training
Human labelers are shown pairs of model outputs for the same prompt and asked to select which is better. These comparison pairs train a **reward model** (RM) — a language model with a scalar head — that learns to predict human preference scores for arbitrary outputs. The RM is trained with a pairwise ranking loss:

```
L_RM = -E[(r(x, y_w) - r(x, y_l))]
```

where `y_w` is the preferred response and `y_l` the dispreferred response.

### Stage 3: Reinforcement Learning via PPO
The SFT model is used as the starting policy and is further trained using Proximal Policy Optimization (PPO) to maximize the reward model's scores. A KL divergence penalty against the original SFT model is added to prevent reward hacking — the model from producing out-of-distribution text that achieves high scores by exploiting weaknesses in the reward model:

```
R(x, y) = r_RM(x, y) - beta * KL[pi_RL(y|x) || pi_SFT(y|x)]
```

This penalty ensures the model remains coherent and doesn't diverge excessively from the supervised baseline.

## Summarization Paper (Stiennon et al., 2020)

The 2020 summarization paper was the first to demonstrate RLHF at scale for language tasks. Applied to the TL;DR Reddit summarization task:

- RLHF-trained models produced summaries humans preferred 70-80% of the time over supervised baselines
- Human raters preferred RLHF summaries over reference (human-written) summaries from the dataset 63% of the time
- Models learned to copy less and extract more meaningful content
- Reward modeling captured quality dimensions beyond ROUGE scores

## InstructGPT (Ouyang et al., 2022)

InstructGPT applied RLHF to GPT-3 to create instruction-following models. Key findings:

- **Human preference:** InstructGPT (1.3B parameters, RLHF-trained) was preferred over GPT-3 (175B) on 85% of prompts despite being 100x smaller
- **Alignment tax reduction:** The KL penalty mitigated performance degradation on standard NLP benchmarks
- **Truthfulness:** InstructGPT outputs were rated more truthful and less hallucinated than GPT-3
- **Reduced toxicity:** RLHF substantially reduced generation of harmful content on sensitive prompts

The InstructGPT work also introduced the concept of "alignment tax" — the tradeoff between instruction following and raw benchmark performance — and showed it could be largely mitigated.

## Significance

RLHF became the foundational alignment technique of the modern LLM era:

1. **ChatGPT** (November 2022) was built on InstructGPT's methodology and demonstrated RLHF's commercial viability
2. **Claude** (Anthropic) used RLHF as a core component alongside Constitutional AI
3. **Llama 2 Chat**, **Gemini**, **Mistral Instruct** all rely on RLHF variants
4. The reward model framework inspired Direct Preference Optimization (DPO), which eliminates the RL loop while retaining preference learning
5. RLHF raised foundational questions about scalable oversight — using AI to assist humans in providing feedback at scales beyond direct human evaluation

RLHF remains the standard methodology for converting capable but unaligned pre-trained language models into safe, helpful, and instruction-following assistants.
