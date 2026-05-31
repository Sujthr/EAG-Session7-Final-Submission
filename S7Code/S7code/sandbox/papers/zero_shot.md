# Zero-Shot Learning in NLP: CLIP, GPT Zero-Shot Prompting, and Chain-of-Thought

**Key Papers:**
- *Learning Transferable Visual Models From Natural Language Supervision* — Radford et al., OpenAI, 2021 (CLIP)
- *Language Models are Few-Shot Learners* — Brown et al., OpenAI, 2020 (GPT-3)
- *Finetuned Language Models are Zero-Shot Learners* — Wei et al., Google Research, 2021 (FLAN)
- *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models* — Wei et al., Google, 2022

---

## Overview

Zero-shot learning refers to a model's ability to perform a task correctly without having seen any labeled examples of that specific task at inference time. This is distinguished from few-shot learning (a handful of examples provided in the prompt) and fine-tuning (gradient updates on task-specific data). The emergence of zero-shot capabilities in large language models represents a fundamental shift in how NLP systems are deployed — moving from task-specific trained models to general-purpose models that understand task descriptions.

## CLIP: Zero-Shot Image Classification

CLIP (Contrastive Language-Image Pretraining) demonstrated powerful zero-shot transfer in the vision domain. Trained on 400 million image-text pairs scraped from the internet, CLIP learns a joint embedding space where image representations and text representations of corresponding captions are aligned via contrastive learning.

**Zero-shot mechanism**: To classify an image into N categories, CLIP constructs N text prompts of the form "a photo of a [category]." The image is encoded and compared (via cosine similarity) against all N text encodings. The highest-scoring category is the prediction — no category-specific training is ever performed.

CLIP achieves 76.2% top-1 accuracy on ImageNet zero-shot, matching the performance of the original supervised ResNet-50, despite never being trained on ImageNet labels. On 27 downstream vision datasets, CLIP outperforms few-shot linear probes on 16 of them using only zero-shot prompting.

The success of CLIP zero-shot prompted the concept of **prompt engineering** — carefully crafting the text description (e.g., "a satellite photo of a [land use type]" rather than just "[land use type]") substantially affects accuracy. This insight transferred directly to text-only LLMs.

## GPT-3 Zero-Shot and Few-Shot Prompting

GPT-3 (175B parameters) demonstrated that very large language models develop emergent zero-shot capabilities. Given a task described in natural language (a "prompt"), GPT-3 can perform sentiment analysis, translation, arithmetic, commonsense reasoning, and many other tasks with no gradient updates.

**Zero-shot**: The model is given only a task description and the input, e.g.:
```
Translate English to French:
sea otter =>
```

**Few-shot (in-context learning)**: The prompt includes 1–32 examples of input-output pairs before the actual query. GPT-3 learns the task format and pattern from these examples, updating only in the forward pass (no weight changes).

GPT-3's zero-shot performance on SuperGLUE approaches fine-tuned BERT-Large on several subtasks. The paper demonstrated a strong scaling relationship: zero-shot and few-shot capabilities improve dramatically with model size, with near-discontinuous jumps at certain parameter counts. This "emergent abilities" phenomenon — capabilities that appear suddenly at scale — has become a central topic in scaling law research.

## FLAN: Instruction Tuning for Zero-Shot Transfer

Wei et al. (2021) identified a key bottleneck in GPT-style zero-shot learning: while massive pretraining provides knowledge, the model does not inherently learn to follow task *descriptions* as instructions. FLAN (Finetuned Language Net) addresses this by fine-tuning a 137B LaMDA-PT model on 62 NLP datasets, each reformulated as natural language instructions.

The critical finding is that **instruction tuning dramatically improves zero-shot performance on unseen tasks** — tasks held out from the instruction-tuning mixture. FLAN outperforms GPT-3 (which has 17x more parameters) on zero-shot evaluations of 20 of 25 tasks. This demonstrated that the *format* of training (instruction-following) matters enormously for zero-shot generalization, not just model size. FLAN directly inspired subsequent instruction tuning approaches: Alpaca, WizardLM, OpenHermes, and the broader RLHF pipeline used in ChatGPT.

## Zero-Shot Chain-of-Thought

Wei et al. (2022) showed that providing chains of reasoning examples (few-shot CoT) dramatically improves LLM performance on multi-step reasoning tasks like arithmetic, symbolic manipulation, and commonsense reasoning. But a key follow-up finding (Kojima et al., 2022) showed that simply appending **"Let's think step by step"** to the prompt elicits chain-of-thought reasoning in a **zero-shot** setting — no examples required.

This zero-shot CoT prompt causes models to generate explicit intermediate reasoning steps before arriving at a final answer. On GSM8K (grade school math word problems), zero-shot CoT with GPT-3 improves accuracy from 10.4% (standard zero-shot) to 40.7%. On MultiArith, the jump is from 17.7% to 78.7%.

Zero-shot CoT works because:
1. The instruction shifts the model's decoding toward slower, more deliberate token sequences that mirror step-by-step reasoning in the training data.
2. Generating intermediate steps provides additional context that makes the final answer more constrained and accurate.
3. Errors in reasoning become visible in the output, making them partially self-correctable.

## Significance

Zero-shot capabilities have transformed the practical deployment of NLP. Rather than training task-specific models, practitioners can write natural language task descriptions and immediately get usable performance. This enables rapid prototyping, reduces labeled data requirements, and generalizes across languages and domains. Chain-of-thought zero-shot has become a standard prompting strategy, and the concept of zero-shot evaluation as a benchmark for general intelligence has shaped how foundation models are evaluated today. These ideas directly underpin modern LLM-based agents and reasoning systems.
