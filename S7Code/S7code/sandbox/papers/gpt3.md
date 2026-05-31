# Language Models are Few-Shot Learners (GPT-3)

**Authors:** Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, Dario Amodei (OpenAI)
**Year:** 2020

---

## Overview

GPT-3 (Generative Pre-trained Transformer 3) introduced the concept of **few-shot learning in large language models** as a practical paradigm. The paper demonstrated that scaling a transformer language model to 175 billion parameters creates emergent capabilities that allow the model to perform well on diverse tasks given only a few natural language examples in the prompt — with no gradient updates or fine-tuning required. This work fundamentally shifted the research community's understanding of what scale could achieve.

## Key Ideas

### The Scaling Hypothesis in Practice
Prior work by Kaplan et al. (2020) established scaling laws showing that language model performance improves smoothly and predictably with model size, data, and compute. GPT-3 was the empirical test of pushing these laws to an unprecedented scale. The core hypothesis was that sufficiently large language models, trained to predict text, would implicitly learn to perform tasks they were never explicitly trained on.

### In-Context Learning
The defining contribution of GPT-3 is demonstrating **in-context learning**: the ability to perform new tasks by conditioning on a small number of examples provided in the prompt. Three regimes are distinguished:

- **Zero-shot:** Only a task description is provided, no examples
- **One-shot:** One labeled example is provided
- **Few-shot:** 10-100 examples are provided in the prompt (limited by context window)

Crucially, no weights are updated — the model adapts its output distribution at inference time based solely on the in-context examples.

## Architecture

GPT-3 uses the same autoregressive decoder-only transformer architecture as GPT-2, with modifications inspired by Sparse Transformer:

- **175 billion parameters** (GPT-3 full size)
- 96 transformer layers
- 96 attention heads
- 12,288 model dimension
- 2,048-token context window
- Alternating dense and locally banded sparse attention patterns

A range of model sizes from 125M to 175B parameters were trained, enabling systematic study of scaling behavior. Training used approximately 300 billion tokens from a filtered version of Common Crawl (60%), WebText2 (22%), Books1 (8%), Books2 (8%), and English Wikipedia (3%).

## Results

GPT-3 achieved strong results across a wide range of benchmarks in the few-shot setting:

- **SuperGLUE:** GPT-3 few-shot matched fine-tuned BERT-Large on several tasks
- **TriviaQA:** 71.2% (few-shot), competitive with fine-tuned open-domain QA systems
- **WebQuestions:** 41.5% (few-shot)
- **Arithmetic tasks:** GPT-3 could perform 2-digit addition/subtraction, multiplication, and even some 3-digit arithmetic in zero-shot
- **Translation:** Competitive with supervised baselines for French, German, Romanian to English
- **SAT Analogy:** 65.2% accuracy vs. human average of 57%

Performance improved consistently with model size across nearly all tasks, confirming the scaling hypothesis.

## Limitations Acknowledged

The authors were candid about significant limitations:

- **In-context learning inefficiency:** Few-shot requires many examples to fit in a finite context window
- **Text generation artifacts:** GPT-3 struggles with tasks requiring multi-step logical reasoning
- **Factual errors and hallucination:** The model confidently generates incorrect facts
- **Bias and toxicity:** Training data encodes social biases that surface in generations
- **No grounding:** The model has no access to external knowledge or real-time information

## Significance

GPT-3 demonstrated that in-context learning is a powerful and practical paradigm, making large language models accessible without task-specific fine-tuning. This opened entirely new application domains — the OpenAI API built on GPT-3 enabled the first wave of LLM-powered products.

The paper also prompted renewed focus on emergent abilities — capabilities that appear suddenly at scale without being explicitly trained. GPT-3 laid the foundation for instruction-tuned successors (InstructGPT, ChatGPT), reinforcement learning from human feedback (RLHF), and the broader frontier model ecosystem. It also catalyzed academic interest in understanding why and how in-context learning works mechanistically.

GPT-3's release marked the beginning of the modern "foundation model" era, where single large pre-trained models serve as the base for an enormous range of downstream applications.
