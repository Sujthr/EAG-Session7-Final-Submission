# Code Generation with Large Language Models: Codex, CodeLlama, AlphaCode, and GitHub Copilot

## Overview

Code generation is one of the most impactful applications of large language models. The ability to translate natural language specifications into executable code, complete partial implementations, explain existing code, and fix bugs represents a fundamental shift in software development productivity. This document surveys the landmark models, datasets, and benchmarks that define the code LLM landscape.

---

## Codex (Chen et al., 2021)

**Authors:** Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, et al. (OpenAI)  
**Paper:** "Evaluating Large Language Models Trained on Code"  
**Published:** arXiv:2107.03374 (2021)

### Training

Codex is a GPT-3 descendant fine-tuned on 159GB of Python code scraped from public GitHub repositories (54 million files). The model family ranges from 12M to 12B parameters. Unlike GPT-3, which was trained on general web text, Codex's fine-tuning corpus consisted entirely of code, dramatically improving its ability to produce syntactically valid, semantically meaningful programs.

### Architecture

Codex inherits GPT-3's decoder-only transformer architecture. Key adaptations for code:
- The tokenizer was adjusted to handle code-specific patterns (indentation, operators, common identifier sequences)
- Training on code with long-range dependencies (function definitions referenced hundreds of lines later) required careful handling of the 4096-token context window

### HumanEval Benchmark

A central contribution of the Codex paper was introducing **HumanEval** — a benchmark of 164 hand-written Python programming problems with unit tests. Each problem consists of a function signature and docstring; the model generates a completion, which is executed against hidden test cases.

**pass@k metric:** The probability that at least one of k sampled completions passes all test cases. This captures the model's ability to generate a correct solution, not just a typical one.

Codex-12B achieved **28.8% pass@1** and **72.3% pass@100** on HumanEval — a substantial improvement over GPT-3 (0% pass@1). The gap between pass@1 and pass@100 illustrated that models often "know" the solution but need multiple attempts to produce a clean correct version.

### Limitations Identified

- Performance degrades sharply on problems requiring multi-function implementations
- Struggles with algorithmic problems requiring dynamic programming or complex graph algorithms
- Generated code may be syntactically correct but semantically incorrect (plausibly-looking bugs)
- Security vulnerabilities: Codex occasionally generates insecure patterns (e.g., SQL injection, hardcoded credentials) present in training data

---

## GitHub Copilot

Built on Codex and launched publicly in 2022, GitHub Copilot integrates code generation into IDEs (VS Code, JetBrains) as an inline autocomplete assistant. Copilot extended Codex with:

- **Fill-in-the-Middle (FIM) training:** Models are trained to complete code given both the prefix (code above the cursor) and the suffix (code below), enabling more coherent completions that respect surrounding context
- **Repository-level context:** Modern Copilot variants retrieve relevant code from the rest of the open repository, injecting similar functions and type definitions into the prompt before generation
- **Telemetry-driven fine-tuning:** Large-scale A/B testing of suggestion acceptance rates informs model improvements

A GitHub/NBER study found that developers using Copilot completed tasks ~55% faster on average, with the greatest gains on boilerplate-heavy tasks. Copilot became the first widely adopted AI pair programmer, establishing the economic case for code LLMs in production.

---

## AlphaCode (Li et al., 2022)

**Authors:** Yujia Li, David Choi, Junyoung Chung, et al. (Google DeepMind)  
**Paper:** "Competition-Level Code Generation with AlphaCode"  
**Published:** Science, 2022

### Key Innovation: Scale + Sampling + Filtering

AlphaCode targeted competitive programming problems from Codeforces — problems that require not just coding ability but problem-solving, algorithm design, and mathematical reasoning. Its approach combined:

- **Large-scale pretraining:** 41B parameter model trained on 715GB of code across 12 programming languages
- **Massive sampling:** Generating up to 1 million candidate solutions per problem
- **Clustering and filtering:** Using test-case-based filtering (keeping only solutions that pass the provided example cases) and clustering by behavior (grouping solutions by which test cases they pass), then submitting the best representative from each cluster

**Results:** AlphaCode achieved an estimated rank in the top 54.3% of Codeforces competitors — comparable to a median human programmer on competitive programming contests. This was the first demonstration of LLM performance at human-competitive levels on algorithmic programming challenges.

---

## CodeLlama (Rozière et al., 2023)

**Authors:** Baptiste Rozière, et al. (Meta AI)  
**Paper:** "Code Llama: Open Foundation Models for Code"  
**Published:** arXiv:2308.12950 (2023)

### Architecture and Training

CodeLlama is built on top of Llama 2 (7B, 13B, 34B parameter variants) and further trained on 500B tokens of code-specific data (additional 100B code tokens beyond Llama 2's pretraining). Training stages:

1. **Code-specialized pretraining:** Additional training on code corpus using the Llama 2 base
2. **Long-context fine-tuning:** Fine-tuned on 16K-token context windows (vs. 4K for standard Llama 2) using position interpolation
3. **Fill-in-the-Middle (FIM) training:** Supports infilling tasks
4. **Instruction fine-tuning (CodeLlama-Instruct):** Fine-tuned to follow natural language coding instructions

**Results:**
- CodeLlama-34B: **53.7% pass@1** on HumanEval (surpassing GPT-3.5's ~48.1%)
- CodeLlama-Python-34B: **53.7%** on HumanEval, state-of-the-art among open models at release
- Strong performance on MBPP (Mostly Basic Programming Problems) benchmark

CodeLlama's release established a strong open-source alternative to proprietary code models, enabling local deployment without API costs or data privacy concerns.

---

## Additional Notable Models

**StarCoder (BigCode, 2023):** 15.5B parameter model trained on The Stack — a carefully curated 6.4TB corpus of permissively licensed code from GitHub. Emphasized data governance: all training code was opt-out, allowing developers to request removal. StarCoder introduced **Multi-Query Attention** for efficient inference.

**DeepSeek-Coder (2024):** Models up to 33B parameters with strong performance on HumanEval (79.3% pass@1 for the 33B-Instruct variant) and LiveCodeBench, trained on 2T tokens of code and related text.

**GPT-4 on Code:** OpenAI reported GPT-4 achieving ~67% pass@1 on HumanEval and competitive performance on more complex benchmarks like SWE-bench (GitHub issue resolution).

---

## HumanEval and Beyond

HumanEval's 164 problems are now largely saturated by frontier models. Newer benchmarks include:

- **MBPP (Austin et al., 2021):** 974 Python problems covering basic programming concepts
- **DS-1000 (Lai et al., 2022):** 1000 data science problems requiring NumPy, Pandas, Sklearn
- **SWE-bench (Jimenez et al., 2023):** Real GitHub issues from major Python repositories — requires understanding large codebases, identifying bug locations, and generating patches
- **LiveCodeBench:** Continuously updated benchmark drawing from competitive programming platforms

---

## Significance

Code generation LLMs have transformed software development in just a few years. From Codex's 28.8% pass@1 in 2021 to frontier models exceeding 80% by 2024, progress has been rapid and consequential. GitHub Copilot alone has millions of active users. More fundamentally, code generation demonstrated that LLMs can operate in formal, executable domains where correctness is objectively verifiable — paving the way for LLMs as general problem-solving engines beyond natural language.
