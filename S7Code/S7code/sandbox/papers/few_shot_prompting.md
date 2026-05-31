# Few-Shot Prompting: In-Context Learning, Chain-of-Thought, and Prompt Engineering

## Overview

Few-shot prompting is a paradigm in which a language model is conditioned on a small number of input-output examples — provided directly in the prompt — to perform a new task without any weight updates. The model learns the task "in context," making it one of the most practically transformative capabilities of large language models. Understanding few-shot prompting requires understanding both the empirical findings that established it and the subsequent techniques that have dramatically extended its effectiveness.

---

## Language Models as Few-Shot Learners (GPT-3)

**Authors:** Brown et al. (OpenAI, 2020)  
**Paper:** "Language Models are Few-Shot Learners"  
**Published:** NeurIPS 2020

### Key Discovery

GPT-3 (175B parameters) demonstrated that language models trained purely on next-token prediction develop strong few-shot learning abilities as an emergent property of scale. By simply prepending examples to a prompt, GPT-3 achieved competitive performance on dozens of NLP benchmarks without any fine-tuning.

### Learning Modes

The paper introduced three distinctions that are now standard terminology:

- **Zero-shot:** Task described in natural language; no examples. ("Translate English to French: cheese =>")
- **One-shot:** One example provided in the prompt.
- **Few-shot:** K examples (typically 10–100) provided in the prompt.

Few-shot performance scaled continuously with model size, with the largest models showing the largest gains from additional examples. This **in-context learning (ICL)** phenomenon — where performance improves without gradient updates — became one of the most studied questions in LLM research.

### Why In-Context Learning Works

Later theoretical work (Xie et al., 2022; Akyürek et al., 2022) proposed that transformers implicitly perform Bayesian inference during in-context learning — the examples update the model's implicit prior over task hypotheses. Other work suggests transformers implement gradient-descent-like algorithms in their forward pass via attention heads.

---

## Chain-of-Thought Prompting (CoT)

**Authors:** Wei et al. (Google Brain, 2022)  
**Paper:** "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models"  
**Published:** NeurIPS 2022

### The Core Idea

Standard few-shot prompting provides (input, output) pairs. CoT prompting provides (input, **reasoning chain**, output) triples, where the reasoning chain is an explicit step-by-step derivation leading to the final answer.

Example (arithmetic):
```
Q: If a train travels at 60 mph for 2.5 hours, how far does it travel?
A: The train travels at 60 miles per hour. Over 2.5 hours, the distance is
   60 × 2.5 = 150 miles. The answer is 150 miles.
```

### Findings

CoT prompting dramatically improved performance on multi-step reasoning benchmarks:
- GSM8K (grade school math): +24 percentage points over standard few-shot
- MATH: significant gains on complex word problems
- CommonsenseQA and StrategyQA: improved logical reasoning

Critically, CoT benefits only emerge in models above ~100B parameters. Smaller models that attempt to produce reasoning chains actually perform worse, suggesting CoT requires sufficient capacity to implement coherent multi-step reasoning.

### Zero-Shot CoT

Kojima et al. (2022) showed that simply appending **"Let's think step by step."** to a prompt — without any worked examples — substantially improves reasoning performance. This "zero-shot CoT" works because the prompt activates the model's implicit chain-of-thought generation capability.

---

## Self-Consistency

**Authors:** Wang et al. (Google Brain, 2022)  
**Paper:** "Self-Consistency Improves Chain of Thought Reasoning in Language Models"

Self-consistency samples multiple diverse reasoning chains from the model (via temperature > 0) for a single question and takes the majority vote over final answers. This exploits the observation that correct reasoning paths are more internally consistent than incorrect ones. Self-consistency improves CoT accuracy by 5–15% on math and reasoning benchmarks with no additional training.

---

## Few-Shot Example Selection

Not all examples are equally effective. Research has identified several principles for selecting high-quality few-shot demonstrations:

**Relevance:** Examples semantically similar to the test query produce better performance (Liu et al., 2022). Retrieval-based example selection (KATE) uses kNN over an example pool to dynamically select the most relevant demonstrations.

**Diversity:** Covering different aspects of the task space reduces variance and improves generalization.

**Order sensitivity:** LLMs are sensitive to the order of few-shot examples; later examples have disproportionate influence (recency bias). Calibration techniques can mitigate this.

**Label balance:** Roughly equal representation of each output class prevents the model from defaulting to the majority class.

---

## Prompt Engineering Strategies

### Role Prompting
Assigning the model a persona ("You are an expert Python developer with 10 years of experience") has been shown to improve performance on domain-specific tasks, likely by activating relevant in-weights knowledge.

### Instruction Following
Rather than demonstrating tasks through examples, instruction-tuned models (InstructGPT, Claude, Llama-2-chat) respond well to explicit natural-language instructions. For these models, clear, specific instructions often outperform few-shot examples.

### Tree of Thoughts (ToT)
Yao et al. (2023) extended CoT by framing problem-solving as a search over a tree of partial solutions. The model generates multiple reasoning branches at each step, uses self-evaluation to score partial solutions, and applies BFS or DFS to navigate toward a complete solution. ToT substantially outperforms CoT on tasks requiring exploration (creative writing, planning, mathematical proofs).

### ReAct: Reason + Act
Yao et al. (2022) introduced ReAct, which interleaves chain-of-thought reasoning with tool-use actions (search, calculator, code execution) in the prompt. The model reasons about what information it needs, takes an action to retrieve it, observes the result, and continues reasoning. ReAct is the foundation of most LLM agent frameworks.

### Least-to-Most Prompting
Zhou et al. (2022) proposed decomposing complex questions into simpler sub-questions, solving them sequentially, and using each answer as context for the next. This addresses the failure mode of CoT on problems requiring more reasoning steps than the model's "working memory" supports.

---

## Automatic Prompt Optimization

**APE (Automatic Prompt Engineer):** Proposes using an LLM to generate and score candidate prompts, selecting the best-performing instruction without human engineering. Achieves performance competitive with human-written prompts on many benchmarks.

**DSPy:** A framework from Stanford that treats prompts and few-shot examples as optimizable parameters, using a compiler that automatically tunes prompts by running small-scale evaluations.

---

## Significance

Few-shot prompting transformed LLMs from fine-tuning-dependent tools into general-purpose reasoning engines that can be adapted to new tasks with only a handful of examples. Chain-of-thought prompting revealed that the reasoning process itself — not just the final answer — is learnable in-context, enabling LLMs to tackle multi-step mathematical, logical, and commonsense reasoning that was previously intractable. These techniques are now foundational to virtually every production LLM application.
