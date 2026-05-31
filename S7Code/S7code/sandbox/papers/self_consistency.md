# Self-Consistency Improves Chain of Thought Reasoning in Language Models

## Paper Details

- **Title:** Self-Consistency Improves Chain of Thought Reasoning in Language Models
- **Authors:** Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, Sharan Narang, Aakanksha Chowdhery, Denny Zhou
- **Affiliation:** Google Research, Google Brain
- **Year:** 2022
- **Venue:** ICLR 2023

---

## Motivation

Chain-of-thought (CoT) prompting (Wei et al., 2022) demonstrated that prompting LLMs to produce intermediate reasoning steps before a final answer significantly improved performance on multi-step reasoning tasks. However, original CoT used greedy decoding — the model generates a single reasoning path and commits to whatever answer that path produces.

This is problematic because:
- A single reasoning chain can go wrong at any step, and errors compound
- Greedy decoding discards the rich uncertainty information encoded in the model's output distribution
- Different valid reasoning paths may reach the same correct answer, while incorrect answers tend to arise from specific flawed paths

**Self-consistency** addresses this by exploiting the intuition that correct answers are more robust: multiple diverse reasoning paths should converge on the right answer even if any individual chain is flawed.

---

## Method

Self-consistency is a simple inference-time technique that requires no additional training or fine-tuning:

1. **Sample multiple reasoning chains:** Instead of using greedy decoding, sample K diverse reasoning chains from the model using temperature sampling (typically temperature 0.5–1.0). Each chain consists of step-by-step reasoning followed by a final answer.

2. **Marginalize over reasoning paths:** Collect the final answers from all K chains. Rather than using any individual chain's answer, apply a **majority vote** (or weighted majority if answer probabilities are available) to select the most consistent answer.

3. **Return the plurality answer:** The answer that appears most frequently across the K sampled chains is returned as the final answer.

Formally, if the model produces answers {a₁, a₂, ..., aₖ} from K sampled chains, the self-consistent answer is argmax_a Σᵢ 𝟙[aᵢ = a].

This approach treats the reasoning process as a latent variable and marginalizes over it, which is principled from a Bayesian standpoint: the final answer is selected based on the aggregated evidence from many plausible reasoning paths rather than a single greedy one.

---

## Experimental Setup and Results

The paper evaluates self-consistency on a broad range of reasoning benchmarks using PaLM (540B parameters) and GPT-3/Codex as backbone models.

**Arithmetic reasoning:**
- GSM8K: Self-consistency improved CoT accuracy from 56.9% (greedy) to 74.4% (+17.5 points) with PaLM 540B
- SVAMP: Improved from 79.0% to 86.8%
- AQuA: Improved from 35.8% to 48.8%
- MultiArith: Improved from 93.1% to 96.2%

**Commonsense reasoning:**
- StrategyQA: Improved from 72.9% to 83.1%
- ARC-Challenge: Gains of several points across configurations

**Symbolic reasoning:**
- Letter concatenation, coin flip tasks: Consistent improvements

These results held across model sizes and model families, demonstrating the method is broadly applicable. The gains were particularly pronounced on tasks where the model's greedy chain frequently made arithmetic or logical errors that could be "voted out" by the majority.

---

## Key Findings and Analysis

**Scaling with K:** Performance consistently improves as K (number of sampled chains) increases, with diminishing returns. Most gains are captured with K=20–40 samples. This presents a direct compute-quality trade-off controllable at inference time.

**Temperature sensitivity:** The method requires sufficient diversity among sampled chains, so temperature cannot be too low (near-greedy) or too high (incoherent). A temperature of ~0.7 was found to work well across tasks.

**Why it works:** The authors analyze the failure modes of individual chains. Incorrect answers tend to arise from specific reasoning errors (e.g., miscounting, wrong operation selection) that produce idiosyncratic wrong answers. Correct answers, being grounded in the problem structure, are reached by multiple independent paths. Majority voting therefore effectively filters noise.

**Comparison to self-consistency vs. fine-tuning:** In several settings, self-consistency with a large LLM matched or exceeded task-specific fine-tuned smaller models, with no training data required beyond the few-shot prompt.

---

## Significance

Self-consistency introduced a new paradigm for improving LLM reasoning: **inference-time compute scaling**. Rather than improving reasoning through better training, one can invest more compute at inference time to improve output quality.

This paper directly inspired a line of research on inference-time scaling (e.g., Best-of-N sampling, process reward models, tree-of-thought reasoning, and OpenAI's o1/o3 series), which has become one of the most active areas in LLM research. The core insight — that sampling diverse reasoning paths and aggregating is more robust than committing to one — remains a foundational principle in modern reasoning system design.
