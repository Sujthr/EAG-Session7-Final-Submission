# Self-Refine: Iterative Refinement with Self-Feedback

**Authors:** Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegreffe, Uri Alon, Nouha Dziri, Shrimai Prabhumoye, Yiming Yang, Shashank Gupta, Bodhisattwa Prasad Majumder, Katherine Hermann, Sean Welleck, Amir Yazdanbakhsh, Peter Clark
**Institution:** Carnegie Mellon University, Allen Institute for AI, University of Washington, Google DeepMind
**Year:** 2023 (NeurIPS 2023)

---

## Overview

Self-Refine introduces an iterative framework in which a single large language model generates an initial output, critiques that output, and then uses the critique to revise and improve the output — all without any additional training, gradient updates, or human feedback. The framework draws inspiration from the human process of drafting, reviewing, and editing, applying it entirely within the inference loop of a frozen LLM. Across seven diverse tasks, Self-Refine consistently improves over single-pass generation, demonstrating that LLMs contain substantial untapped capacity for self-evaluation and correction.

## Key Ideas

The core hypothesis is that LLMs already encode sufficient knowledge to evaluate the quality of their own outputs, and that this evaluative capacity can be leveraged to improve generation quality through iteration. The critical insight is that **generation quality and evaluation quality are separable** — a model may produce a suboptimal initial answer while still being capable of identifying what is wrong with it and suggesting specific improvements.

Self-Refine does not require:
- Fine-tuning or RLHF
- A separate critic or reward model
- Multiple distinct models
- External tool use or retrieval

This distinguishes it from prior work on iterative generation (which often used separate trained models) and from RLHF (which requires explicit human preference data).

## Method

The Self-Refine pipeline consists of three steps repeated iteratively:

**Step 1 — Generate:** The LLM produces an initial response to the task prompt using standard few-shot prompting.

**Step 2 — Feedback:** The same LLM is prompted with a feedback prompt that includes the original task, the generated output, and instructions to critique the output along specific dimensions. The feedback prompt asks for concrete, actionable suggestions rather than binary pass/fail judgments. Example feedback dimensions include: code correctness, sentiment alignment, logical coherence, or stylistic quality depending on the task.

**Step 3 — Refine:** The LLM is given the original task, the prior output, and the feedback, and is asked to produce a revised, improved version.

Steps 2 and 3 repeat until a stopping criterion is met — either a fixed number of iterations, or until the feedback signals the output is satisfactory (the LLM judges "no further changes needed").

All three steps use task-specific few-shot prompts, which are provided in the paper's appendix. The prompts guide the model toward the type of feedback most relevant to each task.

## Tasks Evaluated

Self-Refine is evaluated on seven tasks chosen to span different output modalities and quality criteria:

1. **Math reasoning** (GSM8K) — correctness of arithmetic and logical steps
2. **Code optimization** — improving the runtime efficiency of Python code
3. **Code readability** — improving variable names, structure, comments
4. **Constrained generation** — generating text satisfying specific lexical constraints
5. **Acronym generation** — generating meaningful acronyms for phrases
6. **Dialogue response generation** — improving empathy and coherence in dialog
7. **Sentiment reversal** — rewriting text to flip sentiment while preserving content

## Results

Across all seven tasks, Self-Refine improves over the zero-shot or few-shot single-pass baseline by meaningful margins. Key findings include:

- On code optimization, Self-Refine reduces execution time by 13% over baseline GPT-4 generation.
- On constrained generation, Self-Refine increases constraint satisfaction from 48% to 76%.
- On sentiment reversal, accuracy improves from 56% to 79%.
- On dialogue tasks, human evaluators prefer Self-Refine outputs 65% of the time vs. 35% for single-pass.

The improvements compound over iterations but plateau, typically stabilizing within 3-5 rounds. Using GPT-4 as the backbone model yields larger absolute gains than GPT-3.5, suggesting the method scales with model capability.

## Analysis and Limitations

The authors perform ablation studies showing that both feedback and refinement steps are necessary — skipping feedback and directly re-prompting for refinement yields much smaller gains. This validates that the feedback step genuinely transmits useful signal rather than serving merely as an additional decoding step.

A key limitation is **task dependency on feedback quality**: for tasks where the model's self-evaluation is unreliable (e.g., factual correctness of obscure claims), self-refine can propagate errors rather than fix them. The method works best when the model has strong domain knowledge about what "good" outputs look like.

## Significance

Self-Refine established iterative self-improvement as a viable inference-time technique, influencing a wave of subsequent work. It is closely related to Constitutional AI's critique-revision approach, Reflexion (which adds memory across episodes), and Chain-of-Thought self-consistency. The broader implication is that inference-time compute — spending more computation generating, evaluating, and revising — can substitute partially for additional training. This idea has since become central to test-time scaling research and reasoning-focused models like OpenAI o1 and DeepSeek-R1, which use extended internal "thinking" to improve output quality.
