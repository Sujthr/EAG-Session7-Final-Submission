# Toolformer: Language Models Can Teach Themselves to Use Tools

**Authors:** Timo Schick, Jane Dwivedi-Yu, Roberto Dessi, Roberta Raileanu, Maria Lomeli, Luke Zettlemoyer, Nicola Cancedda, Thomas Scialom (Meta AI Research)
**Year:** 2023 (arXiv February 2023)

---

## Overview

Toolformer addresses a core limitation of language models: despite their broad linguistic capabilities, they cannot access real-time information, perform exact arithmetic, or query external systems. Previous approaches to tool use required either large amounts of human-annotated demonstrations of tool calls or reinforcement learning with sparse rewards. Toolformer's key innovation is a **self-supervised method** that teaches a language model when and how to use tools by generating its own training data — requiring only a handful of human-written examples per tool as seed demonstrations.

## Problem Statement

Large language models exhibit systematic failures on tasks that are trivially solved by simple APIs:

- Mathematical calculations (even basic arithmetic leads to errors at scale)
- Accessing up-to-date facts (the model's knowledge is frozen post-training)
- Translating between languages with high accuracy
- Searching the web for recent information
- Performing calendar or date arithmetic

Integrating tool use via fine-tuning traditionally requires expensive human annotation of when and how to invoke each tool — labeling thousands of examples of API calls embedded in natural text.

## Method: Self-Supervised Tool Learning

Toolformer's training pipeline operates in three stages for each tool:

### Stage 1: Generate Candidate API Call Annotations
Starting from a pre-trained language model (GPT-J in the paper), the model is prompted with a few (5-10) human-written examples of API calls embedded in text. The model then annotates a large unlabeled text corpus by generating candidate API call positions and arguments. For example, given text about the French Revolution, the model might insert `[QA("When did the French Revolution begin?") -> 1789]` at an appropriate position.

The API calls are represented as special tokens with a structured format:
```
[API_NAME(input) -> output]
```

The arrow token is a separator — the model generates the call, executes it against the real API, and the result is substituted.

### Stage 2: Execute and Filter
Each candidate API call is executed against the actual tool (search engine, calculator, calendar, etc.), and the returned result is substituted. A filtering step then determines whether the API call actually helped:

A **utility criterion** compares two perplexities of the text following the API call position:
- `L_with`: Loss of continuing text given the full API call and result
- `L_without`: Loss of continuing text with the API call removed

An API call is kept only if: `L_without - L_with > threshold`

This filters out calls where the retrieved information was not useful for predicting the surrounding text, retaining only genuinely informative API insertions.

### Stage 3: Fine-tune on Filtered Dataset
The language model is fine-tuned on the filtered, API-annotated corpus. After training, the model has learned to autonomously decide when to call each tool, construct appropriate API arguments, and incorporate returned results into its generation.

## Tools Implemented

Toolformer was trained to use five tools:

1. **Calculator:** Evaluates arithmetic expressions (`[Calculator(3*4+2) -> 14]`)
2. **Wikipedia Search:** Retrieves relevant Wikipedia passage summaries
3. **Machine Translation System:** Translates phrases between languages
4. **Question Answering System:** Answers factual questions (Atlas)
5. **Calendar:** Returns the current date

## Results

Toolformer was evaluated on a suite of downstream tasks:

- **LAMA (factual probing):** +8.9 percentage points over GPT-J baseline
- **Math benchmarks (ASDiv, SVAMP):** +40+ percentage points, fixing the arithmetic blindspot
- **Temporal tasks:** Calendar tool eliminates date-arithmetic errors
- **Question Answering (TriviaQA, WebQ, NQ):** Substantial improvement via search
- **Language modeling (WikiText):** Performance maintained — tool use does not degrade core language modeling

Crucially, Toolformer was evaluated **zero-shot** — no task-specific fine-tuning — demonstrating genuine generalization of tool-use behavior.

## Architecture and Scale

The backbone model was **GPT-J (6.7B parameters)**, a publicly available open-source model. Despite being far smaller than GPT-3 (175B), Toolformer-GPT-J outperformed GPT-3 on several benchmarks by leveraging tools.

## Significance

Toolformer made several lasting contributions:

1. **Self-supervised tool learning:** Demonstrated that tool-use training data can be generated autonomously with minimal human effort
2. **Utility-based filtering:** Provided a principled criterion for deciding when tool use adds value
3. **Generalization principle:** Tool use learned on one corpus generalizes to diverse downstream tasks
4. **Scale efficiency:** Smaller models with tools can exceed larger models without tools

The paper influenced the design of modern agent frameworks. The core API-call-in-text representation inspired subsequent work on ReAct, function-calling APIs in GPT-4 and Claude, and the tool-use capabilities built into frontier models. It demonstrated that tool integration is a learnable behavior, not just a hand-engineered scaffolding layer — a foundational insight for building autonomous AI agents.
