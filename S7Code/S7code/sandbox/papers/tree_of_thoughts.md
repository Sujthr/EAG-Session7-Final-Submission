# Tree of Thoughts: Deliberate Problem Solving with Large Language Models

**Authors:** Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Thomas L. Griffiths, Yuan Cao, Karthik Narasimhan (Princeton University / Google DeepMind)
**Year:** 2023 (arXiv May 2023, NeurIPS 2023)

---

## Overview

Tree of Thoughts (ToT) generalizes chain-of-thought prompting into a structured search framework that allows language models to explore multiple reasoning paths simultaneously, evaluate intermediate progress, and backtrack when a path is unpromising. While chain-of-thought prompting improved LLM reasoning by making intermediate steps explicit, it still committed to a single left-to-right reasoning trace. ToT introduces **deliberate search over a tree of partial solutions**, enabling LLMs to tackle tasks that require lookahead, exploration, and strategic backtracking.

## Motivation

Standard language model decoding and chain-of-thought prompting share a critical limitation: they are **serial and greedy**. At each step, the model commits to a single continuation, with no mechanism to explore alternatives or evaluate whether the current path will lead to a solution. This contrasts sharply with how humans approach difficult problems:

- We consider multiple approaches before committing
- We evaluate partial progress against solution criteria
- We abandon unpromising lines of thought and try alternatives
- We plan ahead rather than just extending the most recent step

These capabilities — exploration, evaluation, and backtracking — are precisely what ToT provides. The framework draws explicit connections to classical AI search algorithms (BFS, DFS, A*) and psychological theories of dual-process cognition (System 1 vs. System 2 thinking).

## Framework Components

ToT decomposes deliberate problem solving into four modular components:

### 1. Thought Decomposition
The problem is divided into intermediate steps, where each step is a "thought" — a coherent unit of text that represents partial progress. The granularity of thoughts is task-dependent:
- For creative writing: a paragraph or plot outline
- For mathematical proofs: a single reasoning step
- For Game of 24: selecting which numbers and operator to apply next

### 2. Thought Generation
At each node in the tree, thoughts are generated either by:
- **Sampling independently:** Generate k thoughts from the current state via temperature sampling
- **Proposing sequentially:** Generate k candidates in a single prompt that lists multiple options

### 3. State Evaluation
Each thought (or thought sequence) is evaluated to estimate its likelihood of leading to a correct solution. Evaluation can be performed by:
- **Value estimation:** The LLM assigns a scalar score (sure/likely/impossible) to each partial state
- **Vote across samples:** Multiple LLM evaluations are aggregated by majority vote

The evaluator uses the same LLM as the generator, prompted with evaluation instructions rather than generation instructions.

### 4. Search Algorithm
Standard tree search algorithms are applied over the thought tree:
- **Breadth-First Search (BFS):** Maintain a fixed number of most promising states at each depth level
- **Depth-First Search (DFS):** Explore one path fully before backtracking, with pruning based on value estimates

The search breadth `b` and depth `d` are task-specific hyperparameters controlling the exploration budget.

## Tasks and Results

### Game of 24 (Mathematical Reasoning)
Given four numbers and standard arithmetic operators, reach exactly 24 using each number once. This requires precise multi-step planning.

- **Standard GPT-4 (input-output prompting):** 7.3% success
- **Chain-of-thought (GPT-4):** 4.0% success
- **Tree of Thoughts (BFS, b=5):** **74% success**

ToT nearly eliminates failures by systematically evaluating intermediate arithmetic states and backtracking from dead ends.

### Creative Writing: 24-paragraph Story
Generate a coherent story that naturally incorporates four random sentences as the closing line of four paragraphs.

- **Chain-of-thought:** Coherence score 6.19/10 (human eval)
- **Tree of Thoughts:** Coherence score **7.56/10**
- Human writers preferred ToT outputs in blind evaluation

### Mini Crossword Solving
Fill a 5x5 crossword grid given clues, requiring simultaneous satisfaction of horizontal and vertical constraints.

- **Standard prompting:** 16% word-level accuracy
- **Chain-of-thought:** 20% word-level accuracy
- **Tree of Thoughts (DFS):** **60% word-level accuracy**

## Relationship to Prior Work

ToT synthesizes ideas from multiple traditions:

- **Classical AI planning:** MCTS, A*, alpha-beta search — ToT brings search to neural language generation
- **Chain-of-thought prompting (Wei et al., 2022):** ToT generalizes CoT from a single path to a tree
- **Self-consistency (Wang et al., 2022):** ToT extends majority voting from final answers to intermediate reasoning steps
- **NeuroSymbolic AI:** ToT combines learned language representations with structured symbolic search

## Limitations and Considerations

- **Computational cost:** ToT requires many more LLM calls per problem than greedy decoding (often 10-100x)
- **Task-specific design:** Thought decomposition and evaluation prompts require manual engineering for each domain
- **Evaluation reliability:** The LLM evaluator can be miscalibrated, leading to poor pruning decisions

## Significance

Tree of Thoughts demonstrated that the missing ingredient in LLM reasoning was not model capacity but **search structure**. By providing a principled framework for deliberate exploration, ToT enabled frontier models to solve problems previously considered out of reach. The work connects modern LLMs to the rich tradition of AI planning and search, suggesting that hybrid neural-symbolic approaches are essential for robust reasoning. ToT directly influenced subsequent methods including Graph of Thoughts, Cumulative Reasoning, and reasoning-focused models like OpenAI's o1/o3 series, which internalize search through extended chain-of-thought generation.
