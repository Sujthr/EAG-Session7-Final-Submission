# Reflexion: Language Agents with Verbal Reinforcement Learning

**Authors:** Noah Shinn, Federico Cassano, Edward Berman, Ashwin Gopinath, Karthik Narasimhan, Shunyu Yao (Princeton University / MIT)
**Year:** 2023 (arXiv March 2023, NeurIPS 2023)

---

## Overview

Reflexion introduces a novel framework for training language agents to improve through verbal self-reflection rather than gradient-based learning. Traditional reinforcement learning requires scalar reward signals and many interaction episodes to update model weights. Reflexion sidesteps this by converting environmental feedback into natural language self-evaluations stored in an episodic memory buffer, allowing agents to learn from mistakes within a single session without any parameter updates.

## Motivation

Language model agents deployed on sequential decision-making tasks face a fundamental problem: when they fail, standard approaches either require expensive model retraining or simply retry the same strategy, leading to repetitive failures. Human experts, by contrast, reflect on mistakes — analyzing what went wrong, forming hypotheses about better strategies, and consciously applying those insights in subsequent attempts.

Reflexion operationalizes this human-like reflective learning loop for language agents:

- No gradient updates to model weights
- No large labeled datasets of correct trajectories
- Works with any frozen language model capable of self-evaluation
- Compresses experience into interpretable natural language memories

## Architecture

The Reflexion framework consists of three core components:

### 1. Actor
The Actor is a language model that generates actions and text given the current observation and context. It operates as the policy, mapping the current state (observation + memory) to the next action. The Actor can be any capable LLM (GPT-4, GPT-3.5-turbo, etc.) used with chain-of-thought prompting.

### 2. Evaluator
The Evaluator produces a scalar or binary reward signal given the actor's trajectory. Depending on the task domain, this can be:
- An external environment (code execution result, game score)
- An LLM-based judge scoring the response quality
- A heuristic function over the final state

### 3. Self-Reflection Model
The Self-Reflection Model is also an LLM (potentially the same as the Actor) that takes the action trajectory, final outcome, and current reflection as input and produces a **verbal reflection** — a natural language analysis of what went wrong and what should be done differently. This reflection is stored in a sliding-window episodic memory.

### Memory Structure

Reflexion maintains two types of memory:
- **Short-term context window:** The current episode's trajectory (observations, actions, rewards)
- **Long-term verbal memory:** A collection of reflections from previous failed episodes, prepended to future episodes as context

At the start of each new attempt, the agent sees its previous reflections and is thereby "reminded" of its mistakes and hypothesized corrections.

## Task Domains Evaluated

Reflexion was evaluated across three distinct task families:

### Sequential Decision-Making: AlfWorld
AlfWorld is a text-based household environment where agents complete multi-step tasks (e.g., "put a heated egg in the fridge"). Reflexion achieved **97% success rate** after multiple reflection cycles, compared to **53% for ReAct** (the baseline without reflection) and significantly better than standard prompting approaches.

### Code Generation: HumanEval and MBPP
For Python programming tasks, the agent generates code, executes it against test cases, and reflects on failing tests. Reflexion-GPT-4 achieved **91% pass@1 on HumanEval** — a new state-of-the-art at the time of publication — compared to 80% for GPT-4 without reflection. Gains were especially large on problems requiring complex algorithmic reasoning.

### Question Answering: HotpotQA
On multi-hop reasoning tasks, Reflexion improved accuracy by allowing the agent to reflect on failed retrieval steps and revise its search strategy. Gains of 14+ percentage points were observed over baseline ReAct.

## The Reflection Process in Practice

A typical Reflexion episode looks like:
1. Agent attempts task, fails (e.g., wrong answer, code fails tests)
2. Evaluator assigns failure signal
3. Reflection model generates: *"I attempted to find the capital city, but searched for the country name instead of the specific region. Next time I should first identify the precise sub-region before searching."*
4. This reflection is stored in memory
5. Next episode begins with reflection prepended — agent adjusts behavior accordingly

## Limitations

- **Memory overflow:** Performance can degrade if too many reflections accumulate, requiring careful window management
- **Hallucinated reflections:** If the LLM's self-evaluation is incorrect, it can reinforce wrong strategies
- **No genuine learning:** Improvements disappear across independent sessions — there is no persistent weight update
- **Dependency on model capability:** Reflection quality is bounded by the base model's self-evaluation ability

## Significance

Reflexion demonstrated that verbal self-reflection is a viable and practically effective form of agent learning, establishing several important principles:

1. **Language as reinforcement signal:** Natural language can replace scalar rewards for learning purposes
2. **In-context learning as optimization:** Iterative prompting with accumulated memory can substitute for gradient-based RL in many practical settings
3. **Interpretability:** Unlike neural RL, Reflexion's learning process is fully auditable as natural language

The framework influenced subsequent agent architectures including Self-Refine, CRITIC, and many production agent systems that incorporate retry-with-reflection loops. It represents a practical and immediately deployable approach to building self-improving AI agents using existing frontier models.
