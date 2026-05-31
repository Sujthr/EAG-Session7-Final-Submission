# LLM-Powered Autonomous Agents: Architectures, Memory, Planning, and Tool Use

## Overview

**Authors/Survey:** Weng, Lilian (2023) — "LLM-powered Autonomous Agents" (Lil'Log blog, widely cited); Wang et al. (2023) — "A Survey on Large Language Model based Autonomous Agents" (arXiv:2308.11432)

An LLM-powered autonomous agent is a system in which a large language model serves as the central reasoning engine, directing its own actions across multiple steps to accomplish complex, open-ended goals. Unlike a single-turn LLM query, agents operate in loops: observe the environment, reason about the current state, decide on an action, execute the action through tools, observe results, and repeat until the goal is achieved. This paradigm has shifted LLMs from passive text generators to active decision-making systems capable of multi-step task execution.

---

## Core Agent Architecture

A canonical LLM agent system has four primary components:

### 1. Planning

The agent must decompose complex goals into manageable sub-tasks and decide on the sequence of actions to take. Key planning strategies include:

**Task Decomposition:**
- **Chain-of-Thought (CoT):** Sequential step-by-step reasoning within a single LLM call
- **Tree of Thoughts (ToT):** Explores multiple reasoning branches simultaneously, enabling backtracking
- **LLM+P:** Uses classical planning languages (PDDL) with an external planner for structured domains

**Self-Reflection and Refinement:**
- **ReAct (Yao et al., 2022):** Interleaves reasoning traces ("Thought") with executable actions and observations in a tight loop. The model reasons about what to do, takes an action (API call, search), and incorporates the observation into the next reasoning step.
- **Reflexion (Shinn et al., 2023):** After task failure, the agent generates a verbal self-reflection critiquing what went wrong and stores it in episodic memory. Subsequent attempts condition on this reflection, enabling trial-and-error learning without weight updates.

---

### 2. Memory

Agents require persistent information storage across reasoning steps and across sessions. The literature distinguishes four memory types:

**Sensory Memory:** The raw input stream — text, images, tool outputs — presented in the current context window. Analogous to human sensory buffer; highly transient.

**In-Context / Working Memory:** The active context window of the LLM (typically 8K–128K tokens). All current observations, tool results, and intermediate reasoning steps reside here. Limited by context length; oldest information is "forgotten" as context overflows.

**External Memory (Long-term):** A vector database or key-value store that persists information across sessions. At each step, relevant memories are retrieved via semantic search and injected into the context. Enables effectively unlimited long-term retention.

**Parametric Memory:** Knowledge encoded in the LLM's weights during pretraining. Always available but static — cannot be updated without retraining.

**MemGPT (Packer et al., 2023):** Proposes an OS-inspired memory management system where the LLM explicitly manages its own context, deciding what to page in/out of working memory from an external memory store, enabling unbounded long-context operation.

---

### 3. Tool Use

Agents extend their capabilities by calling external tools — APIs, databases, code interpreters, browsers, and specialized models. Tool use addresses fundamental LLM limitations:

- **Factuality:** Web search tools (Bing, Google) provide up-to-date, verifiable information
- **Computation:** Code execution (Python interpreter, calculator) provides exact arithmetic and symbolic computation
- **Action:** APIs (email, calendar, file system, browser) enable real-world effects

**Toolformer (Schick et al., 2023):** Fine-tunes LLMs to autonomously learn when and how to call APIs by training on self-generated API call annotations. The model learns to insert API calls inline in text generation.

**Function Calling:** OpenAI's function calling API (2023) provides a structured mechanism for LLMs to output structured JSON specifying which tool to call and with what parameters, making tool integration significantly more reliable than text-parsing approaches.

---

### 4. Action Space

Agent actions span a spectrum from "read-only" to "write/execute":

- **Information retrieval:** Web search, document lookup, database queries
- **Code generation and execution:** Producing and running Python/bash in a sandboxed interpreter
- **Environment interaction:** Clicking, form-filling, navigation in a browser (WebGPT, WebAgent)
- **Agent orchestration:** Spawning sub-agents for parallel task execution
- **Memory operations:** Writing to or querying external memory stores

---

## Notable Agent Systems

### AutoGPT (2023)

AutoGPT was one of the first widely publicized autonomous agent implementations. Given a high-level goal, it autonomously plans tasks, uses web search, file operations, and code execution to complete them, and self-evaluates progress. AutoGPT demonstrated the viability of autonomous LLM agents to a broad audience but also exposed core reliability challenges: agents frequently enter planning loops, lose track of goals, or exhaust token budgets on irrelevant sub-tasks.

### BabyAGI (Nakajima, 2023)

A minimal agent that maintains a task queue. The LLM creates tasks, prioritizes them, executes them using tools, and adds new tasks based on results. BabyAGI demonstrated the self-organizing task management pattern that underlies many production agent systems.

### HuggingGPT / JARVIS (Shen et al., 2023)

Uses ChatGPT as a task planner that decomposes user requests into sub-tasks, routes each sub-task to the most appropriate specialist model on HuggingFace (image captioning, object detection, translation, etc.), collects outputs, and synthesizes a final response. Illustrates the "LLM as orchestrator" pattern for multi-modal, multi-model pipelines.

### Voyager (Wang et al., 2023)

A Minecraft agent that autonomously discovers skills, writes reusable code for each skill, stores them in a code library, and retrieves them by description for future tasks. Voyager demonstrated emergent curriculum learning and skill generalization without human supervision.

---

## Multi-Agent Systems

Single agents struggle with tasks requiring diverse expertise, parallelism, or long-horizon planning. Multi-agent architectures assign specialized roles:

**MetaGPT (Hong et al., 2023):** Encodes software engineering workflows (product manager, architect, engineer, QA) as separate agents that communicate via structured documents, mimicking human team workflows for software development.

**AutoGen (Wu et al., 2023, Microsoft):** A framework for building multi-agent conversations where agents with different configurations (tools, personas, human proxies) collaborate through structured dialogue to complete tasks.

---

## Key Challenges

**Reliability and Hallucination:** Agents must execute dozens of sequential decisions; errors compound. A single hallucinated tool call or incorrect reasoning step can derail an entire workflow.

**Context Length Management:** Long agent sessions overflow context windows. Summarization, memory retrieval, and context compression are active research areas.

**Planning Depth:** Current LLMs struggle with tasks requiring 10+ sequential reasoning steps. Error recovery and replanning remain weak compared to classical planners.

**Safety and Alignment:** Autonomous agents with internet access and file system permissions can cause irreversible harm. Sandboxing, permission systems, and human-in-the-loop checkpoints are essential.

---

## Significance

LLM agents represent the frontier of AI capability deployment — moving from answering questions to autonomously completing complex, multi-step real-world tasks. The ReAct, Reflexion, and tool-use paradigms have become standard components of modern AI application stacks. As agent reliability improves, the boundary between "AI assistant" and "AI colleague" continues to blur.
