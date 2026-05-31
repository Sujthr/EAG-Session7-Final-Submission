# Voyager: An Open-Ended Embodied Agent with Large Language Models

## Paper Details

- **Title:** Voyager: An Open-Ended Embodied Agent with Large Language Models
- **Authors:** Guanzhi Wang, Yuqi Xie, Yunfan Jiang, Ajay Mandlekar, Chaowei Xiao, Yuke Zhu, Linxi Fan, Anima Anandkumar
- **Affiliation:** NVIDIA, Caltech, UT Austin, Stanford
- **Year:** 2023
- **Venue:** NeurIPS 2023 / arXiv

---

## Problem: Open-Ended Autonomous Learning in Complex Environments

Building autonomous agents that can continuously learn new skills in open-ended environments is a long-standing challenge in AI. Prior approaches to embodied agents relied heavily on task-specific reward functions, hand-crafted exploration heuristics, or imitation learning from human demonstrations — all of which limit generalization and autonomous skill acquisition.

Voyager addresses the question: **can an LLM-powered agent learn to play Minecraft indefinitely, acquiring increasingly complex skills autonomously, without human intervention?**

Minecraft serves as an ideal testbed because it is open-ended, has a complex technology tree (tools, resources, crafting) that requires sequential skill composition, and lacks a fixed "goal" — making it suitable for evaluating continual learning and exploration.

---

## System Architecture

Voyager consists of three key components that work in concert:

### 1. Automatic Curriculum

A curriculum generator, implemented as an LLM (GPT-4) prompted with the agent's current state, proposes the next skill to practice. The state includes:
- Current inventory (items held)
- Nearby environment description (biome, nearby blocks, mobs)
- Time of day, health, equipment
- Skills already mastered

Given this context, the LLM proposes a developmentally appropriate next goal — for example, "collect 10 wood logs" early on, progressing to "build a stone pickaxe" and eventually "explore a mine and collect iron ore" as the agent gains capabilities. This creates an automatic scaffold that mirrors how a human player would naturally progress.

### 2. Skill Library

A growing library of reusable skills is maintained as a key-value store:
- **Keys:** Natural language descriptions and embedding vectors of each skill
- **Values:** JavaScript code (using the Mineflayer API) that implements the skill in the Minecraft environment

When a new task arises, the agent retrieves the top-k most relevant existing skills using embedding-based similarity search, and includes them as context when generating the new skill. Successful new skills are added to the library, making the agent's capabilities monotonically non-decreasing.

Skills range from primitive actions (mine_block, craft_item) to complex compound behaviors (build_shelter, farm_crops), and the library grows to hundreds of skills over a long run.

### 3. Iterative Prompting Mechanism

Code generation for a new skill is not a one-shot process. Voyager uses an iterative refinement loop:

1. GPT-4 generates JavaScript code for the proposed skill
2. The code is executed in the Minecraft environment via Mineflayer (a JavaScript bot API)
3. Execution feedback — including error messages, stack traces, and game state changes — is fed back to the LLM
4. The LLM revises the code based on the feedback
5. This loop runs for up to 5 iterations, or until the task is completed successfully

This self-debugging loop is crucial: initial code generation often fails due to environmental constraints, inventory limitations, or logical errors. The iterative mechanism allows the agent to recover from failures autonomously.

---

## Results

Voyager was evaluated against several baselines including ReAct, Reflexion, and AutoGPT adapted to Minecraft, as well as an ablation of Voyager without each of its three components.

**Technology tree progression:** Voyager unlocked the full Minecraft technology tree significantly faster than all baselines. After 3 million environment steps, baselines had typically reached iron-level tools; Voyager reached diamond-level equipment (the hardest tier in vanilla Minecraft).

**Map coverage:** Voyager explored 2–3x more unique terrain (biomes, structures) compared to baselines, demonstrating superior exploration behavior.

**Skill library growth:** The skill library grew to 200+ skills over a full run, and crucially, these skills transferred successfully to new Minecraft worlds — demonstrating genuine generalization, not memorization.

**Zero-shot generalization:** When placed in a new world, Voyager could retrieve and apply skills from its library without additional interaction, outperforming baselines by a wide margin on novel tasks.

---

## Key Insights

**LLMs as world models for code generation:** Rather than generating low-level motor commands, Voyager leverages the LLM's existing knowledge of Minecraft mechanics (present in pre-training data) to write functional code. This bypasses the sample inefficiency of learning primitive skills from scratch.

**Skill compositionality:** The skill library enables compositional generalization — complex skills are built from simpler ones, mirroring how humans learn procedural knowledge.

**Exploration vs. exploitation via curriculum:** The automatic curriculum balances pushing toward new content (exploration) while consolidating mastered skills (exploitation), without requiring hand-designed reward shaping.

---

## Significance

Voyager is among the first systems to demonstrate **lifelong learning** in an embodied, open-ended environment using LLMs as the core reasoning and code-generation engine. It established a design pattern — curriculum + skill library + iterative refinement — that has influenced subsequent work on LLM-based autonomous agents (e.g., CodeAct, OpenDevin, SWE-agent). The demonstration that an LLM can autonomously grow its own skill library in an interactive environment represents a significant step toward self-improving agents.
