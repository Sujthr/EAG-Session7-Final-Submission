# Constitutional AI: Harmlessness from AI Feedback

**Authors:** Yuntao Bai, Saurav Kadavath, Sandipan Kundu, Amanda Askell, Jackson Kernion, Andy Jones, Anna Chen, Anna Goldie, Azalia Mirhoseini, Cameron McKinnon, Carol Chen, Catherine Olsson, Christopher Olah, Danny Hernandez, Dawn Drain, Deep Ganguli, Dustin Li, Eli Tran-Johnson, Ethan Perez, Jamie Kerr, Jared Mueller, Jeffrey Ladish, Joshua Landau, Kamal Ndousse, Kamile Lukosuite, Liane Lovitt, Michael Sellitto, Nelson Elhage, Nicholas Schiefer, Noemi Mercado, Nova DasSarma, Robert Lasenby, Robin Larson, Sam Ringer, Scott Johnston, Shauna Kravec, Sheer El Showk, Stanislav Fort, Tamera Lanham, Timothy Telleen-Lawton, Tom Conerly, Tom Henighan, Tristan Hume, Samuel R. Bowman, Zac Hatfield-Dodds, Ben Mann, Dario Amodei, Nicholas Joseph, Sam McCandlish, Tom Brown, Jared Kaplan (Anthropic)
**Year:** 2022

---

## Overview

Constitutional AI (CAI) is Anthropic's methodology for training AI assistants to be both helpful and harmless using a set of explicit natural language principles (a "constitution") and AI-generated feedback, rather than relying solely on human labelers to evaluate harmful content. CAI reduces dependence on human annotation of sensitive material while producing models that are more consistently non-harmful, more transparent in their principles, and less prone to the sycophantic helpfulness that can arise from pure RLHF.

## Motivation

Standard RLHF alignment faces several practical challenges:

- **Labeler burden:** Human labelers must repeatedly evaluate harmful, offensive, or disturbing content — creating psychological harm and annotation inconsistency
- **Limited scalability:** Human feedback is expensive and slow; alignment bottlenecks model development
- **Implicit principles:** RLHF encodes human preferences implicitly in reward model weights, making alignment criteria opaque and hard to audit
- **Helpfulness-harmlessness tension:** Models trained to maximize human approval can become sycophantic — refusing too broadly to avoid any perceived harm, or agreeing with false premises

Constitutional AI addresses these issues by making alignment principles explicit and using AI to apply them at scale.

## The Constitution

The "constitution" is a set of **natural language principles** that define the desired behavior of the AI. Anthropic's constitution includes principles drawn from multiple sources:

- The UN Declaration of Human Rights
- Apple's terms of service (as an example of deployed AI guidelines)
- DeepMind's Sparrow rules
- Principles authored by Anthropic researchers

Example principles include:
- *"Choose the response that is least likely to contain harmful or unethical content."*
- *"Which response is more honest, and doesn't just say what the user wants to hear?"*
- *"Choose the response that a thoughtful, senior Anthropic employee would consider optimal."*

The constitution is modular and editable — different principles can be swapped in to target different alignment objectives.

## Two-Phase Training Pipeline

### Phase 1: Supervised Learning from AI Feedback (SL-CAI)

**Step 1 — Red-teaming and critique generation:**
The initial helpful-only model is prompted with harmful instructions to elicit potentially harmful responses. These responses are then critiqued by the same model using a principle sampled from the constitution:

*"Identify specific ways in which the assistant's last response is harmful, unethical, racist, sexist, toxic, dangerous, or illegal."*

**Step 2 — Revision:**
Using the critique, the model revises its response:

*"Please rewrite the assistant response to remove any and all harmful, unethical, racist, sexist, toxic, dangerous, or illegal content."*

This critique-revision cycle can be applied **multiple times** iteratively, with different principles sampled each time, progressively making responses more aligned. The final revised responses form a supervised fine-tuning dataset.

### Phase 2: Reinforcement Learning from AI Feedback (RLAIF)

**Step 1 — AI preference labeling:**
Instead of human labelers evaluating response pairs for harmfulness, a feedback model (prompted with constitution principles) evaluates pairs of AI-generated responses:

*"Consider the following conversation and a pair of possible responses. Which response is more harmless? State your choice as 'Response A' or 'Response B'."*

This generates a large dataset of AI-labeled preferences across harmlessness dimensions.

**Step 2 — Reward model training:**
A harmlessness reward model is trained on the AI-labeled preference data using standard pairwise ranking loss.

**Step 3 — PPO fine-tuning:**
The SL-CAI model is further optimized via PPO using the AI-trained reward model, exactly as in standard RLHF — but without requiring human labelers to evaluate harmful content.

## Results

- **Harmlessness:** CAI models were substantially preferred over RLHF models by human raters on harmlessness dimensions, particularly for requests involving sensitive topics, dangerous information, and manipulation
- **Helpfulness maintained:** Unlike pure harmlessness training, CAI models maintained helpfulness scores because the constitution explicitly balances both objectives
- **Non-evasiveness:** CAI models were less likely to refuse legitimate requests due to overly conservative harm avoidance
- **Transparency:** Alignment criteria are human-readable and auditable, unlike reward model weights
- **Scaling:** AI feedback allows generating preference labels at 10-100x the rate of human labeling

## Chain-of-Thought Critique

A key extension in the paper is **chain-of-thought critique**: the feedback model is asked to reason step-by-step about which response better satisfies a principle before giving its final judgment. This substantially improved the reliability and consistency of AI-generated preference labels.

## Significance

Constitutional AI established several important contributions to AI alignment:

1. **RLAIF as viable alternative to RLHF:** AI feedback can substitute for human feedback at scale without substantially degrading quality
2. **Explicit, auditable principles:** Alignment criteria need not be buried in reward model parameters
3. **Reduced labeler harm:** Human annotators no longer need to engage with the most offensive content
4. **Claude's foundation:** CAI is the core methodology underlying all Claude models deployed by Anthropic
5. **Scalable oversight:** CAI is an early example of using AI to assist in its own alignment — a key component of the scalable oversight research agenda

The CAI framework directly influenced subsequent work on AI-generated preference data (RLAIF), principle-following LLMs, and model spec / system card approaches to explicit behavioral specification across the industry.
