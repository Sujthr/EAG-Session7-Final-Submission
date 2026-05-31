# Proximal Policy Optimization (PPO): Algorithm, Theory, and RLHF Applications

## Overview

**Authors:** John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov (OpenAI)  
**Paper:** "Proximal Policy Optimization Algorithms"  
**Published:** arXiv:1707.06347 (2017)

Proximal Policy Optimization (PPO) is a family of policy gradient reinforcement learning algorithms that achieve strong performance across a wide range of tasks while being significantly simpler to implement and tune than its predecessor, Trust Region Policy Optimization (TRPO). PPO has become the dominant RL algorithm in robotics, game playing, and — most consequentially — Reinforcement Learning from Human Feedback (RLHF) for aligning large language models.

---

## Background: Policy Gradient Methods

In reinforcement learning, a policy π_θ(a|s) (parameterized by θ) maps states to action probability distributions. The goal is to maximize expected cumulative reward:

```
J(θ) = E_{τ~π_θ}[Σ_t r_t]
```

**Vanilla Policy Gradient (REINFORCE):** Computes gradient estimates as:
```
∇_θ J(θ) = E_t[∇_θ log π_θ(a_t|s_t) · A_t]
```

where A_t is the advantage estimate (how much better action a_t was compared to the average). While unbiased, vanilla PG suffers from high variance and is sensitive to step size — too large a gradient update can collapse the policy catastrophically.

---

## Trust Region Policy Optimization (TRPO)

**Authors:** Schulman et al. (2015)

TRPO solved the step-size problem by constraining policy updates to stay within a "trust region" defined by a KL divergence constraint:

```
maximize: L(θ)
subject to: E_t[KL[π_θ_old(·|s_t) || π_θ(·|s_t)]] ≤ δ
```

TRPO guarantees monotonic policy improvement but requires computing the Fisher Information Matrix and solving a constrained optimization problem at each step — computationally expensive and implementation-complex due to conjugate gradient and line search procedures.

---

## PPO: The Core Algorithm

PPO achieves similar stability to TRPO while replacing the hard KL constraint with a simpler clipped surrogate objective.

### Probability Ratio

Define the probability ratio:
```
r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)
```

When r_t = 1, the new policy equals the old. When r_t > 1, the action is more likely under the new policy.

### Clipped Surrogate Objective (PPO-Clip)

```
L^CLIP(θ) = E_t[min(r_t(θ) A_t, clip(r_t(θ), 1-ε, 1+ε) A_t)]
```

The clip operation constrains r_t to the interval [1-ε, 1+ε] (typically ε = 0.2). The minimum prevents the objective from benefiting from policy changes larger than this range:

- If A_t > 0 (action was better than average): the objective is maximized but capped at (1+ε)·A_t — no reward for moving the policy too far in the positive direction
- If A_t < 0 (action was worse than average): the objective is minimized but floored at (1-ε)·A_t — no reward for overcorrecting

This creates a pessimistic lower bound on policy improvement without requiring constrained optimization.

### PPO-KL Penalty (Alternative Formulation)

```
L^KL(θ) = E_t[r_t(θ) A_t] - β · KL[π_θ_old || π_θ]
```

The KL penalty coefficient β is adaptively adjusted: increased if the KL divergence exceeds a target value d_target, decreased if below. PPO-Clip is generally preferred due to its robustness to hyperparameter choice.

---

## Actor-Critic Architecture

PPO uses an **actor-critic** architecture where:

- **Actor:** The policy network π_θ(a|s) that selects actions
- **Critic:** A value function V_φ(s) that estimates the expected return from state s, used to compute advantage estimates

**Generalized Advantage Estimation (GAE, Schulman et al. 2016):**
```
A_t = Σ_{l=0}^{∞} (γλ)^l δ_{t+l}
```
where δ_t = r_t + γV(s_{t+1}) - V(s_t) is the TD residual. The λ parameter interpolates between high-bias/low-variance (λ=0, pure TD) and low-bias/high-variance (λ=1, Monte Carlo) advantage estimates.

The full PPO loss combines policy, value, and entropy terms:
```
L_total = L^CLIP - c_1 · L^VF + c_2 · S[π_θ](s_t)
```

where L^VF is the value function MSE loss and S is an entropy bonus that encourages exploration.

---

## Training Procedure

1. Collect T timesteps of experience under the current policy π_θ_old
2. Compute advantages A_t using GAE
3. Perform K epochs of minibatch gradient updates on the collected data using L^CLIP
4. Update θ_old ← θ
5. Repeat

The ability to perform multiple gradient updates on each batch of experience (via the clipping mechanism that prevents overfitting) gives PPO substantially better sample efficiency than vanilla policy gradient.

---

## Results

PPO achieved state-of-the-art performance on:
- **MuJoCo continuous control:** Ant, Hopper, Walker2d locomotion tasks — matching or exceeding TRPO with 3–10x fewer compute operations
- **Atari games:** Competitive with A3C using similar compute budget
- **Dexterous robotics:** OpenAI's robotic hand solving a Rubik's cube used PPO

PPO's combination of simplicity, stability, and scalability made it the default RL algorithm in OpenAI's research stack.

---

## PPO in RLHF for LLM Alignment

PPO is the RL algorithm underlying **Reinforcement Learning from Human Feedback (RLHF)**, the training procedure used to align ChatGPT, InstructGPT, Claude, and other instruction-following LLMs.

### RLHF Pipeline

**Step 1 — Supervised Fine-Tuning (SFT):** Fine-tune the base LLM on human-written demonstrations of desired behavior.

**Step 2 — Reward Model Training:** Collect human preference data: for each prompt, humans compare pairs of model responses and indicate which is better. Train a reward model R_φ(prompt, response) → scalar to predict human preferences using the Bradley-Terry model.

**Step 3 — PPO Fine-Tuning:** Use PPO to optimize the policy (the LLM) to maximize the reward model's score, subject to a KL penalty against the SFT model:

```
reward = R_φ(prompt, response) - β · KL[π_θ(response|prompt) || π_SFT(response|prompt)]
```

The KL penalty prevents the policy from "reward hacking" — exploiting idiosyncrasies in the reward model by drifting far from the SFT distribution. Without it, LLMs quickly collapse to repetitive or degenerate outputs that maximize the imperfect reward model.

**Challenges in LLM PPO:**
- The action space is the full vocabulary (50K+ tokens); each token generation is an RL action
- Episodes are variable-length (full response generation)
- Reward is only observed at the end of a response (sparse reward)
- The reward model itself may be misspecified (Goodhart's Law)

**RLHF Alternatives:**
- **DPO (Direct Preference Optimization, Rafailov et al. 2023):** Reformulates RLHF to directly optimize on preference data without an explicit reward model or RL loop — simpler and more stable, increasingly preferred for smaller-scale alignment
- **REINFORCE with baseline:** Simpler than PPO, used in some LLM alignment pipelines (GRPO used in DeepSeek-R1)

---

## Significance

PPO's elegant combination of empirical effectiveness and implementation simplicity made it the go-to policy optimization algorithm for a decade of RL research. Its application to RLHF enabled the development of instruction-following LLMs that are safe, helpful, and aligned with human preferences — a breakthrough that defined the modern era of conversational AI. ChatGPT's success is in large part the story of PPO applied at unprecedented scale.
