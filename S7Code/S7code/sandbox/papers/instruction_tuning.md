# Instruction Tuning: Finetuned Language Models are Zero-Shot Learners

## Key Papers

- **Finetuned Language Models are Zero-Shot Learners (FLAN)** — Jason Wei, Maarten Bosma, Vincent Zhao, Kelvin Guu, Adams Wei Yu, Brian Lester, Nan Du, Andrew M. Dai, Quoc V. Le — Google Research, 2021 (ICLR 2022)
- **LIMA: Less Is More for Alignment** — Chunting Zhou, Pengfei Liu, Puxin Xu, Srini Iyer, Jian Hou, Yangsibo Huang, Xilun Chen, et al. — Meta AI, 2023

---

## The Core Problem

Pretrained language models are trained to predict the next token given a context — an objective that does not naturally align with how users want to interact with them. A language model may "know" how to translate French to English but fail to do so when prompted as "Translate the following to English: ...", because this framing differs from patterns seen in pre-training.

**Instruction tuning** addresses this by fine-tuning a pretrained model on a diverse collection of tasks expressed as natural language instructions with expected outputs. The hypothesis is that training on enough instructed examples enables the model to **generalize to new, unseen tasks** when given instructions at inference time — zero-shot task generalization.

---

## FLAN (2021) — Foundational Instruction Tuning

### Method

The FLAN paper collected 62 NLP datasets spanning diverse task categories: natural language inference, reading comprehension, closed-book QA, translation, commonsense reasoning, coreference resolution, and more. Each dataset was reformulated into 10 unique instruction templates using manual prompting.

This yielded over 60 task clusters with multiple phrasings per task. The model (a 137B parameter LaMDA-style decoder-only language model) was fine-tuned on all tasks **except the target evaluation task**, then evaluated zero-shot on held-out tasks.

The key experimental design — **leave-one-out evaluation** — ensures that zero-shot performance reflects genuine generalization to new task types, not just improved performance on seen tasks.

### Results

- FLAN significantly outperformed zero-shot GPT-3 (175B) on 20 of 25 evaluated datasets
- FLAN matched or exceeded few-shot GPT-3 on many tasks, despite using no in-context examples at inference time
- Instruction tuning was most effective for tasks that benefit from understanding natural language task descriptions: NLI, reading comprehension, closed-book QA
- Tasks with simple output spaces (single tokens, classification) benefited most; generation tasks showed smaller improvements

### Key Findings

- Scale matters: instruction tuning improved large models (100B+) dramatically but showed minimal benefit for smaller models (<8B), suggesting a capability threshold
- Cluster diversity matters: more task types (not just more data from the same tasks) drove better zero-shot generalization
- Instruction phrasing diversity helps: multiple templates per task improved robustness

### FLAN-T5 and Instruction Tuning at Scale (Follow-up)

Subsequent work scaled instruction tuning significantly. FLAN-T5 and FLAN-PaLM used 1,800+ tasks across dozens of task clusters, demonstrating that scaling the instruction tuning mixture continued to improve zero-shot and few-shot performance well beyond the original 62-dataset setup.

---

## LIMA (2023) — Less Is More for Alignment

### Motivation and Hypothesis

LIMA ("Less Is More for Alignment") challenges the prevailing assumption that instruction tuning requires massive datasets with tens of thousands or millions of examples. The central hypothesis is the **Superficial Alignment Hypothesis**: a model's knowledge and capabilities are almost entirely determined by pre-training, while alignment (the ability to follow instructions in the expected format) is a relatively shallow skill that can be learned from very few examples.

### Dataset and Method

LIMA fine-tuned LLaMA-65B on exactly **1,000 carefully curated examples** drawn from:
- Stack Exchange (high-quality Q&A from community-voted posts)
- wikiHow articles
- Manually authored examples

All examples were selected or written to be high quality, diverse in task type and domain, and stylistically consistent. No automated data generation was used; quality was prioritized over quantity.

Fine-tuning used standard supervised fine-tuning (SFT) on these 1,000 examples with no RLHF or additional alignment techniques.

### Results

- LIMA produced outputs preferred by human raters over GPT-4 in 19% of comparisons, over Bard in 29%, and over DaVinci003 in 58% — despite using far fewer training examples than any competing system
- On the GPT-4-evaluated OpenLLM benchmark, LIMA performed comparably to Alpaca (fine-tuned on 52K examples) and Vicuna (fine-tuned on 70K examples)
- LIMA showed strong generalization to test prompts significantly different in style and content from training examples

### Key Findings

- **Quality >> quantity:** Carefully curated examples dramatically outperformed noisily collected large datasets
- **Format consistency matters:** Examples that followed a consistent assistant-style response format improved model outputs more than diverse but inconsistent formats
- **Failure mode:** LIMA showed weaknesses in multi-turn dialogue (designed for single-turn), and occasionally gave incorrect information confidently — limitations not fully addressed by alignment alone

---

## Significance of Instruction Tuning

Instruction tuning is the standard recipe for transforming a pretrained base LLM into a usable assistant. Every major deployed LLM (ChatGPT, Claude, Gemini, LLaMA-Chat) uses some form of instruction tuning as a critical step.

FLAN established that instruction tuning enables zero-shot task generalization at scale. LIMA provided evidence that the data requirement for effective alignment may be far smaller than believed, democratizing alignment research and suggesting that data quality and curation are the critical bottlenecks — not dataset size.

Together, these papers define the modern understanding of instruction tuning as a bridge between raw capability (acquired in pre-training) and usable behavior (acquired through targeted fine-tuning on diverse, high-quality instructed examples).
