# Agentic RAG: Combining LLM Agents with Retrieval-Augmented Generation

**Key Works:**
- **RAG** — Lewis et al., Meta AI / UCL, 2020 (arXiv:2005.11401)
- **Self-RAG** — Asai et al., University of Washington, 2023 (arXiv:2310.11511)
- **CRAG (Corrective RAG)** — Yan et al., 2024 (arXiv:2401.15884)
- **IRCoT** — Trivedi et al., 2022 (arXiv:2212.10509)
- **ReAct** — Yao et al., 2022 (arXiv:2210.03629)

---

## What is Agentic RAG?

Standard Retrieval-Augmented Generation (RAG) is a pipeline: retrieve relevant documents once, append them to the context, generate a response. This works well for simple factoid questions but fails for complex queries requiring multi-step reasoning, iterative clarification, or synthesis across multiple sources. **Agentic RAG** extends standard RAG by giving the language model agency over the retrieval process itself — deciding when to retrieve, what to retrieve, whether retrieved information is sufficient, and how to iteratively build toward an answer.

The distinction is architectural: in standard RAG, retrieval is a fixed preprocessing step outside the model's control. In agentic RAG, retrieval is a tool the model invokes dynamically, interleaved with reasoning, and the model can critique, correct, and iterate on its own retrieval decisions.

---

## Standard RAG: Foundation and Limitations

Lewis et al. (2020) introduced RAG as a hybrid of parametric memory (model weights) and non-parametric memory (a retrieved document store). A query is encoded, a dense retriever (DPR) fetches top-k documents from a corpus indexed with FAISS, and a seq2seq model (BART) generates the answer conditioned on the retrieved passages.

**Limitations of standard RAG:**
- **Single retrieval step:** One retrieval cannot always surface all necessary information for complex, multi-hop questions.
- **No quality assessment:** The model cannot tell if retrieved documents are actually relevant to the question.
- **Fixed retrieval:** The model cannot refine its query based on what it finds.
- **Hallucination risk:** The model may generate plausible-sounding content not grounded in retrieved documents.

---

## Multi-Hop Retrieval and Iterative Retrieval

### IRCoT (Interleaving Retrieval with Chain-of-Thought)

IRCoT (Trivedi et al., 2022) interleaves retrieval steps with chain-of-thought reasoning. The model reasons one step at a time; each reasoning step generates a sub-question or intermediate conclusion, which is used as the query for the next retrieval step. This continues until the model produces a final answer.

**Procedure:**
1. Generate first reasoning step from the original question.
2. Use the reasoning step as a retrieval query; retrieve top-k passages.
3. Append retrieved passages to the context; generate next reasoning step.
4. Repeat until the reasoning chain produces an answer.

On multi-hop QA benchmarks (MuSiQue, HotpotQA, 2WikiMultiHopQA), IRCoT substantially outperforms single-step RAG, demonstrating that iterative retrieval is essential for questions requiring reasoning chains across multiple documents.

---

## ReAct: Reason + Act

ReAct (Yao et al., 2022) is a general framework for interleaving reasoning traces with actions in language models. In the retrieval context, the model generates **thought** (chain-of-thought reasoning) and **act** (tool invocations such as Wikipedia search, calculator) interleaved in a single output stream. The action results are observed and fed back to the model as new context.

```
Thought: I need to find the population of Paris.
Action: Search[Paris population 2023]
Observation: Paris has a population of approximately 2.1 million in the city proper...
Thought: Now I can answer the question.
Action: Finish[2.1 million]
```

ReAct enables dynamic query formulation, progressive context building, and graceful handling of retrieval failure (the model can try different queries if the first retrieval is uninformative). It became the conceptual foundation for tool-using agents and agentic RAG pipelines.

---

## Self-RAG: Self-Reflective Retrieval

Self-RAG (Asai et al., 2023) trains a single language model to interleave generation with retrieval and self-critique, using special reflection tokens. Rather than always retrieving, the model learns **when retrieval is necessary** and **whether retrieved documents are useful**.

### Reflection Tokens

Self-RAG introduces four types of special tokens inserted during generation:

- **[Retrieve]:** Should the model retrieve additional documents? (Yes/No/Continue)
- **[IsRel]:** Is the retrieved passage relevant to the query? (Relevant/Irrelevant)
- **[IsSup]:** Does the generated segment faithfully use the retrieved passage? (Fully supported/Partially supported/No support)
- **[IsUse]:** Is the generated response useful? (Utility score 1–5)

### Training Procedure

1. A critic model is fine-tuned to generate these reflection tokens given (query, retrieved passage, generation) triples.
2. The critic labels a large corpus of (input, output) pairs with reflection tokens.
3. A generator model is fine-tuned on this augmented corpus to simultaneously generate text and reflection tokens.
4. At inference, the model generates text and reflection tokens jointly; retrieval is invoked only when [Retrieve=Yes] is produced.

### Results

Self-RAG outperforms both standard RAG and instruction-tuned GPT-3.5 on a range of tasks including open-domain QA, long-form generation (ASQA), fact verification, and medical QA. The self-critique tokens also allow post-hoc inspection of the model's retrieval and citation quality.

---

## Corrective RAG (CRAG)

CRAG (Yan et al., 2024) adds a retrieval evaluator that assesses the quality of retrieved documents and triggers corrective actions when retrieval quality is low:

- **High confidence:** Use retrieved documents directly.
- **Low confidence:** Fall back to web search (using search engine APIs) to find more relevant information.
- **Ambiguous:** Decompose the retrieved documents, filter irrelevant portions, and refine the query.

The corrective mechanism makes RAG more robust to retrieval failure — a critical practical concern when document corpora are incomplete or queries are unusual.

---

## Agentic RAG System Design

Modern agentic RAG systems build on these ideas to create full pipelines:

**Query analysis:** The agent first analyzes the query to determine its complexity. Simple factoid questions are routed to single-step RAG; complex multi-hop questions trigger iterative retrieval.

**Tool use:** The agent has access to multiple retrieval tools (vector search, BM25, web search, structured database query, code execution) and selects appropriate tools based on query type.

**Plan-and-execute:** Some systems (LangGraph, LlamaIndex AgentRunner) implement a plan step where the agent explicitly outlines a retrieval strategy before executing it, enabling more coherent multi-step retrieval.

**Reflection and verification:** After generation, a separate step checks factual claims against retrieved sources, identifying unsupported claims and either retrieving additional evidence or flagging uncertainty.

**State management:** Multi-turn agentic RAG maintains a scratchpad or working memory of retrieved passages, intermediate reasoning steps, and generated sub-answers, allowing the agent to build on prior retrieval steps.

---

## Challenges and Open Problems

- **Retrieval attribution:** Ensuring generated claims are traceable to specific retrieved passages remains difficult, especially with multi-document synthesis.
- **Latency:** Multiple retrieval steps and self-reflection add significant latency compared to single-step RAG. Production systems must carefully balance quality and response time.
- **Retrieval quality cascades:** Errors in early retrieval steps propagate through the reasoning chain. Robust retrieval evaluation is critical.
- **Context window limits:** As more documents are retrieved across multiple steps, context windows may be exceeded, requiring compression or selective context management.

Agentic RAG represents the convergence of retrieval systems, tool-using agents, and self-reflective LLMs — and is currently one of the most active research areas in applied NLP.
