# HyDE: Precise Zero-Shot Dense Retrieval without Relevance Labels

## Paper Details

- **Title:** Precise Zero-Shot Dense Retrieval without Relevance Labels
- **Authors:** Luyu Gao, Xueguang Ma, Jimmy Lin, Jamie Callan
- **Affiliation:** Carnegie Mellon University, University of Waterloo
- **Year:** 2022
- **Venue:** ACL 2023

---

## Problem Statement

Dense retrieval systems encode queries and documents into a shared embedding space, then retrieve documents by nearest-neighbor search. State-of-the-art dense retrievers (e.g., DPR, Contriever) require large amounts of labeled relevance data (query-document pairs) to train effective query encoders. Without such supervision, dense retrievers often underperform traditional sparse methods like BM25.

The core challenge in zero-shot dense retrieval is the **query-document asymmetry problem**: at retrieval time, a short natural language query must match against longer, information-rich documents. The query embedding and document embedding may occupy different regions of the vector space even when they are semantically related, particularly when no task-specific fine-tuning has been performed.

HyDE (Hypothetical Document Embeddings) is a zero-shot dense retrieval method that addresses this by reformulating the query into a form more compatible with the document embedding space — without requiring any labeled training data.

---

## Core Idea and Method

HyDE's key insight is elegantly simple: **instead of embedding the query directly, generate a hypothetical document that would answer the query, then embed that hypothetical document**.

The two-stage pipeline works as follows:

**Stage 1 — Hypothetical Document Generation:**
Given a query q, prompt an instruction-following LLM (e.g., InstructGPT/GPT-3.5) to generate a hypothetical document that could be a relevant answer. For example:
- Query: "What causes aurora borealis?"
- Hypothetical document: A generated paragraph explaining the interaction of solar wind particles with Earth's magnetic field and atmosphere, even though it is entirely fabricated.

The generated document will contain factual errors and hallucinations, but it captures the **vocabulary, style, and topical content** that a relevant document would likely contain.

**Stage 2 — Dense Retrieval with Hypothetical Embedding:**
The hypothetical document (not the original query) is encoded using a dense encoder such as Contriever or SimCSE. This embedding is used as the query vector to retrieve nearest-neighbor documents from the actual corpus.

The intuition is that:
- Documents in the index are encoded with a document-side encoder
- The hypothetical document is also a document in terms of length, style, and content distribution
- Therefore, the hypothetical document's embedding lies in a more compatible region of the embedding space than a short query would

Multiple hypothetical documents can be generated and their embeddings averaged for more robust retrieval.

---

## Architecture Details

**LLM component:** Any capable instruction-following LLM can serve as the generator. The paper uses InstructGPT (text-davinci-003) with task-specific prompts for different retrieval domains (web search, fact verification, open-domain QA, etc.).

**Encoder component:** HyDE is designed to work with **unsupervised** or **weakly supervised** encoders. The paper demonstrates it with Contriever (trained with contrastive learning on Wikipedia and CCNet without human labels) and shows it generalizes across encoders.

**Prompting strategy:** Task-specific prompts instruct the LLM to generate documents in the appropriate style. For web search, the prompt asks for a web passage; for scientific retrieval, it asks for a scientific article excerpt.

---

## Results

HyDE was evaluated on BEIR (a zero-shot retrieval benchmark spanning 18 heterogeneous datasets) and the TREC DL benchmarks.

**Key results on BEIR:**
- HyDE with Contriever encoder outperformed supervised DPR on 11 of 18 BEIR datasets without using any relevance labels
- Achieved an average nDCG@10 of ~0.49 vs. ~0.40 for Contriever alone and ~0.38 for BM25
- Particularly strong on knowledge-intensive tasks (NQ, HotpotQA, FiQA) and biomedical retrieval

**TREC DL 2019/2020:**
- HyDE approached the performance of DPR fine-tuned with supervised labels on the target domain
- Demonstrated strong generalization to out-of-domain retrieval without any domain-specific training

**Ablations:**
- Averaging embeddings across multiple hypothetical documents consistently improved results
- Quality of the generator LLM matters: stronger generators produced better hypothetical documents and better retrieval
- Even factually incorrect hypothetical documents improved retrieval over direct query embedding, confirming that vocabulary/style matters more than factual accuracy

---

## Significance

HyDE demonstrated that the **query-document asymmetry problem in dense retrieval can be resolved using LLM generation** rather than supervised training. This is significant for several reasons:

1. **Zero-shot generalization:** Strong retrieval performance is achievable on new domains and languages without any labeled data.
2. **Decoupling generation from retrieval:** The generator and encoder can be independently improved, offering a modular system design.
3. **Practical impact on RAG systems:** HyDE is widely adopted in retrieval-augmented generation (RAG) pipelines where domain-specific retrieval labels are unavailable. It is built into libraries like LangChain and LlamaIndex.
4. **Theoretical insight:** The success of HyDE validates the hypothesis that embedding space geometry is better navigated by document-like inputs, informing future retrieval architecture design.
