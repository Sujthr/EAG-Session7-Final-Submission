# Cross-Encoders for Reranking in RAG: ColBERT, MonoBERT, BGE Reranker, and the Bi-Encoder vs. Cross-Encoder Tradeoff

**Key Papers:**
- *Passage Re-ranking with BERT* — Nogueira & Cho, 2019 (MonoBERT)
- *ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT* — Khattab & Zaharia, Stanford, 2020
- *BGE M3-Embedding and Reranker* — Chen et al., Beijing Academy of AI, 2024

---

## Overview

Retrieval-Augmented Generation (RAG) systems typically operate in two stages: (1) fast first-stage retrieval using a bi-encoder to narrow a large corpus down to a candidate set, followed by (2) slower but more accurate second-stage reranking using a cross-encoder to precisely score and reorder the candidates. Understanding the architectural tradeoffs between these approaches is essential for building high-quality RAG pipelines.

## Bi-Encoders: Fast but Approximate

A **bi-encoder** (also called a dual encoder) independently encodes the query and each document into dense vectors, then computes relevance as a dot product or cosine similarity:

```
score(q, d) = encode_q(q) · encode_d(d)
```

Because document embeddings can be precomputed and indexed offline (using FAISS, HNSW, or similar ANN indexes), retrieval is extremely fast — millions of documents can be searched in milliseconds. This makes bi-encoders the standard first-stage retriever in RAG.

The limitation is **representational compression**: the entire meaning of a document must be packed into a single fixed-size vector (typically 768 or 1024 dimensions). The query and document never "see" each other during encoding, so the relevance score cannot capture fine-grained token-level interactions. This leads to false positives — documents that seem topically related but don't actually answer the query.

## Cross-Encoders: Accurate but Slow

A **cross-encoder** concatenates the query and document into a single input sequence and processes them jointly through a transformer:

```
Input: [CLS] query [SEP] document [SEP]
score = linear_layer(CLS_representation)
```

Because both query and document tokens attend to each other in every layer, the cross-encoder can model arbitrarily complex interactions — exact phrase matching, entity co-reference, negation, and semantic nuance that bi-encoders miss. This makes cross-encoders substantially more accurate at ranking.

The cost is that documents cannot be precomputed independently — every (query, document) pair requires a full forward pass. For a query against 1 million documents, this is computationally prohibitive. Cross-encoders are therefore used only to rerank the top-K candidates (K typically 50–200) returned by the bi-encoder.

## MonoBERT: BERT as a Reranker

Nogueira & Cho (2019) demonstrated that fine-tuning BERT as a binary relevance classifier is an extremely effective reranking approach. Given query q and passage p, BERT processes the concatenation and the [CLS] token's representation is passed through a linear layer to produce a relevance score. Fine-tuned on MS MARCO's ~400K query-passage relevance pairs, MonoBERT dramatically outperformed all prior retrieval methods on TREC Deep Learning tracks.

MonoBERT established the cross-encoder reranker as the dominant second-stage paradigm and demonstrated that pretrained language models contain rich semantic knowledge that can be fine-tuned for relevance judgment with relatively little data.

## ColBERT: Late Interaction — A Middle Ground

ColBERT (Khattab & Zaharia, 2020) introduces **late interaction** as a compromise between bi-encoder speed and cross-encoder accuracy. Instead of producing a single vector per document, ColBERT produces one vector per token:

```
Query encoding:  Q = [q_1, q_2, ..., q_|q|]  (token-level embeddings)
Document encoding: D = [d_1, d_2, ..., d_|d|]  (token-level embeddings)
```

Relevance is computed as the **MaxSim** operator: for each query token, find its maximum cosine similarity to any document token, then sum these maxima:

```
score(q, d) = Σ_i max_j (q_i · d_j)
```

This preserves token-level granularity in matching (catching exact phrase matches and specific entity mentions that single-vector bi-encoders miss) while allowing document representations to be precomputed and indexed offline. ColBERT requires storing ~128-dimensional vectors for each token in the corpus (rather than a single vector), which increases index size by ~100x over bi-encoders but remains manageable.

ColBERT v2 adds supervised contrastive training with hard negatives and knowledge distillation from a cross-encoder teacher, substantially improving quality. On MS MARCO and TREC benchmarks, ColBERT v2 matches or exceeds cross-encoder quality while remaining much faster.

**PLAID** (ColBERT's production serving system) further optimizes index traversal using centroid-based filtering and approximate scoring, achieving sub-10ms latency for 140M passages on a single GPU.

## BGE Reranker

The BGE (BAAI General Embedding) model family from Beijing Academy of AI includes a dedicated cross-encoder reranker (bge-reranker-v2-m3) that supports multilingual reranking across 100+ languages. Key features:

- Based on XLM-RoBERTa with cross-encoder fine-tuning on multilingual relevance data.
- Supports three retrieval modes: dense (single vector), sparse (BM25-style lexical), and ColBERT-style multi-vector.
- The BGE-M3 model can function as bi-encoder, sparse encoder, and ColBERT encoder simultaneously (called "M3" for Multi-Functionality, Multi-Linguality, Multi-Granularity).
- Reranker models (bge-reranker-base, large, v2-m3) are straightforward cross-encoders fine-tuned on BEIR, MS MARCO, and machine-translated multilingual data.

BGE rerankers consistently rank among the top performers on the MTEB (Massive Text Embedding Benchmark) reranking tracks and are widely used as drop-in rerankers in LangChain, LlamaIndex, and similar RAG frameworks.

## Practical RAG Pipeline Design

A well-optimized RAG pipeline typically combines:

1. **First stage (bi-encoder)**: Dense retrieval using a model like E5-large, BGE-large, or text-embedding-3-large. Retrieves top 50–200 candidates from the full corpus in milliseconds.

2. **Second stage (cross-encoder reranker)**: Reranks the top-K candidates using bge-reranker-v2-m3, cohere-rerank-v3, or a similar model. Processes ~100 candidates in ~500ms on a GPU.

3. **Final selection**: Top 3–10 reranked passages are provided as context to the generator LLM.

The quality gain from adding a reranker is typically 5–15% on downstream QA tasks (measured by EM or ROUGE), with the largest gains on queries requiring precise matching rather than broad topic retrieval. The latency cost (~500ms) is acceptable for most interactive applications.

## Tradeoff Summary

| Property | Bi-Encoder | ColBERT | Cross-Encoder |
|----------|-----------|---------|---------------|
| Retrieval latency | <10ms | 50-200ms | N/A (reranking only) |
| Reranking latency | — | 50-200ms | 200-1000ms |
| Index size | Small | Large | None |
| Accuracy | Moderate | High | Highest |
| Multilingual | Varies | Varies | Good (XLM) |

## Significance

The bi-encoder + cross-encoder pipeline has become the canonical architecture for knowledge-intensive NLP tasks and RAG systems. It cleanly separates the recall (retrieval) and precision (reranking) problems, allowing each component to be optimized independently. ColBERT's late interaction represents an important middle ground that is increasingly practical with modern infrastructure. As RAG systems are deployed in production, the quality-latency tradeoff at the reranking stage is one of the most consequential engineering decisions in the pipeline.
