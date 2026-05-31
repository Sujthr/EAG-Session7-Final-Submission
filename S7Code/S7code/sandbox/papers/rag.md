# Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks

**Authors:** Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Kuttler, Mike Lewis, Wen-tau Yih, Tim Rocktaschel, Sebastian Riedel, Douwe Kiela (Facebook AI Research / University College London)
**Year:** 2020 (NeurIPS)

---

## Overview

Retrieval-Augmented Generation (RAG) addresses a fundamental limitation of parametric language models: their knowledge is frozen at training time, making them unreliable for tasks requiring current, specific, or verifiable factual information. RAG combines the expressive generation capabilities of pre-trained sequence-to-sequence models with non-parametric retrieval over a dense vector index of documents, allowing the model to access and reason over external knowledge at inference time.

## Problem Statement

Large language models store knowledge implicitly in their parameters, but this approach has critical drawbacks:

- **Knowledge staleness:** Facts learned during training become outdated
- **Hallucination:** Models generate plausible-sounding but incorrect facts when queried beyond their knowledge
- **Opacity:** It is difficult to trace which training examples influenced a generation
- **Storage inefficiency:** Encoding all world knowledge in parameters requires enormous model sizes

RAG reframes this as: rather than encoding all facts parametrically, use retrieval to access a large corpus of documents and condition generation on the retrieved content.

## Architecture

The RAG framework consists of two components operating in tandem:

### Retriever: Dense Passage Retrieval (DPR)
The retriever uses two BERT-based encoders — one for queries and one for passages — to embed both into a shared dense vector space. Document retrieval is performed via Maximum Inner Product Search (MIPS) using FAISS (Facebook AI Similarity Search) over a pre-built index. Given an input query `x`, the retriever returns the top-k documents `z_1, ..., z_k` ranked by dot product similarity between query and passage embeddings.

The DPR retriever was pre-trained to assign higher similarity scores to passages known to answer a given question, using in-batch negative sampling.

### Generator: BART
The generator is BART (Bidirectional and Auto-Regressive Transformer), a sequence-to-sequence model pre-trained with a denoising objective. For each retrieved document `z_i`, the generator receives the concatenation `[x; z_i]` as input and produces output tokens autoregressively.

### Two RAG Variants

**RAG-Sequence:** Each retrieved document independently produces a full output sequence. Final predictions are marginalized over the top-k retrieved documents by summing their probabilities:
```
p(y|x) = sum_z p_retriever(z|x) * p_generator(y|x, z)
```

**RAG-Token:** Each token in the output can attend to a different retrieved document. This allows more flexible integration of information across sources, useful when different parts of the answer come from different documents.

## Training

Both the retriever and generator are trained jointly end-to-end. The document index is kept fixed during training (not backpropagated through), while the DPR encoder and BART parameters are updated jointly. The system is trained on question-answer pairs without any document-level supervision — the model must learn to retrieve relevant documents purely from answer supervision.

## Results

RAG was evaluated on knowledge-intensive NLP benchmarks:

- **Natural Questions (open-domain QA):** 44.5 EM (exact match), outperforming T5-11B (a purely parametric model with 11B parameters) with RAG using only 400M parameters
- **TriviaQA:** 56.8 EM in the open-book setting, competitive with state-of-the-art retrieval-augmented systems
- **WebQuestions:** 45.5 EM
- **CuratedTrec:** 52.2 EM
- **Jeopardy Question Generation:** RAG generations rated more factual, specific, and diverse by human evaluators
- **FEVER fact verification:** 71.8% label accuracy

RAG also demonstrated better factual grounding, generating fewer hallucinated facts compared to purely parametric baselines.

## Significance

RAG established the retrieval-augmented approach as a first-class methodology in NLP research, and its influence has only grown since. Key contributions include:

1. **Hybrid parametric/non-parametric memory:** Demonstrating that external memory can augment LLMs without retraining
2. **Updatable knowledge:** The document index can be refreshed without retraining the model
3. **Interpretability:** Retrieved documents provide a citation trail for generated content
4. **Parameter efficiency:** Much smaller models achieve performance competitive with much larger parametric ones

The RAG framework became the architectural blueprint for modern production AI systems. It directly inspired LlamaIndex, LangChain retrieval chains, and countless enterprise RAG pipelines. The core idea — retrieve then generate — is now the standard approach for building question-answering systems over private document corpora, and the combination of vector databases with LLMs is a defining pattern of applied AI in 2023-2025.
