# Chunking Strategies for RAG Systems

## Overview

Chunking is the process of splitting source documents into smaller units before embedding and indexing them in a vector store. It is one of the most practically impactful — and often underappreciated — design decisions in a Retrieval-Augmented Generation pipeline. The chunking strategy determines what unit of text the retrieval system works with, directly affecting both retrieval precision (are the right passages found?) and generation quality (does the LLM receive coherent, sufficient context?).

There is no universally optimal chunking strategy. The right approach depends on document structure, embedding model capabilities, LLM context window size, and the nature of user queries.

---

## Why Chunking Matters

**Embedding models have token limits.** Most sentence transformer models cap at 512 tokens. OpenAI's text-embedding-3 models support up to 8191 tokens, but embedding very long passages dilutes semantic focus — a 3000-token document produces a single vector that averages over many topics, reducing retrieval precision for specific sub-questions.

**LLM context windows are finite (and expensive).** Even with 128K-token context windows, retrieving top-k chunks that collectively fit without redundancy requires thoughtful sizing. Chunks that are too large retrieve irrelevant content alongside relevant content; chunks that are too small lack sufficient context for the LLM to generate grounded answers.

**Chunk boundaries affect coherence.** A chunk that begins mid-sentence or ends before completing a key argument provides a confusing, incomplete unit of meaning to the LLM.

---

## Strategy 1: Fixed-Size Chunking

**Approach:** Split documents into chunks of exactly N tokens (or characters), regardless of content structure.

**Implementation:** Use a tokenizer or character count to split text at every N tokens, optionally with an overlap of O tokens between consecutive chunks.

**Pros:**
- Extremely simple to implement and reason about
- Predictable index size and retrieval behavior
- Compatible with all document types

**Cons:**
- Ignores semantic boundaries — a chunk may split mid-sentence, mid-paragraph, or mid-table
- Overlapping windows increase index size significantly
- Poor performance on structured documents (PDFs with headers, code files, legal clauses)

**Typical parameters:** 256–1024 tokens per chunk, 10–20% overlap. LangChain's `CharacterTextSplitter` and `TokenTextSplitter` implement this approach.

---

## Strategy 2: Sliding Window Chunking

**Approach:** A variant of fixed-size chunking where chunks overlap by a fixed stride. A window of size W slides over the document with step size S, creating chunks at positions [0:W], [S:S+W], [2S:2S+W], ...

**Key benefit:** Ensures that any phrase spanning a naive chunk boundary appears fully within at least one chunk, improving recall for queries that target boundary-spanning content.

**Trade-off:** Overlap ratio W/S directly controls redundancy. A 50% overlap doubles the number of chunks and doubles index storage and embedding compute costs.

**When to use:** Documents where important statements frequently span what would be natural boundaries (transcripts, continuous narrative prose, scientific abstracts).

---

## Strategy 3: Recursive Character Text Splitting

**Approach:** Attempts to split on progressively finer-grained separators: `["\n\n", "\n", ". ", " ", ""]`. First tries to split on paragraph breaks; if chunks are still too large, splits on line breaks; if still too large, on sentences; and so on down to individual characters.

**Why this works:** Natural text has hierarchical structure. Paragraph breaks signal the strongest semantic boundaries, followed by sentences, then clauses. By respecting this hierarchy, recursive splitting produces chunks that are more semantically coherent than fixed-size splitting.

**Implementation:** LangChain's `RecursiveCharacterTextSplitter` is the most widely used implementation and is the recommended default for general-purpose RAG over unstructured text.

**Pros:** Better semantic coherence than naive fixed-size splitting; simple and fast.  
**Cons:** Still blind to document-level structure (headers, sections, tables).

---

## Strategy 4: Semantic Chunking

**Approach:** Instead of splitting on character/token boundaries, semantic chunking embeds each sentence (or small group of sentences) and groups consecutive sentences that are semantically similar into a single chunk. A new chunk begins when the semantic similarity between adjacent sentences drops below a threshold.

**Process:**
1. Split document into sentences using a sentence boundary detector
2. Embed each sentence using a lightweight embedding model
3. Compute cosine similarity between consecutive sentence embeddings
4. Insert chunk boundaries where similarity drops sharply (breakpoint detection)

**Pros:** Produces chunks that align with genuine topic shifts in the document; very effective for long documents with multiple distinct sections.  
**Cons:** Computationally expensive at indexing time (requires embedding every sentence); chunk sizes are variable and unpredictable; requires tuning the similarity threshold; adds complexity to the pipeline.

**Implementation:** LangChain's `SemanticChunker` and LlamaIndex's `SemanticSplitterNodeParser`.

---

## Strategy 5: Structure-Aware / Document-Specific Splitting

**Approach:** Use document structure metadata to define chunk boundaries explicitly.

- **Markdown/HTML:** Split on headers (`#`, `##`, `<h1>`, `<h2>`), preserving each section as a chunk
- **PDF:** Use layout analysis (PDFMiner, PyMuPDF, Unstructured) to identify paragraphs, columns, tables, and captions separately
- **Code:** Split on function or class definitions using an AST parser rather than token count
- **Legal/regulatory:** Split on clauses and numbered sections

**Pros:** Maximally coherent chunks; each chunk corresponds to a meaningful document unit.  
**Cons:** Requires document-type-specific parsers; complex to generalize across document types.

---

## Hierarchical / Parent-Child Chunking

A recent architectural pattern involves creating two levels of chunks: **small child chunks** (128–256 tokens) for high-precision embedding and retrieval, and **larger parent chunks** (512–1024 tokens) that are returned to the LLM. The child chunk matches the query precisely; the parent chunk provides the broader context needed for coherent generation. This approach is implemented in LangChain as `ParentDocumentRetriever`.

---

## Practical Recommendations

| Document Type | Recommended Strategy |
|---|---|
| General prose | Recursive character splitting, 512 tokens, 10% overlap |
| Structured markdown | Header-based splitting |
| Scientific papers | Section-based + semantic chunking |
| Code | AST-based function splitting |
| Conversational transcripts | Fixed window with high overlap |

---

## Significance

Chunking sits at the intersection of information retrieval and NLP. Poor chunking is a silent RAG killer — it degrades recall without obvious error signals. Investing in document-appropriate chunking strategies, testing with retrieval recall metrics, and using hierarchical designs for complex documents are among the highest-return optimizations available in production RAG systems.
