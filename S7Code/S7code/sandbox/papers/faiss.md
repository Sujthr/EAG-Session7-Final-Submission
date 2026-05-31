# FAISS: Billion-Scale Similarity Search with GPUs

**Authors:** Jeff Johnson, Matthijs Douze, Hervé Jégou  
**Institution:** Meta AI Research (Facebook AI Research)  
**Year:** 2017 (arXiv:1702.08734); updated and expanded through 2019  
**Paper:** "Billion-scale similarity search with GPUs"

---

## Overview

FAISS (Facebook AI Similarity Search) is an open-source library for efficient similarity search and clustering of dense vector embeddings. It provides algorithms and GPU-accelerated implementations that scale from millions to billions of vectors — far beyond what naive approaches can handle. FAISS has become the de facto standard for vector search in production RAG systems, recommendation engines, image retrieval, and any system that needs to find nearest neighbors in high-dimensional embedding spaces.

The core problem FAISS solves: given a database of N vectors in d-dimensional space and a query vector q, find the k vectors most similar to q (by L2 distance or inner product). For N = 1 billion and d = 128, a brute-force scan is computationally infeasible at query time. FAISS provides a hierarchy of index types that trade exactness for speed and memory.

---

## Index Types

### IndexFlatIP and IndexFlatL2 (Exact Search)

The simplest indexes store vectors in a flat array and compute exact distances via brute-force. **IndexFlatIP** computes maximum inner product (for cosine similarity with normalized vectors); **IndexFlatL2** computes minimum squared L2 distance. These are exact — no approximation — but O(N) per query.

FAISS's GPU implementation of flat search is surprisingly fast: using GPU batched matrix multiplication (essentially computing the query-database inner product as a matrix-matrix product), FAISS can search 1 billion 128-dimensional vectors in ~0.3 seconds on a single GPU. For datasets up to ~10M vectors, flat GPU search is often the most practical choice.

**Key detail:** FAISS normalizes vectors when requested and expresses L2 search as:
‖q - xᵢ‖² = ‖q‖² - 2qᵀxᵢ + ‖xᵢ‖²
allowing the expensive inner product term to dominate the computation and enabling GPU acceleration.

### IVF (Inverted File Index)

For larger datasets, FAISS partitions the vector space into Voronoi cells using k-means clustering. The IVF index has two stages:

1. **Training:** k-means is run on a training set to produce k centroids (k typically ranges from √N to 4√N).
2. **Indexing:** Each database vector is assigned to its nearest centroid. Vectors assigned to centroid c are stored in the "inverted list" for c.
3. **Search:** For a query q, FAISS finds the nprobe nearest centroids, then searches only the vectors in those lists (typically 1–4% of the full database).

**IVF complexity:** O(k + nprobe × (N/k)) per query, versus O(N) for flat search. With k = √N and nprobe = 1, this is O(√N) — a substantial speedup.

**IVFFlat** stores raw vectors; **IVFPQ** combines IVF with product quantization (see below) for memory compression. The nprobe parameter controls the accuracy-speed tradeoff: higher nprobe finds more accurate neighbors at greater computational cost.

### HNSW (Hierarchical Navigable Small World)

HNSW constructs a multi-layer proximity graph where each node (vector) is connected to M nearest neighbors at each layer. The graph has a hierarchical structure: the top layers are sparse (long-range connections), the bottom layer is dense (local connections). Search traverses from the top layer down, greedily navigating toward the query vector.

**Complexity:** O(log N) per query (empirically), with very low constant factors. HNSW typically achieves 99%+ recall at 1-10ms query times for million-scale databases.

**Parameters:**
- **M:** Number of connections per node (16–64 typically). Higher M improves recall but increases memory and build time.
- **efConstruction:** Size of the candidate list during graph construction. Higher values improve graph quality.
- **efSearch:** Size of the candidate list during search. The primary runtime recall-speed knob.

HNSW does not require training and has excellent recall but higher memory usage (each vector requires storing graph edges). It is the default choice for latency-critical applications at <100M scale.

### PQ (Product Quantization)

Product Quantization compresses high-dimensional vectors into compact codes that drastically reduce memory:

1. The d-dimensional space is split into m subspaces of d/m dimensions each.
2. Each subspace is quantized independently using a k-means codebook of size 256 (8 bits).
3. A vector is encoded as m bytes (one byte per subspace), representing indices into the m codebooks.

A 128-dimensional float32 vector (512 bytes) can be compressed to 16 bytes (m=16 subspaces) — a 32× memory reduction — at the cost of approximate distance computation.

**Asymmetric Distance Computation (ADC):** During search, the query is compared against precomputed lookup tables (one per subspace), making distance computation extremely fast — just m table lookups and additions.

PQ is almost always combined with IVF (**IVFPQ**) for billion-scale search: IVF reduces the candidate set, PQ compresses those candidates for fast reranking.

### ScaNN and FAISS's Role in the Ecosystem

FAISS popularized and made practical the IVF-PQ pipeline. Subsequent systems (ScaNN from Google, Milvus, Weaviate, Pinecone) build on the same conceptual foundation, adding features like dynamic insertion, distributed search, and learned quantization.

---

## GPU Acceleration

FAISS provides GPU implementations of flat search, k-means (used in IVF training), and IVF search. The key insight is that similarity search reduces to large batched matrix multiplication (queries × database vectors), which is precisely the operation GPUs excel at.

For flat search: a batch of Q queries against N vectors in d dimensions is computed as Q×d times d×N matrix multiplication — O(QNd) FLOPs efficiently parallelized across GPU threads. FAISS achieves near-peak GPU FLOP utilization for this operation.

For IVF-PQ on GPU: FAISS implements a "pass-through" approach where the GPU maintains the quantized codes in device memory and processes multiple queries simultaneously, achieving throughput of ~1M queries per second on a single V100 GPU for 100M-vector databases.

---

## Practical Recommendations

| Dataset size | Recommended index | Notes |
|---|---|---|
| < 1M | IndexFlatIP or HNSW | Exact or near-exact |
| 1M–100M | IVFFlat or HNSW | Balance of speed and recall |
| 100M–1B | IVFPQ with GPU | Memory-efficient, fast |
| > 1B | IVFPQ + multi-GPU sharding | Distributed required |

---

## Significance

FAISS democratized billion-scale vector search: before it, such systems required custom engineering. By open-sourcing highly optimized GPU kernels alongside flexible index structures, Meta AI enabled the entire vector database ecosystem. In the context of RAG systems, LLM deployment, and embedding-based retrieval, FAISS (or its conceptual descendants) is the critical infrastructure layer that makes semantic search practical at scale.
