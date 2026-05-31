# Sparse Attention: Generating Long Sequences and the Longformer

**Primary Papers:**
- **Generating Long Sequences with Sparse Transformers** — Rewon Child, Scott Gray, Alec Radford, Ilya Sutskever (OpenAI, 2019), arXiv:1904.10509
- **Longformer: The Long-Document Transformer** — Iz Beltagy, Matthew Peters, Arman Cohan (Allen AI, 2020), arXiv:2004.05150

---

## The Problem: Quadratic Attention Complexity

Standard transformer self-attention has O(n²) time and memory complexity with respect to sequence length n. For a sequence of 1024 tokens, the attention matrix has ~1 million entries; at 8192 tokens, it has ~67 million. This quadratic scaling makes full attention computationally infeasible for long documents, high-resolution images, audio waveforms, and genomic sequences. Both the Sparse Transformer and Longformer address this by replacing full attention with structured sparse patterns that reduce complexity to O(n√n) or O(n log n) while preserving most of the representational power.

---

## Sparse Transformers (Child et al., 2019)

### Key Idea

The central observation is that learned attention patterns in trained transformers are not uniformly distributed — most attention heads specialize in attending to nearby tokens, specific structural positions, or a small set of globally important tokens. Sparse Transformers formalize this by specifying attention patterns a priori, restricting each query to attend to only a subset S(i) of the positions.

### Sparse Attention Patterns

Two factorized sparse patterns are proposed:

**Strided attention:** Each token attends to the previous l tokens (local window) and every l-th token before it (strided global view). This creates two interleaved attention "heads" — one capturing local context, one capturing periodic long-range structure. Strided attention is particularly effective for structured data like music and images where periodic structure is meaningful.

**Fixed attention:** Some tokens are designated as "global summary" positions. Every token attends to the most recent l tokens plus a fixed set of positions that appear every l steps. This allows information to flow efficiently across long distances through the fixed positions.

**Factorization principle:** Instead of having every head attend to all O(n) positions, pairs of attention heads together cover all pairwise interactions: if head A attends to positions {A_i} and head B attends to positions {B_j}, then together they span the full attention matrix in two layers. This factorization brings complexity down to O(n√n).

### Architecture Details

- **Custom GPU kernels:** Sparse Transformers required custom CUDA kernels to efficiently compute block-sparse attention matrices, since standard dense matrix multiplication is wasteful when most entries are zero.
- **Recomputation (gradient checkpointing):** Memory savings from sparse attention are augmented by recomputing activations during the backward pass rather than storing them, enabling training of models with 30+ layers at sequence lengths of 12,288 tokens.
- **Mixture of patterns:** Different attention heads use different sparsity patterns, combining local, strided, and global attention behaviors across the layer stack.

### Results

On image generation (CIFAR-10, ImageNet 64×64), Sparse Transformers matched or exceeded PixelCNN++ and full-attention transformers while handling sequences 4–8× longer. On raw audio (OpenAI's internal datasets), the model generated coherent musical sequences at 65,536 timesteps — far beyond what full attention could accommodate.

---

## Longformer (Beltagy et al., 2020)

### Key Idea

Longformer combines two complementary attention patterns into a unified model for long document understanding: **local windowed attention** for most tokens, and **global attention** for task-specific tokens (e.g., the [CLS] token for classification, or question tokens in QA). This combination retains BERT-style bidirectional attention while scaling to sequences of 4,096+ tokens.

### Attention Patterns

**Sliding window attention:** Each token attends to w/2 tokens on each side (a window of size w). With multiple stacked layers, the receptive field grows linearly with depth — similar to how dilated convolutions aggregate global context. For a 12-layer model with window size 512, the effective receptive field at the top layer spans the entire document.

**Dilated sliding window:** To increase receptive field without increasing computation, attention windows can be dilated (spaced with gaps), analogous to dilated convolutions. This is applied in higher layers where long-range dependencies are more important.

**Global attention:** A small number of pre-selected tokens attend to and are attended by all other tokens. This creates O(n × g) additional attention computations (where g is the number of global tokens), which is negligible when g << n. Global tokens serve as information aggregation points and are task-defined: [CLS] for classification, question tokens for QA.

### Implementation

Longformer uses TVM-based custom CUDA kernels that implement the sliding window attention using banded matrix operations. The key insight is that a sliding window attention matrix is equivalent to a banded sparse matrix, which can be computed efficiently using tensor operations. Longformer achieves a linear speedup over full attention for long sequences.

**Positional embeddings:** Longformer extends BERT's positional embeddings by simply copying the 512-position embeddings multiple times to cover 4,096 positions, then fine-tuning. This "position embedding copying" trick allows leveraging pretrained BERT weights.

### Results and Benchmarks

Longformer was evaluated on:
- **Long document classification:** achieved state-of-the-art on Hyperpartisan news classification (94.8 accuracy) and several other long-doc benchmarks.
- **QA over long documents:** Strong performance on WikiHop, TriviaQA, and HotpotQA — tasks requiring reasoning over thousands of tokens.
- **Longformer-Encoder-Decoder (LED):** A variant with encoder-decoder architecture for summarization (arXiv summarization dataset), handling documents up to 16,384 tokens.

### Significance

**Sparse Transformers** proved that structured sparsity could match dense attention quality while enabling qualitatively new applications (long audio, images). **Longformer** made sparse attention practically accessible by building on pretrained BERT, enabling fine-tuning rather than training from scratch. Together, these works initiated a rich research thread — including BigBird (combining random, window, and global attention with theoretical guarantees), Reformer (locality-sensitive hashing for approximate attention), and Linformer (low-rank approximation) — that fundamentally changed how the field approaches sequence length limitations in transformers.
