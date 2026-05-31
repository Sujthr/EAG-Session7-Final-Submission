# Language Models are Unsupervised Multitask Learners (GPT-2)

**Authors:** Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever  
**Institution:** OpenAI  
**Year:** 2019  
**Paper:** OpenAI Blog / Technical Report (not peer-reviewed)

---

## Overview

GPT-2 (Generative Pre-trained Transformer 2) is a large-scale language model that demonstrated, for the first time at meaningful scale, that a single language model trained purely on next-token prediction could perform competently across a wide range of NLP tasks — without any task-specific fine-tuning. The paper's central claim is captured in its title: language models are, implicitly, multitask learners, because natural language text on the internet contains implicit demonstrations of translation, summarization, question answering, and reasoning.

GPT-2 was trained on WebText, a dataset of 40GB of text scraped from URLs shared on Reddit with at least 3 upvotes — a quality filter based on human curation signals. The largest model had 1.5 billion parameters, making it the largest publicly discussed language model at the time.

## Key Ideas

**Zero-shot task transfer:** GPT-2's defining contribution was demonstrating that a language model trained only to predict the next word could, at test time, be prompted to perform tasks like reading comprehension (CoQA), translation (WMT English-French), and summarization (CNN/DailyMail) without any gradient updates or labeled examples. The model implicitly learns to condition on natural language task descriptions embedded in the prompt.

**Task conditioning through prompting:** Rather than appending a task-specific classification head, the model is prompted in natural language. For translation, the model sees examples like "Q: What is the French translation of 'cheese'? A:". For summarization, the document is followed by "TL;DR:". This formalism — which we now call zero-shot prompting — was a conceptual precursor to the instruction-following paradigm.

**The internet as implicit supervision:** The core philosophical argument is that a sufficiently large and diverse corpus of text contains implicit demonstrations of virtually every NLP task that humans care about. Reading comprehension, translation, and logical inference all appear naturally in web text. A model that learns to model this distribution must, in some sense, learn to perform these tasks.

## Architecture

GPT-2 uses the same transformer decoder architecture as GPT-1, with modifications:

- **Layer normalization** is moved to the *input* of each sub-block (pre-norm), with an additional layer normalization after the final self-attention block.
- **Vocabulary size** expanded to 50,257 tokens using byte-pair encoding.
- **Context window** extended to 1024 tokens (from 512 in GPT-1).
- **Initialization:** Residual layer weights are scaled by 1/√N where N is the number of residual layers, preventing gradient explosion in deep networks.

Four model sizes were trained: 117M, 345M, 762M, and 1542M parameters. The 1542M parameter model (GPT-2 Large) was the primary model evaluated and withheld from release initially due to concerns about misuse.

The architecture uses:
- **Masked self-attention** (causal masking) ensuring the model can only attend to previous tokens.
- **Learned positional embeddings** (not sinusoidal).
- **GELU activations** in the feed-forward layers.
- **Weight tying** between the token embedding and the final output projection layer.

## Training Details

GPT-2 was trained using:
- **Dataset:** WebText — 8 million web documents, 40GB after filtering.
- **Optimizer:** Adam with learning rate scheduling.
- **Batch size:** 512.
- **BPE tokenizer** that operates on bytes rather than Unicode characters, enabling the model to represent any string without unknown tokens.

The byte-level BPE tokenizer was a significant practical innovation: it avoids out-of-vocabulary tokens entirely, which is critical for multilingual text and unusual characters.

## Results

GPT-2 achieved state-of-the-art results on 7 out of 8 language modeling benchmarks (PTB, WikiText-2, WikiText-103, 1BW, enwiki8, text8, lambada) in a zero-shot setting — the model was never fine-tuned on these datasets.

On question answering (CoQA), GPT-2 achieves 55 F1 zero-shot versus a supervised baseline of 89 — respectable for zero-shot but not competitive with fine-tuned models of the era. On summarization (CNN/DailyMail), the model produces reasonable summaries when prompted with "TL;DR:", scoring 21 ROUGE-2 (for reference, the lead-3 baseline is 17.7). On translation (English to French), GPT-2 achieves 5 BLEU — weak but nonzero, demonstrating implicit multilingual learning.

The most striking result was on the Lambada dataset — predicting the final word of a passage — where GPT-2 improved from 45.99% (GPT-1) to 63.24% accuracy zero-shot.

## Controlled Release and Impact

OpenAI initially released only the 117M parameter model, citing concerns that the 1.5B model could be misused to generate disinformation. This decision was controversial and widely debated. The full model was released six months later, with no significant observed misuse, which prompted reflection on AI release policies.

GPT-2 had several lasting impacts:

1. **Prompt engineering as a discipline:** GPT-2 established that natural language prompts can condition model behavior — a precursor to the prompt engineering and instruction-tuning research that followed.
2. **Foundation for GPT-3:** The architectural patterns and training philosophy of GPT-2 scaled directly into GPT-3 (175B parameters, 2020), which demonstrated few-shot learning at a qualitatively different level.
3. **Text generation quality:** GPT-2's generations were coherent enough to be mistaken for human writing in short excerpts, marking a transition point in public awareness of language model capabilities.
4. **Open-source ecosystem:** The 117M and 345M models were widely used in research and application development, spawning tools like Hugging Face's transformers library as a primary distribution mechanism.

GPT-2 remains a landmark paper not because it solved any particular NLP task, but because it reframed the question: rather than asking "how do we build a model for task X," it asked "what does a model that learns from all of language know how to do?"
