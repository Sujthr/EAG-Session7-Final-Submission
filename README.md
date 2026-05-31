# Session 7 — RAG Agent (Final Submission)

A multi-layer autonomous agent that fetches, indexes, and reasons over documents using OpenAI models and a FAISS vector store.

## Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  Agent Loop (agent7.py)                                      │
│                                                             │
│  1. Memory Read   — retrieve relevant facts from FAISS      │
│  2. Perception    — decompose query into goals, track done  │
│  3. Decision      — choose: call a tool OR produce answer   │
│  4. Action        — execute tool via MCP (fetch/search/…)   │
│  5. Memory Write  — persist tool outcome back to FAISS      │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
Gateway V7 (FastAPI :8107)
    │
    ▼
OpenAI gpt-4.1-mini  +  text-embedding-3-small (1536-dim)
```

## Quick Start

### 1. Configure API key

```
cd Gateway
copy .env.example .env
# Edit .env — set OPENAI_API_KEY=sk-...
```

### 2. Start the Gateway

```
cd Gateway/llm_gatewayV7
uv run main.py
```

### 3. Run a query

```
cd S7Code/S7code
uv run agent7.py "Fetch https://en.wikipedia.org/wiki/Claude_Shannon and extract his birth date, death date, and 3 key contributions."
```

### 4. Or use the Streamlit dashboard

```
cd S7Code/S7code
streamlit run app.py          # opens http://localhost:8501
```

## Key Files

| Path | Purpose |
|------|---------|
| `Gateway/llm_gatewayV7/main.py` | LLM gateway — start this first |
| `S7Code/S7code/agent7.py` | Main agent loop |
| `S7Code/S7code/perception.py` | Goal decomposition + completion tracking |
| `S7Code/S7code/decision.py` | Tool-call vs. answer routing |
| `S7Code/S7code/memory.py` | FAISS-backed vector memory |
| `S7Code/S7code/app.py` | Streamlit UI |
| `S7Code/S7code/sandbox/papers/` | 55-document corpus (.md files) |
| `S7Code/S7code/state/` | Persisted memory (survives restarts) |

## Sample Run — Run `7a9d48f7` — 2026-05-31 17:18:24 UTC

**Query:** Fetch https://en.wikipedia.org/wiki/Claude_Shannon and tell me his birth date, death date, and three key contributions to information theory.

**Duration:** 67.9s &nbsp;|&nbsp; **Iterations:** 2 &nbsp;|&nbsp; **Memory hits (final iter):** 8

### Iteration 1

| Memory hits | 8 |
|---|---|

**Goals:**

- ✓ `g:9ceeff54` — Fetch the Wikipedia page for Claude Shannon
- ○ `g:ae0b82f4` — Extract Claude Shannon's birth date, death date, and three key contributions to information theory from the page
- ○ `g:a4dfe6c1` — Summarise Claude Shannon's birth date, death date, and three key contributions to information theory

**Decision:** `ANSWER`

> Claude Shannon was born on April 30, 1916, and died on February 24, 2001. He is known as the "father of information theory." Three key contributions of Claude Shannon to information theory include:
>
> 1. Founding the field of information theory, which transformed modern conceptions of communication, randomness, and information.
> 2. Laying the theoretical foundations for digital circuits and communication systems.

### Iteration 2

| Memory hits | 8 |
|---|---|

**Goals:**

- ✓ `g:9ceeff54` — Fetch the Wikipedia page for Claude Shannon
- ✓ `g:ae0b82f4` — Extract Claude Shannon's birth date, death date, and three key contributions to information theory from the page
- ○ `g:a4dfe6c1` — Summarise Claude Shannon's birth date, death date, and three key contributions to information theory

**Decision:** `ANSWER`

> Claude Shannon was born on April 30, 1916, and died on February 24, 2001. He is known as the "father of information theory." Three key contributions of Claude Shannon to information theory include:
>
> 1. Founding the field of information theory, which mathematically models communication and information transmission.
> 2. Laying the theoretical foundations for digital circuits and communication systems.
> 3. Developing concepts such as the minimax procedure for computer chess and applying information theory principles to game theory and cryptography.

---

**Final Answer:**

> Claude Shannon was born on April 30, 1916, and died on February 24, 2001. He is known as the "father of information theory." Three key contributions of Claude Shannon to information theory include:
>
> 1. Founding the field of information theory, which mathematically models communication and information transmission.
> 2. Laying the theoretical foundations for digital circuits and communication systems.
> 3. Developing concepts such as the minimax procedure for computer chess and applying information theory principles to game theory and cryptography.
>
> These contributions have profoundly influenced modern communication, digital electronics, and cryptography.
