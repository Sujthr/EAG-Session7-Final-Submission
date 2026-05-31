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

## Sample Run — Claude Shannon (Run `7a9d48f7`)

**Query:** Fetch https://en.wikipedia.org/wiki/Claude_Shannon and tell me his birth date, death date, and three key contributions to information theory.

**Result:** 2 iterations · 67.9s

| Iter | Action |
|------|--------|
| 1 | `fetch_url` → Wikipedia page fetched, artifact stored |
| 2 | `ANSWER` → full answer produced from artifact |

**Answer:**
> Claude Shannon was born on **April 30, 1916**, and died on **February 24, 2001**. He is known as the "father of information theory." Three key contributions:
>
> 1. Founding the field of information theory, which mathematically models communication and information transmission.
> 2. Laying the theoretical foundations for digital circuits and communication systems.
> 3. Developing concepts such as the minimax procedure for computer chess and applying information theory principles to game theory and cryptography.

Full iteration-level traces for all runs: `S7Code/S7code/traces/query_traces.md`

---

## Full Documentation

See [RUNBOOK.md](RUNBOOK.md) for:
- Prerequisites and one-time setup
- All 8 mandatory queries (A–H)
- Troubleshooting guide
- RAG vs no-RAG comparison
