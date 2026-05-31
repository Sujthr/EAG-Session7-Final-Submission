# RUNBOOK — Session 7 RAG Agent (Final Submission — OpenAI Only)

Everything you need to start, run, and validate the system from scratch.

---

## System Overview

```
OpenAI gpt-4.1-mini       ← LLM (all layers: perception, memory, decision)
OpenAI text-embedding-3-small  ← embeddings (1536-dim, FAISS-backed)
```

Two processes you run:
1. **Gateway V7** — FastAPI on port 8107
2. **Agent** — runs per query, auto-starts gateway if needed

One optional process:
3. **Streamlit dashboard** — port 8501

---

## Prerequisites

### 1. Get an OpenAI API key

Sign in at https://platform.openai.com and create an API key.
Minimum account credit: a few dollars is enough for all 8 mandatory queries.

### 2. Install uv (Python package manager)

```
pip install uv
```

---

## One-Time Setup

### Step 1 — Configure Gateway

```
cd "D:\EAG\EAG\Class 23 May\Final_Submission\Gateway"
copy .env.example .env
```

Open `Gateway\.env` and fill in your OpenAI key:

```env
OPENAI_API_KEY=sk-your-real-key-here
OPENAI_MODEL=gpt-4.1-mini
EMBED_OPENAI_MODEL=text-embedding-3-small
GATEWAY_V7_PORT=8107
```

Save and close.

### Step 2 — Install Gateway dependencies

```
cd "D:\EAG\EAG\Class 23 May\Final_Submission\Gateway\llm_gatewayV7"
uv sync
```

### Step 3 — Configure Agent (optional — only needed for Tavily web search)

```
cd "D:\EAG\EAG\Class 23 May\Final_Submission\S7Code\S7code"
copy .env.example .env
```

Open `S7Code\S7code\.env` and add your Tavily key (optional — DuckDuckGo works without it):

```env
TAVILY_API_KEY=tvly-...your-key
```

### Step 4 — Install Agent dependencies

```
cd "D:\EAG\EAG\Class 23 May\Final_Submission\S7Code\S7code"
uv sync
```

---

## Starting the System

### Terminal 1 — Start Gateway

```
cd "D:\EAG\EAG\Class 23 May\Final_Submission\Gateway\llm_gatewayV7"
uv run main.py
```

Expected output:
```
[providers] OpenAI active: gpt-4.1-mini
[embedders] OpenAI embedder active: text-embedding-3-small (1536-dim)
INFO:     Uvicorn running on http://0.0.0.0:8107
```

Keep this terminal open.

### Terminal 2 — Run Queries

```
cd "D:\EAG\EAG\Class 23 May\Final_Submission\S7Code\S7code"
uv run agent7.py "your query here"
```

The agent auto-starts the gateway if Terminal 1 is not running.

### Terminal 2 (alternative) — Streamlit Dashboard

```
cd "D:\EAG\EAG\Class 23 May\Final_Submission\S7Code\S7code"
streamlit run app.py
```

Opens at http://localhost:8501

---

## Running the 8 Mandatory Queries

Run each from `S7Code\S7code\` with `uv run agent7.py "..."`.

**Query A** — Web fetch + extraction
```
uv run agent7.py "Fetch https://en.wikipedia.org/wiki/Claude_Shannon and extract his birth date, death date, and 3 key contributions to information theory."
```

**Query B** — Multi-step + weather
```
uv run agent7.py "Find 3 family-friendly things to do in Tokyo this weekend. Check Saturday's weather forecast. Pick the most appropriate one given the weather."
```

**Query C — Run 1** — Memory write
```
uv run agent7.py "Mom's birthday is on 15 May 2026. Create reminder files: one 2 weeks before and one on the day itself."
```

**Query C — Run 2** — Cross-run memory recall (run as a separate command after Run 1)
```
uv run agent7.py "When is mom's birthday?"
```

**Query D** — Web research + synthesis
```
uv run agent7.py "Search for Python asyncio best practices, read the top 3 results, and summarize the agreed-upon advice."
```

**Query E** — Single document index + extract
```
uv run agent7.py "Index the file papers/attention.md and extract 3 key contributions of the Transformer architecture."
```

**Query F — Run 1** — Bulk index
```
uv run agent7.py "Index all .md files under papers/. How many total chunks were created?"
```

**Query F — Run 2** — Semantic recall (run after F-Run1)
```
uv run agent7.py "Across all indexed papers, what do they collectively say about chain-of-thought reasoning?"
```

**Query G** — Semantic retrieval
```
uv run agent7.py "Across all indexed papers, how do they approach the credit assignment problem?"
```

**Query H** — Cross-document synthesis
```
uv run agent7.py "Compare the ReAct and Chain-of-Thought papers on their use of intermediate reasoning steps."
```

---

## Run All Queries at Once

```
cd "D:\EAG\EAG\Class 23 May\Final_Submission\S7Code\S7code"
uv run python run_all_queries.py
```

Logs go to `logs/<session_id>/`. Summary in `logs/<session_id>/summary.md`.

---

## Indexing the Full 55-Document Corpus

Run this once before attempting Queries F, G, H, or any custom semantic queries:

```
uv run agent7.py "Index all .md files under papers/. How many chunks were created?"
```

After indexing:
```
uv run agent7.py "How many fact chunks are currently stored in memory?"
```

---

## RAG vs No-RAG Comparison

```
# With FAISS (default)
uv run agent7.py "How do ReAct and Chain-of-Thought papers differ on intermediate reasoning?"

# Without FAISS
set S7_DISABLE_FAISS=1
uv run agent7.py "How do ReAct and Chain-of-Thought papers differ on intermediate reasoning?"
set S7_DISABLE_FAISS=
```

---

## Quick Health Checks

```
# Is Gateway running?
curl http://localhost:8107/v1/status

# Which embedding model is active?
curl -X POST http://localhost:8107/v1/embed -H "Content-Type: application/json" -d "{\"text\":\"test\",\"task_type\":\"retrieval_query\"}"
```

---

## Troubleshooting

### "OPENAI_API_KEY not set or invalid"
- Verify `Gateway\.env` exists (not just `.env.example`)
- Key must start with `sk-` or `sk-proj-`

### Gateway won't start
- Run `uv sync` in `Gateway\llm_gatewayV7\` to ensure dependencies are installed
- Check that port 8107 is not in use

### Embedding fails / dimension mismatch
- Delete `state/index.faiss` and `state/index_ids.json` if you previously used Ollama/Gemini embeddings (768-dim)
- OpenAI embeddings are 1536-dim — the two are incompatible in the same index

### FAISS index missing after restart
- Normal — rebuilt automatically from `state/memory.json`

### Query C Run 2 returns wrong answer
- Confirm Run 1 completed and `state\memory.json` contains a "15 May 2026" entry
- If empty, Run 1 didn't persist — re-run it

---

## File Locations Reference

```
Final_Submission\
├── RUNBOOK.md                    ← this file
├── Gateway\
│   ├── .env                      ← your OpenAI key (create from .env.example)
│   ├── .env.example              ← template
│   └── llm_gatewayV7\
│       └── main.py               ← start with: uv run main.py
└── S7Code\S7code\
    ├── .env                      ← Tavily key (create from .env.example)
    ├── .env.example              ← template
    ├── agent7.py                 ← run with: uv run agent7.py "query"
    ├── app.py                    ← run with: streamlit run app.py
    ├── state\
    │   ├── memory.json           ← persisted memory (survives restarts)
    │   ├── index.faiss           ← FAISS binary index (1536-dim, OpenAI)
    │   └── index_ids.json        ← FAISS id map
    └── sandbox\papers\           ← 55 .md corpus files
```
