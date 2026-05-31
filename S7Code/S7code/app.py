"""
app.py — Session 7 RAG Agent: Streamlit Dashboard

6 pages via sidebar navigation:
  Home          — architecture overview + live stats
  Index         — manage the 50+ document corpus
  Search        — run queries through the full agent (RAG vs No-RAG)
  Diagnostics   — memory.json stats, FAISS status, run history
  Settings      — API key validation, model selection
  Provider Status — gateway health (live LLM + embedder status)

Run from S7Code/S7code/:
    streamlit run app.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Optional

import streamlit as st

# ── paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

GATEWAY_URL = "http://localhost:8107"
MEMORY_PATH = BASE_DIR / "state" / "memory.json"
PAPERS_DIR = BASE_DIR / "sandbox" / "papers"
SANDBOX_DIR = BASE_DIR / "sandbox"

st.set_page_config(
    page_title="S7 RAG Agent",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── sidebar navigation ────────────────────────────────────────────────────────
st.sidebar.title("🧠 S7 RAG Agent")
st.sidebar.markdown("**Session 7 — FAISS-backed RAG**")
st.sidebar.divider()

PAGES = {
    "🏠 Home": "home",
    "📚 Index Corpus": "index",
    "🔍 Search / Query": "search",
    "🔬 Diagnostics": "diagnostics",
    "⚙️ Settings": "settings",
    "📊 Provider Status": "provider_status",
}
page_label = st.sidebar.selectbox("Page", list(PAGES.keys()), key="page_nav")
page = PAGES[page_label]

st.sidebar.divider()
st.sidebar.caption(f"Base: `{BASE_DIR.name}/`")
st.sidebar.caption(f"Gateway: {GATEWAY_URL}")


# ── helpers ──────────────────────────────────────────────────────────────────

def _gateway_get(path: str, timeout: float = 5.0) -> Optional[dict]:
    try:
        import httpx
        r = httpx.get(f"{GATEWAY_URL}{path}", timeout=timeout)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def _gateway_up() -> bool:
    return _gateway_get("/v1/status") is not None


def _load_memory() -> list[dict]:
    if not MEMORY_PATH.exists():
        return []
    try:
        return json.loads(MEMORY_PATH.read_text(encoding="utf-8"))
    except Exception:
        return []


def _corpus_files() -> list[Path]:
    if not PAPERS_DIR.exists():
        return []
    return sorted(PAPERS_DIR.glob("*.md"))


def _sandbox_files() -> list[Path]:
    """All indexable files in sandbox (recursive)."""
    if not SANDBOX_DIR.exists():
        return []
    return sorted(SANDBOX_DIR.rglob("*.md")) + sorted(SANDBOX_DIR.rglob("*.txt"))


# ── PAGE: HOME ────────────────────────────────────────────────────────────────

if page == "home":
    st.title("🧠 Session 7 RAG Agent")
    st.markdown(
        "A **4-layer cognitive architecture** backed by FAISS vector retrieval "
        "with 5-key Gemini fallback rotation."
    )

    col1, col2, col3, col4 = st.columns(4)

    mem = _load_memory()
    fact_count = sum(1 for m in mem if m.get("kind") == "fact")
    embedded_count = sum(1 for m in mem if m.get("embedding") is not None)
    corpus_count = len(_corpus_files())

    col1.metric("Memory Items", len(mem))
    col2.metric("Facts (indexed chunks)", fact_count)
    col3.metric("Items with Embedding", embedded_count)
    col4.metric("Corpus Files", corpus_count)

    st.divider()
    st.subheader("Architecture")
    st.code("""
User Query
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  1. MEMORY READ   vector search (FAISS) → keyword fallback │
│     Returns up to 12 relevant MemoryItems from past runs  │
└────────────────────────┬────────────────────────────────┘
                         │ hits
                         ▼
┌─────────────────────────────────────────────────────────┐
│  2. PERCEPTION   (TOOL-BLIND)                            │
│     Observes query + history + hits → structured Goals   │
│     Never mentions tools, MCP, or index_document         │
└────────────────────────┬────────────────────────────────┘
                         │ Goal
                         ▼
┌─────────────────────────────────────────────────────────┐
│  3. DECISION   one LLM call per turn                     │
│     Sees goal + hits + optional artifact bytes            │
│     Outputs: ANSWER  OR  TOOL_CALL                        │
└──────────┬─────────────────────┬───────────────────────┘
           │ answer               │ tool_call
           ▼                     ▼
      [return final]   ┌──────────────────────┐
                       │  4. ACTION  (no LLM)  │
                       │  Dispatches MCP tool  │
                       │  Stores >4KB as art:  │
                       └──────────┬───────────┘
                                  │ result
                                  ▼
                       ┌──────────────────────┐
                       │  MEMORY WRITE         │
                       │  embed + FAISS index  │
                       └──────────────────────┘
    """, language="")

    st.divider()
    st.subheader("Gemini Key-Ring Failover")
    st.markdown("""
| Tier | Provider chain |
|------|---------------|
| **Embedding** (primary) | Ollama `nomic-embed-text` (768-dim, local) |
| **Embedding** (fallback) | Gemini `gemini-embedding-001` (768-dim, keys 1→5) |
| **LLM** (primary) | Ollama `qwen3:8b` |
| **LLM** (fallback) | Gemini 2.5-flash (keys 1→5) → Groq → NVIDIA → Cerebras |

On a 429 or quota error the gateway automatically advances to the next key.
All 5 keys share the same model and vector space — transparent to the agent.
    """)

    st.divider()
    st.subheader("MCP Tools Available")
    tools_md = """
| Tool | Purpose |
|------|---------|
| `web_search` | Tavily (primary) + DuckDuckGo fallback, ≤5 results |
| `fetch_url` | crawl4ai headless Chromium → clean markdown |
| `get_time` | Current time in any IANA timezone |
| `currency_convert` | ISO-3 currency conversion via frankfurter.dev |
| `read_file` | Read sandbox file |
| `list_dir` | List sandbox directory |
| `create_file` | Create file in sandbox |
| `update_file` | Overwrite sandbox file |
| `edit_file` | Find-and-replace in sandbox file |
| `index_document` | Chunk file → fact records → FAISS index |
| `search_knowledge` | Vector search over indexed fact chunks |
"""
    st.markdown(tools_md)


# ── PAGE: INDEX ────────────────────────────────────────────────────────────────

elif page == "index":
    st.title("📚 Corpus Index Manager")

    corpus = _corpus_files()
    st.metric("Corpus files", len(corpus), help="Files in sandbox/papers/*.md")

    st.subheader(f"Corpus Manifest ({len(corpus)} documents)")
    if corpus:
        rows = []
        for f in corpus:
            size = f.stat().st_size
            rows.append({"File": f.name, "Size (bytes)": size, "Path": str(f.relative_to(BASE_DIR))})
        st.dataframe(rows, use_container_width=True)
    else:
        st.warning("No .md files found in sandbox/papers/. Add documents to the corpus folder.")

    st.divider()
    st.subheader("Index a Document")
    st.markdown("Select a file from the sandbox and index it into the FAISS vector store.")

    sandbox_files = _sandbox_files()
    sandbox_rel = [str(f.relative_to(SANDBOX_DIR)) for f in sandbox_files]

    if sandbox_files:
        selected = st.selectbox("Select file to index", sandbox_rel)
        chunk_size = st.slider("Chunk size (words)", 100, 800, 400)
        overlap = st.slider("Overlap (words)", 0, 200, 80)

        if st.button("Index Document", type="primary"):
            with st.spinner(f"Indexing {selected}..."):
                query = f"Please index the document at papers/{selected} using index_document tool with chunk_size {chunk_size} and overlap {overlap}"
                result = subprocess.run(
                    [sys.executable, str(BASE_DIR / "agent7.py"), query],
                    cwd=str(BASE_DIR),
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
            st.code(result.stdout[-3000:] if result.stdout else "(no output)", language="")
            if result.returncode != 0:
                st.error(f"Process error:\n{result.stderr[-1000:]}")
    else:
        st.info("No files found in sandbox/. Place .md or .txt files under sandbox/ to index them.")

    st.divider()
    st.subheader("Index All Papers")
    st.markdown("Index all .md files in `sandbox/papers/` in one agent run.")
    if st.button("Index All Papers in sandbox/papers/", type="secondary"):
        with st.spinner("Indexing all papers..."):
            query = "Index all .md files under papers/ folder. Use index_document for each one."
            result = subprocess.run(
                [sys.executable, str(BASE_DIR / "agent7.py"), query],
                cwd=str(BASE_DIR),
                capture_output=True,
                text=True,
                timeout=300,
            )
        st.code(result.stdout[-4000:] if result.stdout else "(no output)", language="")


# ── PAGE: SEARCH ──────────────────────────────────────────────────────────────

elif page == "search":
    st.title("🔍 Query the RAG Agent")

    query = st.text_area(
        "Your query",
        placeholder="e.g. What do the indexed papers say about chain-of-thought reasoning?",
        height=100,
    )

    col_l, col_r = st.columns([2, 1])
    with col_l:
        mode = st.radio("Mode", ["RAG (with FAISS)", "No-RAG (FAISS disabled)"],
                        horizontal=True, index=0)
    with col_r:
        max_iter = st.number_input("Max iterations", min_value=1, max_value=30, value=20)

    if st.button("Run Query", type="primary", disabled=not query.strip()):
        env = os.environ.copy()
        env["S7_MAX_ITERATIONS"] = str(max_iter)

        if mode == "No-RAG (FAISS disabled)":
            env["S7_DISABLE_FAISS"] = "1"
            st.warning("FAISS disabled — agent will use keyword search only (no vector retrieval)")

        with st.spinner("Running agent..."):
            t0 = time.time()
            result = subprocess.run(
                [sys.executable, str(BASE_DIR / "agent7.py"), query.strip()],
                cwd=str(BASE_DIR),
                capture_output=True,
                text=True,
                timeout=300,
                env=env,
            )
            elapsed = time.time() - t0

        st.success(f"Completed in {elapsed:.1f}s | Exit code: {result.returncode}")

        st.subheader("Full Agent Trace")
        stdout = result.stdout or "(no output)"
        st.code(stdout[-6000:], language="")

        if result.stderr:
            with st.expander("stderr"):
                st.code(result.stderr[-2000:], language="")

    st.divider()
    st.subheader("Session 7 Mandatory Queries (A–H)")
    queries = {
        "A": "Fetch https://en.wikipedia.org/wiki/Claude_Shannon and extract his birth date, death date, and 3 key contributions to information theory.",
        "B": "Find 3 family-friendly things to do in Tokyo this weekend. Check Saturday's weather forecast. Pick the most appropriate one given the weather.",
        "C-run1": "Mom's birthday is on 15 May 2026. Create reminder files: one 2 weeks before and one on the day itself.",
        "C-run2": "When is mom's birthday?",
        "D": "Search for 'Python asyncio best practices', read the top 3 results, and summarize the agreed-upon advice.",
        "E": "Index the file papers/attention.md and extract 3 key contributions of the Transformer architecture.",
        "F-run1": "Index all .md files under papers/. How many chunks were created total?",
        "F-run2": "Across all indexed papers, what do they collectively say about chain-of-thought reasoning?",
        "G": "Across all indexed papers, how do they approach the credit assignment problem?",
        "H": "Compare the ReAct and Chain-of-Thought papers on their use of intermediate reasoning steps.",
    }
    for q_id, q_text in queries.items():
        with st.expander(f"Query {q_id}"):
            st.code(q_text, language="")
            if st.button(f"Run Query {q_id}", key=f"btn_{q_id}"):
                with st.spinner(f"Running Query {q_id}..."):
                    result = subprocess.run(
                        [sys.executable, str(BASE_DIR / "agent7.py"), q_text],
                        cwd=str(BASE_DIR),
                        capture_output=True,
                        text=True,
                        timeout=300,
                    )
                st.code(result.stdout[-5000:], language="")


# ── PAGE: DIAGNOSTICS ─────────────────────────────────────────────────────────

elif page == "diagnostics":
    st.title("🔬 Diagnostics")

    mem = _load_memory()

    col1, col2, col3 = st.columns(3)
    col1.metric("Total memory items", len(mem))
    col2.metric("Fact chunks", sum(1 for m in mem if m.get("kind") == "fact"))
    col3.metric("Embedded items", sum(1 for m in mem if m.get("embedding") is not None))

    # FAISS index
    state_dir = BASE_DIR / "state"
    faiss_path = state_dir / "index.faiss"
    ids_path = state_dir / "index_ids.json"

    st.subheader("FAISS Index")
    if faiss_path.exists() and ids_path.exists():
        try:
            ids = json.loads(ids_path.read_text(encoding="utf-8"))
            faiss_size = faiss_path.stat().st_size
            st.success(f"Index loaded: **{len(ids)} vectors**, {faiss_size:,} bytes on disk")
        except Exception as e:
            st.error(f"Index read error: {e}")
    else:
        st.warning("No FAISS index found. Run a query or index documents first.")

    st.divider()
    st.subheader("Memory Breakdown by Kind")
    kind_counts: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    run_counts: dict[str, int] = {}

    for m in mem:
        k = m.get("kind", "unknown")
        kind_counts[k] = kind_counts.get(k, 0) + 1
        s = m.get("source", "unknown")
        source_counts[s] = source_counts.get(s, 0) + 1
        r = m.get("run_id", "unknown")
        run_counts[r] = run_counts.get(r, 0) + 1

    if kind_counts:
        st.bar_chart(kind_counts)

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("By Source")
        st.json(source_counts)
    with col_b:
        st.subheader("Runs")
        st.json({k: v for k, v in sorted(run_counts.items(), key=lambda x: -x[1])[:20]})

    st.divider()
    st.subheader("Recent Memory Items")
    if mem:
        recent = mem[-20:][::-1]  # last 20, newest first
        for item in recent:
            kind = item.get("kind", "?")
            desc = item.get("descriptor", "")[:100]
            src = item.get("source", "?")
            created = item.get("created_at", "")[:16]
            has_vec = "✓" if item.get("embedding") else "✗"
            st.markdown(f"**[{kind}]** `{desc}` — src=`{src}` vec={has_vec} {created}")
    else:
        st.info("No memory items yet.")

    st.divider()
    st.subheader("Actions")
    col_x, col_y = st.columns(2)
    with col_x:
        if st.button("Clear All Memory (irreversible)", type="secondary"):
            confirm = st.checkbox("I understand this wipes all memory and FAISS index")
            if confirm:
                if MEMORY_PATH.exists():
                    MEMORY_PATH.unlink()
                if faiss_path.exists():
                    faiss_path.unlink()
                if ids_path.exists():
                    ids_path.unlink()
                st.success("Memory and FAISS index cleared.")
                st.rerun()
    with col_y:
        if st.button("Reload from Disk"):
            st.rerun()


# ── PAGE: SETTINGS ────────────────────────────────────────────────────────────

elif page == "settings":
    st.title("⚙️ Settings & API Key Validation")

    st.subheader("Gemini API Keys (5-key rotation ring)")
    st.markdown(
        "Set these in `Gateway/.env` as `GEMINI_API_KEY_1` … `GEMINI_API_KEY_5`. "
        "The gateway rotates to the next key automatically on quota errors."
    )

    gemini_status = []
    for i in range(1, 6):
        k = os.environ.get(f"GEMINI_API_KEY_{i}", "")
        masked = f"{k[:8]}…{k[-4:]}" if len(k) > 12 else ("(set)" if k else "(not set)")
        gemini_status.append({
            "Key slot": f"GEMINI_API_KEY_{i}",
            "Status": "✅ Set" if k else "⚠️ Not set",
            "Preview": masked,
        })
    st.dataframe(gemini_status, use_container_width=True)

    st.divider()
    st.subheader("Test Gateway Connection")
    if st.button("Ping Gateway"):
        status = _gateway_get("/v1/status")
        if status:
            st.success(f"Gateway is UP at {GATEWAY_URL}")
            providers = list(status.get("live", {}).keys())
            st.json({"active_providers": providers})
        else:
            st.error(f"Gateway unreachable at {GATEWAY_URL}. Start it with: `uv run main.py` in Gateway/llm_gatewayV7/")

    st.divider()
    st.subheader("Test Embed Endpoint")
    test_text = st.text_input("Test embedding text", value="Hello from the RAG system")
    if st.button("Test Embed"):
        try:
            import httpx
            r = httpx.post(
                f"{GATEWAY_URL}/v1/embed",
                json={"text": test_text, "task_type": "retrieval_query"},
                timeout=30,
            )
            if r.status_code == 200:
                d = r.json()
                st.success(f"Embed OK — provider={d['provider']} model={d['model']} dim={d['dim']} latency={d['latency_ms']}ms")
                st.json({"first_5_dims": d["embedding"][:5]})
            else:
                st.error(f"HTTP {r.status_code}: {r.text[:300]}")
        except Exception as e:
            st.error(f"Error: {e}")

    st.divider()
    st.subheader("Environment Variables (agent)")
    env_vars = {
        "TAVILY_API_KEY": os.environ.get("TAVILY_API_KEY", "(not set)"),
    }
    for k, v in env_vars.items():
        if v and v != "(not set)":
            masked = f"{v[:6]}…{v[-4:]}" if len(v) > 10 else "****"
            st.text(f"{k} = {masked}")
        else:
            st.warning(f"{k} is not set")

    st.divider()
    st.subheader("Memory Path")
    st.code(str(MEMORY_PATH), language="")
    st.text(f"Exists: {MEMORY_PATH.exists()}")
    if MEMORY_PATH.exists():
        st.text(f"Size: {MEMORY_PATH.stat().st_size:,} bytes")


# ── PAGE: PROVIDER STATUS ─────────────────────────────────────────────────────

elif page == "provider_status":
    st.title("📊 Provider Status")

    if not _gateway_up():
        st.error(f"Gateway not running at {GATEWAY_URL}. Start it with: `cd Gateway/llm_gatewayV7 && uv run main.py`")
        st.stop()

    st.success(f"Gateway UP at {GATEWAY_URL}")

    tab1, tab2, tab3, tab4 = st.tabs(["LLM Workers", "Embedders", "Routers", "Recent Calls"])

    with tab1:
        st.subheader("LLM Worker Providers")
        status = _gateway_get("/v1/status") or {}
        live = status.get("live", {})
        limits = status.get("limits", {})
        if live:
            rows = []
            for name, s in live.items():
                rows.append({
                    "Provider": name,
                    "Available": "✅" if not s.get("unavailable_until") else "❌",
                    "RPM used": s.get("rpm_used", 0),
                    "Cooldown (s)": round(s.get("cooldown_remaining", 0), 1),
                    "Backoff (s)": round(s.get("backoff_remaining", 0), 1),
                    "Reason": s.get("backoff_reason") or s.get("unavailable_reason") or "",
                })
            st.dataframe(rows, use_container_width=True)
        else:
            st.info("No provider status available.")

        caps = _gateway_get("/v1/capabilities") or {}
        if caps:
            st.subheader("Capabilities")
            st.json(caps)

    with tab2:
        st.subheader("Embedding Providers")
        emb = _gateway_get("/v1/embedders") or {}
        if emb:
            st.json({
                "order": emb.get("order"),
                "fixed_dim": emb.get("fixed_dim"),
                "max_input_chars": emb.get("max_input_chars"),
                "models": emb.get("models"),
            })
            live_emb = emb.get("live", {})
            if live_emb:
                rows = []
                for name, s in live_emb.items():
                    rows.append({
                        "Provider": name,
                        "RPM used": s.get("rpm_used", 0),
                        "RPM limit": s.get("rpm_limit", 0),
                        "Backoff (s)": round(s.get("backoff_remaining", 0), 1),
                        "Reason": s.get("backoff_reason", ""),
                    })
                st.dataframe(rows, use_container_width=True)
        else:
            st.info("No embedder status available.")

    with tab3:
        st.subheader("Router Pool")
        rp = _gateway_get("/v1/routers") or {}
        if rp:
            st.json({
                "order": rp.get("order"),
                "models": rp.get("models"),
                "tier_to_order": rp.get("tier_to_order"),
            })
            live_r = rp.get("live", {})
            if live_r:
                rows = []
                for name, s in live_r.items():
                    rows.append({
                        "Provider": name,
                        "Available": "✅" if not s.get("unavailable_until") else "❌",
                        "RPM used": s.get("rpm_used", 0),
                    })
                st.dataframe(rows, use_container_width=True)
        else:
            st.info("No router status available.")

    with tab4:
        st.subheader("Recent Calls")
        n = st.slider("Number of calls", 10, 500, 100)
        calls = _gateway_get(f"/v1/calls?limit={n}") or []
        if calls:
            rows = []
            for c in calls[:100]:
                rows.append({
                    "ID": c.get("id", ""),
                    "Provider": c.get("provider", ""),
                    "Model": (c.get("model") or "")[:30],
                    "Status": c.get("status", ""),
                    "Role": c.get("call_role", ""),
                    "Latency ms": c.get("latency_ms", 0),
                    "In tok": c.get("input_tokens", 0),
                    "Out tok": c.get("output_tokens", 0),
                    "Time": (c.get("created_at") or "")[:16],
                })
            st.dataframe(rows, use_container_width=True)
        else:
            st.info("No calls logged yet.")

    st.divider()
    if st.button("Refresh"):
        st.rerun()
