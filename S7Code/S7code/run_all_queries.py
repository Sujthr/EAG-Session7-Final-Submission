"""run_all_queries.py — Automated sequential runner for the exact Session-7 benchmark.

Runs queries A-H (mandatory, from CLAUDE_FINAL_ASSIGNMENT_SPEC.md) then Q1-Q5
(custom) without any user interaction. Each query gets its own log file. A
session summary is written to logs/<session_id>/summary.{json,md} at the end.

Usage (from S7Code/S7code directory):
    uv run python run_all_queries.py

Or invoked automatically by start.ps1.
"""

from __future__ import annotations

import asyncio
import sys
import time
from datetime import datetime
from pathlib import Path

# ── exact query manifest from CLAUDE_FINAL_ASSIGNMENT_SPEC.md ────────────────

QUERIES: list[dict] = [
    # ── Mandatory A-H ─────────────────────────────────────────────────────────
    {
        "label": "A",
        "text": (
            "Fetch https://en.wikipedia.org/wiki/Claude_Shannon and tell me "
            "his birth date, death date, and three key contributions to "
            "information theory."
        ),
    },
    {
        "label": "B",
        "text": (
            "Find 3 family-friendly things to do in Tokyo this weekend. "
            "Check Saturday's weather forecast there and tell me which one "
            "is most appropriate."
        ),
    },
    {
        "label": "C1",
        "text": (
            "My mom's birthday is 15 May 2026. "
            "Remember that and create reminders for two weeks before and on the day."
        ),
    },
    {
        "label": "C2",
        "text": "When is mom's birthday?",
    },
    {
        "label": "D",
        "text": (
            'Search for "Python asyncio best practices", read the top 3 results, '
            "and give me a short numbered list of the advice they agree on."
        ),
    },
    {
        "label": "E",
        "text": (
            "Index the file papers/attention.md and tell me what the three key "
            "contributions of the Transformer architecture are according to this paper."
        ),
    },
    {
        "label": "F1",
        "text": (
            "Index every .md file under papers/. "
            "Confirm how many chunks were indexed in total."
        ),
    },
    {
        "label": "F2",
        "text": (
            "Across the papers I have indexed, what do they say about "
            "chain-of-thought reasoning?"
        ),
    },
    {
        "label": "G",
        "text": (
            "Across these papers, how do they handle the credit assignment problem?"
        ),
    },
    {
        "label": "H",
        "text": (
            "Compare how the ReAct paper and the Chain-of-Thought paper differ "
            "in their treatment of intermediate reasoning."
        ),
    },
    # ── Custom Q1-Q5 ──────────────────────────────────────────────────────────
    {
        "label": "Q1",
        "text": "How do these papers achieve model alignment?",
    },
    {
        "label": "Q2",
        "text": "How do these papers reduce memory requirements during training?",
    },
    {
        "label": "Q3",
        "text": "Compare ReAct and AutoGPT task execution.",
    },
    {
        "label": "Q4",
        "text": "Which papers discuss reasoning before acting?",
    },
    {
        "label": "Q5",
        "text": "What strategies improve retrieval quality?",
    },
]

INTER_QUERY_DELAY = 5  # seconds between queries to avoid rate-limit bursts


# ── runner ────────────────────────────────────────────────────────────────────

async def _run_one(query_text: str) -> str:
    import agent7
    return await agent7.run(query_text)


def main() -> None:
    from log_setup import (
        QueryLogger,
        SessionLog,
        LOGS_DIR,
        write_summary,
        _ensure_session_dir,
    )

    session_id = "session_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = _ensure_session_dir(session_id)
    session_log = SessionLog(session_dir)
    session_log.session_start(session_id, QUERIES)

    print(f"\n{'#' * 78}")
    print(f"# Session: {session_id}")
    print(f"# Queries: {len(QUERIES)}")
    print(f"# Logs:    {session_dir}")
    print(f"{'#' * 78}\n")

    results: list[dict] = []
    session_start = time.monotonic()

    for i, q in enumerate(QUERIES):
        label = q["label"]
        text = q["text"]

        session_log.query_start(label, text)

        with QueryLogger(session_dir, label, text) as qlog:
            t0 = time.monotonic()
            answer = ""
            error = ""
            try:
                answer = asyncio.run(_run_one(text))
            except Exception as exc:
                error = repr(exc)
                print(f"[runner] EXCEPTION in query {label}: {exc!r}")
            elapsed = time.monotonic() - t0

        session_log.query_done(label, elapsed, answer, error)
        results.append({
            "label": label,
            "query": text,
            "elapsed_s": round(elapsed, 2),
            "answer": answer,
            "error": error,
            "log": f"{label}.log",
        })

        if i < len(QUERIES) - 1:
            print(f"[runner] sleeping {INTER_QUERY_DELAY}s before next query...")
            time.sleep(INTER_QUERY_DELAY)

    total_elapsed = time.monotonic() - session_start
    ok = sum(1 for r in results if not r["error"])
    failed = len(results) - ok

    session_log.session_end(session_id, total_elapsed, ok, failed)
    write_summary(session_dir, session_id, results, total_elapsed)

    print(f"\n{'#' * 78}")
    print(f"# Session complete: {ok}/{len(results)} queries succeeded")
    print(f"# Total time: {total_elapsed:.1f}s")
    print(f"# Logs: {session_dir}")
    print(f"{'#' * 78}\n")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
