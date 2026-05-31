"""Logging infrastructure for the automated query runner.

TeeWriter captures stdout to both the console and a per-query log file.
session_logger writes structured JSON events to a session-level log file.
"""

from __future__ import annotations

import io
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path


LOGS_DIR = Path(__file__).parent / "logs"


def _ensure_session_dir(session_id: str) -> Path:
    d = LOGS_DIR / session_id
    d.mkdir(parents=True, exist_ok=True)
    return d


# ── TeeWriter ────────────────────────────────────────────────────────────────

class TeeWriter(io.TextIOBase):
    """Writes to both an underlying stream and a log file simultaneously."""

    def __init__(self, original: io.TextIOBase, log_path: Path):
        self._original = original
        self._log = open(log_path, "w", encoding="utf-8", buffering=1)
        self._log.write(f"# Log started {datetime.utcnow().isoformat()}Z\n\n")

    def write(self, s: str) -> int:
        self._original.write(s)
        self._original.flush()
        if not self._log.closed:
            self._log.write(s)
            self._log.flush()
        return len(s)

    def flush(self) -> None:
        self._original.flush()
        if not self._log.closed:
            self._log.flush()

    def close_log(self) -> None:
        self._log.write(f"\n# Log closed {datetime.utcnow().isoformat()}Z\n")
        self._log.close()

    @property
    def encoding(self):
        return self._original.encoding

    @property
    def errors(self):
        return self._original.errors


# ── query-level context manager ───────────────────────────────────────────────

class QueryLogger:
    """Context manager: installs TeeWriter for the duration of one query run."""

    def __init__(self, session_dir: Path, query_label: str, query_text: str):
        self.session_dir = session_dir
        self.query_label = query_label
        self.query_text = query_text
        self.log_path = session_dir / f"{query_label}.log"
        self._tee: TeeWriter | None = None
        self._orig_stdout = None
        self.start_time: float = 0.0

    def __enter__(self) -> "QueryLogger":
        self.start_time = time.monotonic()
        self._orig_stdout = sys.stdout
        self._tee = TeeWriter(self._orig_stdout, self.log_path)
        sys.stdout = self._tee
        print(f"{'=' * 78}")
        print(f"QUERY [{self.query_label}]: {self.query_text}")
        print(f"Started: {datetime.utcnow().isoformat()}Z")
        print(f"{'=' * 78}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        elapsed = time.monotonic() - self.start_time
        print(f"\n{'=' * 78}")
        if exc_type is not None:
            print(f"QUERY [{self.query_label}] FAILED after {elapsed:.1f}s: {exc_val!r}")
        else:
            print(f"QUERY [{self.query_label}] completed in {elapsed:.1f}s")
        print(f"{'=' * 78}\n")
        if self._tee is not None:
            self._tee.close_log()
        sys.stdout = self._orig_stdout
        return False  # propagate exceptions


# ── session-level structured log ─────────────────────────────────────────────

class SessionLog:
    """Append-only structured JSON log for the entire session."""

    def __init__(self, session_dir: Path):
        self.path = session_dir / "session.jsonl"

    def _append(self, event: dict) -> None:
        event.setdefault("ts", datetime.utcnow().isoformat() + "Z")
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")

    def session_start(self, session_id: str, queries: list[dict]) -> None:
        self._append({
            "event": "session_start",
            "session_id": session_id,
            "query_count": len(queries),
            "queries": [{"label": q["label"], "text": q["text"]} for q in queries],
        })

    def query_start(self, label: str, text: str) -> None:
        self._append({"event": "query_start", "label": label, "text": text})

    def query_done(self, label: str, elapsed_s: float, answer: str, error: str = "") -> None:
        self._append({
            "event": "query_done",
            "label": label,
            "elapsed_s": round(elapsed_s, 2),
            "answer_preview": answer[:300],
            "error": error,
            "status": "error" if error else "ok",
        })

    def session_end(self, session_id: str, total_s: float, ok: int, failed: int) -> None:
        self._append({
            "event": "session_end",
            "session_id": session_id,
            "total_elapsed_s": round(total_s, 2),
            "queries_ok": ok,
            "queries_failed": failed,
        })


# ── summary writers ───────────────────────────────────────────────────────────

def write_summary(
    session_dir: Path,
    session_id: str,
    results: list[dict],
    total_elapsed: float,
) -> None:
    """Write summary.json and summary.md for the session."""
    ok = sum(1 for r in results if not r.get("error"))
    failed = sum(1 for r in results if r.get("error"))

    summary = {
        "session_id": session_id,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "total_elapsed_s": round(total_elapsed, 2),
        "queries_total": len(results),
        "queries_ok": ok,
        "queries_failed": failed,
        "results": results,
    }
    (session_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    md_lines = [
        f"# Session {session_id} Summary",
        "",
        f"- **Date**: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC",
        f"- **Total time**: {total_elapsed:.1f}s",
        f"- **Queries**: {len(results)} total, {ok} succeeded, {failed} failed",
        "",
        "## Results",
        "",
        "| Label | Status | Time (s) | Answer Preview |",
        "|-------|--------|----------|----------------|",
    ]
    for r in results:
        status = "FAIL" if r.get("error") else "OK"
        preview = (r.get("answer") or r.get("error") or "")[:80].replace("|", " ")
        md_lines.append(
            f"| {r['label']} | {status} | {r.get('elapsed_s', 0):.1f} | {preview} |"
        )

    md_lines += ["", "## Query Details", ""]
    for r in results:
        md_lines += [
            f"### [{r['label']}] {r['query']}",
            "",
            f"**Status**: {'FAILED' if r.get('error') else 'Success'}  ",
            f"**Elapsed**: {r.get('elapsed_s', 0):.1f}s  ",
            f"**Log**: `{r['label']}.log`",
            "",
        ]
        if r.get("error"):
            md_lines += [f"**Error**: `{r['error']}`", ""]
        else:
            answer = r.get("answer", "")
            md_lines += [f"**Answer**:", "", f"> {answer[:500]}", ""]

    (session_dir / "summary.md").write_text(
        "\n".join(md_lines), encoding="utf-8"
    )
    print(f"[session] summary written to {session_dir}")
