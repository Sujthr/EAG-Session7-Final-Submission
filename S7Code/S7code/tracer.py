"""tracer.py — records every agent run and appends to traces/query_traces.md.

One markdown section is appended per run. The file is never overwritten —
all history accumulates chronologically so you can compare runs over time.

Usage (from agent7.py):
    tracer = RunTracer(run_id, query)
    tracer.record_iter(...)   # once per loop iteration
    tracer.finish(final_answer)
    tracer.save()
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

TRACES_DIR = Path(__file__).parent / "traces"
TRACES_FILE = TRACES_DIR / "query_traces.md"


@dataclass
class IterRecord:
    iteration: int
    memory_hits: int
    goals: list[dict]           # [{"id": str, "text": str, "done": bool}]
    decision_type: str          # "answer" | "tool_call"
    tool_name: str = ""
    tool_args: dict = field(default_factory=dict)
    action_result: str = ""
    artifact_id: Optional[str] = None
    answer_text: str = ""


@dataclass
class RunTracer:
    run_id: str
    query: str
    _start: float = field(default_factory=time.monotonic)
    _started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    _iters: list[IterRecord] = field(default_factory=list)
    _final_answer: str = ""
    _elapsed: float = 0.0

    def record_iter(
        self,
        *,
        iteration: int,
        memory_hits: int,
        goals: list,                    # list[Goal] from schemas
        decision_type: str,
        tool_name: str = "",
        tool_args: dict | None = None,
        action_result: str = "",
        artifact_id: Optional[str] = None,
        answer_text: str = "",
    ) -> None:
        self._iters.append(IterRecord(
            iteration=iteration,
            memory_hits=memory_hits,
            goals=[{"id": g.id, "text": g.text, "done": g.done} for g in goals],
            decision_type=decision_type,
            tool_name=tool_name,
            tool_args=tool_args or {},
            action_result=action_result,
            artifact_id=artifact_id,
            answer_text=answer_text,
        ))

    def finish(self, final_answer: str) -> None:
        self._final_answer = final_answer
        self._elapsed = time.monotonic() - self._start

    def save(self) -> Path:
        TRACES_DIR.mkdir(exist_ok=True)

        lines: list[str] = []
        ts = self._started_at.strftime("%Y-%m-%d %H:%M:%S UTC")

        lines += [
            "\n---\n\n",
            f"## Run `{self.run_id}` — {ts}\n\n",
            f"**Query:** {self.query}\n\n",
            f"**Duration:** {self._elapsed:.1f}s &nbsp;|&nbsp; "
            f"**Iterations:** {len(self._iters)} &nbsp;|&nbsp; "
            f"**Memory hits (final iter):** "
            f"{self._iters[-1].memory_hits if self._iters else 0}\n\n",
        ]

        for rec in self._iters:
            lines.append(f"### Iteration {rec.iteration}\n\n")

            # Memory
            lines.append(f"| Memory hits | {rec.memory_hits} |\n")
            lines.append("|---|---|\n\n")

            # Goals
            lines.append("**Goals:**\n\n")
            for g in rec.goals:
                flag = "✓" if g["done"] else "○"
                lines.append(f"- {flag} `{g['id']}` — {g['text']}\n")
            lines.append("\n")

            # Decision
            if rec.decision_type == "tool_call":
                args_str = json.dumps(rec.tool_args, ensure_ascii=False)
                if len(args_str) > 200:
                    args_str = args_str[:200] + "…"
                lines.append(f"**Decision:** `TOOL_CALL` → `{rec.tool_name}({args_str})`\n\n")

                # Action result
                result_preview = rec.action_result.replace("\n", " ")
                if len(result_preview) > 400:
                    result_preview = result_preview[:400] + "…"
                lines.append(f"**Action result:**\n```\n{result_preview}\n```\n\n")

                if rec.artifact_id:
                    lines.append(f"**Artifact stored:** `{rec.artifact_id}`\n\n")
            else:
                answer_preview = rec.answer_text
                if len(answer_preview) > 400:
                    answer_preview = answer_preview[:400] + "…"
                lines.append(f"**Decision:** `ANSWER`\n\n> {answer_preview}\n\n")

        # Final answer block
        if self._final_answer:
            final_preview = self._final_answer
            if len(final_preview) > 800:
                final_preview = final_preview[:800] + "\n\n*(truncated — see full output in terminal)*"
            lines.append("---\n\n")
            lines.append("**Final Answer:**\n\n")
            lines.append(f"> {final_preview}\n")
        else:
            lines.append("**Final Answer:** *(none — loop exited without answer)*\n")

        with TRACES_FILE.open("a", encoding="utf-8") as fh:
            fh.writelines(lines)

        return TRACES_FILE
