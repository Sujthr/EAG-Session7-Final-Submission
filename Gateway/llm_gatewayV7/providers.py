"""Provider adapters — OpenAI only.

Each provider implements:
  async chat(messages, *, max_tokens, temperature, model, tools, tool_choice,
             reasoning, response_format, system_blocks) -> dict

The returned dict is normalised:
  {
    "text": str,
    "tool_calls": [ {"id","name","arguments"} ],
    "input_tokens": int, "output_tokens": int,
    "cache_creation_input_tokens": int, "cache_read_input_tokens": int,
    "stop_reason": "tool_use"|"end_turn"|"max_tokens",
    "model": str,
    "tool_call_dialect": "native"|"none",
    "reasoning_applied": bool,
  }
"""
from __future__ import annotations
import os, json, uuid
from typing import AsyncIterator, Optional
import httpx


class ProviderError(Exception):
    def __init__(self, msg, status=None, retryable=True):
        super().__init__(msg)
        self.status = status
        self.retryable = retryable


def _flatten_system(system_blocks) -> tuple[str, list[dict], bool]:
    """Returns (joined_text, raw_blocks, has_cache_marker)."""
    if system_blocks is None:
        return "", [], False
    if isinstance(system_blocks, str):
        return system_blocks, [{"text": system_blocks, "cache": False}], False
    blocks = []
    has_cache = False
    parts = []
    for b in system_blocks:
        if isinstance(b, dict):
            t = b.get("text", "")
            c = bool(b.get("cache", False))
        else:
            t = getattr(b, "text", "")
            c = bool(getattr(b, "cache", False))
        blocks.append({"text": t, "cache": c})
        parts.append(t)
        if c:
            has_cache = True
    return "\n".join(parts), blocks, has_cache


def _empty_result(model: str) -> dict:
    return {
        "text": "", "tool_calls": [],
        "input_tokens": 0, "output_tokens": 0,
        "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0,
        "stop_reason": "end_turn", "model": model,
        "tool_call_dialect": "none", "reasoning_applied": False,
    }


class BaseProvider:
    name: str = ""

    def __init__(self, api_key: str, model: str, base_url: str = ""):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url

    async def chat(self, messages, *, max_tokens=2048, temperature=0.7, model=None,
                   tools=None, tool_choice=None, reasoning=None, response_format=None,
                   system_blocks=None, cache_system=False) -> dict:
        raise NotImplementedError

    async def stream(self, messages, *, max_tokens=2048, temperature=0.7, model=None,
                     tools=None, tool_choice=None, reasoning=None, response_format=None,
                     system_blocks=None, cache_system=False) -> AsyncIterator[str]:
        result = await self.chat(messages, max_tokens=max_tokens, temperature=temperature,
                                 model=model, tools=tools, tool_choice=tool_choice,
                                 reasoning=reasoning, response_format=response_format,
                                 system_blocks=system_blocks, cache_system=cache_system)
        if result["text"]:
            yield result["text"]


REASONING_MODEL_HINTS = ("o1", "o3", "o4", "gpt-5")


def _model_supports_reasoning(model: str) -> bool:
    m = (model or "").lower()
    return any(h in m for h in REASONING_MODEL_HINTS)


class OpenAICompatProvider(BaseProvider):
    capabilities = {
        "tools": True, "caching": True, "reasoning": False,
        "structured": True, "parallel_tools": True,
    }

    def _headers(self):
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def _translate_tools(self, tools):
        out = []
        for t in tools or []:
            d = t if isinstance(t, dict) else t.model_dump()
            out.append({
                "type": "function",
                "function": {
                    "name": d["name"],
                    "description": d.get("description", ""),
                    "parameters": d.get("input_schema") or {"type": "object", "properties": {}},
                },
            })
        return out

    def _translate_messages(self, messages, system_text):
        out = []
        if system_text:
            out.append({"role": "system", "content": system_text})
        for m in messages:
            r = m.get("role")
            if r == "system":
                if not system_text:
                    out.append({"role": "system", "content": m.get("content", "")})
                continue
            if r == "tool":
                out.append({
                    "role": "tool",
                    "tool_call_id": m.get("tool_call_id") or m.get("id") or "",
                    "content": (m.get("content", "")
                                if isinstance(m.get("content"), str)
                                else json.dumps(m.get("content"))),
                })
                continue
            if r == "assistant" and m.get("tool_calls"):
                tcs = []
                for tc in m["tool_calls"]:
                    tcs.append({
                        "id": tc.get("id") or f"call_{uuid.uuid4().hex[:8]}",
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": json.dumps(tc.get("arguments") or {}),
                        },
                    })
                out.append({"role": "assistant", "content": m.get("content") or "", "tool_calls": tcs})
                continue
            out.append({"role": r, "content": m.get("content", "")})
        return out

    def _apply_response_format(self, body, response_format):
        if not response_format:
            return
        rf = response_format if isinstance(response_format, dict) else response_format.model_dump(by_alias=True)
        if rf.get("type") == "json_schema" and rf.get("schema"):
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": rf.get("name", "out"),
                    "schema": rf["schema"],
                    "strict": bool(rf.get("strict", True)),
                },
            }
        elif rf.get("type") == "json_object":
            body["response_format"] = {"type": "json_object"}

    def _apply_reasoning(self, body, reasoning, model):
        if not reasoning or reasoning == "off":
            return False
        if not _model_supports_reasoning(model):
            return False
        body["reasoning_effort"] = reasoning
        return True

    async def chat(self, messages, *, max_tokens=2048, temperature=0.7, model=None,
                   tools=None, tool_choice=None, reasoning=None, response_format=None,
                   system_blocks=None, cache_system=False):
        m = model or self.model
        system_text, _, _ = _flatten_system(system_blocks)
        body = {
            "model": m,
            "messages": self._translate_messages(messages, system_text),
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        if tools:
            body["tools"] = self._translate_tools(tools)
            if tool_choice is not None:
                body["tool_choice"] = tool_choice if isinstance(tool_choice, (str, dict)) else "auto"
        self._apply_response_format(body, response_format)
        reasoning_applied = self._apply_reasoning(body, reasoning, m)

        async with httpx.AsyncClient(timeout=120) as c:
            r = await c.post(f"{self.base_url}/chat/completions", headers=self._headers(), json=body)
            if r.status_code != 200:
                txt = r.text
                if reasoning_applied and "reasoning_effort" in txt:
                    body.pop("reasoning_effort", None)
                    reasoning_applied = False
                    r = await c.post(f"{self.base_url}/chat/completions", headers=self._headers(), json=body)
                if r.status_code != 200 and "json_schema" in str(
                    (body.get("response_format") or {}).get("type", "")
                ):
                    body["response_format"] = {"type": "json_object"}
                    r = await c.post(f"{self.base_url}/chat/completions", headers=self._headers(), json=body)
                if r.status_code != 200:
                    raise ProviderError(
                        f"{self.name} HTTP {r.status_code}: {r.text[:300]}",
                        status=r.status_code,
                        retryable=(r.status_code not in (400, 401)),
                    )
            d = r.json()
            choice = (d.get("choices") or [{}])[0]
            msg = choice.get("message") or {}
            text = msg.get("content") or ""
            tool_calls_out = []
            for tc in (msg.get("tool_calls") or []):
                fn = tc.get("function") or {}
                args_str = fn.get("arguments") or "{}"
                try:
                    args = json.loads(args_str) if isinstance(args_str, str) else args_str
                except Exception:
                    args = {"_raw": args_str}
                tool_calls_out.append({
                    "id": tc.get("id") or f"call_{uuid.uuid4().hex[:8]}",
                    "name": fn.get("name", ""),
                    "arguments": args,
                })
            usage = d.get("usage") or {}
            details = usage.get("prompt_tokens_details") or {}
            cache_read = details.get("cached_tokens", 0) or 0
            stop = choice.get("finish_reason") or "stop"
            stop_norm = "tool_use" if tool_calls_out else (
                "max_tokens" if stop == "length" else "end_turn"
            )
            return {
                "text": text or "",
                "tool_calls": tool_calls_out,
                "input_tokens": usage.get("prompt_tokens", 0) or 0,
                "output_tokens": usage.get("completion_tokens", 0) or 0,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": cache_read,
                "stop_reason": stop_norm,
                "model": m,
                "tool_call_dialect": "native",
                "reasoning_applied": reasoning_applied,
            }

    async def stream(self, messages, *, max_tokens=2048, temperature=0.7, model=None,
                     tools=None, tool_choice=None, reasoning=None, response_format=None,
                     system_blocks=None, cache_system=False):
        m = model or self.model
        system_text, _, _ = _flatten_system(system_blocks)
        body = {
            "model": m,
            "messages": self._translate_messages(messages, system_text),
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }
        if tools:
            body["tools"] = self._translate_tools(tools)
            if tool_choice is not None:
                body["tool_choice"] = tool_choice if isinstance(tool_choice, (str, dict)) else "auto"
        self._apply_response_format(body, response_format)
        self._apply_reasoning(body, reasoning, m)
        async with httpx.AsyncClient(timeout=120) as c:
            async with c.stream("POST", f"{self.base_url}/chat/completions",
                                headers=self._headers(), json=body) as r:
                if r.status_code != 200:
                    text = (await r.aread()).decode("utf-8", "ignore")[:300]
                    raise ProviderError(f"{self.name} HTTP {r.status_code}: {text}",
                                        status=r.status_code)
                async for line in r.aiter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    payload = line[6:]
                    if payload.strip() == "[DONE]":
                        return
                    try:
                        d = json.loads(payload)
                        delta = d["choices"][0].get("delta", {})
                        if delta.get("content"):
                            yield delta["content"]
                        if delta.get("tool_calls"):
                            yield "[[TOOL_CALL_DELTA]] " + json.dumps(delta["tool_calls"])
                    except Exception:
                        continue


class OpenAIProvider(OpenAICompatProvider):
    """OpenAI API — gpt-4.1-mini by default.

    Set OPENAI_MODEL env var to override (e.g. gpt-4o-mini, gpt-4o).
    """
    name = "openai"
    capabilities = {**OpenAICompatProvider.capabilities, "reasoning": True}

    def __init__(self, api_key, model):
        super().__init__(api_key, model, "https://api.openai.com/v1")


def model_capabilities(provider_name: str, model: str, default_caps: dict) -> dict:
    caps = dict(default_caps)
    if provider_name == "openai":
        caps["reasoning"] = _model_supports_reasoning(model)
    return caps


def build_providers(cache_store=None):
    """Worker pool — OpenAI only."""
    k = os.getenv("OPENAI_API_KEY", "").strip()
    model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    if not k or not (k.startswith("sk-") or k.startswith("sk-proj-")):
        raise RuntimeError(
            "OPENAI_API_KEY not set or invalid in Gateway/.env. "
            "Set it to a valid sk-... or sk-proj-... key."
        )
    provider = OpenAIProvider(k, model)
    print(f"[providers] OpenAI active: {model}")
    return {"openai": provider}


def build_router_providers():
    """No router LLM — single provider needs no routing."""
    return {}
