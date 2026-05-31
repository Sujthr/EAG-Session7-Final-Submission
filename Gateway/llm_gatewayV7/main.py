import os, time, json
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from jsonschema import Draft202012Validator, ValidationError

ROOT = Path(__file__).parent
load_dotenv(ROOT.parent / ".env")

import db
import providers as P
from router import Router, LIMITS, SHORTCUTS, resolve
from schemas import ChatRequest, ChatResponse, ToolCall, EmbedRequest, EmbedResponse
import embedders as E

DEFAULT_ORDER = ["openai"]
ORDER = [x.strip() for x in os.getenv("LLM_ORDER", "openai").split(",") if x.strip()]
PORT = int(os.getenv("GATEWAY_V7_PORT", "8107"))


def _estimate_tokens(text: str) -> int:
    return int(len(text.split()) * 1.4)


def _backoff_for(err: Exception, has_model_override: bool = False):
    msg = str(err).lower()
    status = getattr(err, "status", None)
    if status == 429:
        if "quota" in msg or "rpm" in msg or "per minute" in msg:
            return 60, "RPM quota burned"
        if "per day" in msg or "daily" in msg:
            return 3600, "RPD quota burned"
        return 30, "rate limited"
    if status and 500 <= status < 600:
        return 20, f"upstream {status}"
    if status == 408 or "timeout" in msg:
        return 10, "timeout"
    if status in (401, 403):
        if has_model_override:
            return 0, ""
        return 600, "auth error"
    if status == 404 and has_model_override:
        return 0, ""
    return 0, ""


def _attempts_str(attempts):
    return "; ".join(f"{a['provider']}:{a['reason']}" for a in attempts)


def _required_caps(req: ChatRequest):
    caps = []
    if req.tools:
        caps.append("tools")
    if req.reasoning and req.reasoning != "off":
        caps.append("reasoning")
    if req.response_format:
        caps.append("structured")
    return caps


def _validate_structured(text: str, schema: dict):
    try:
        obj = json.loads(text)
    except Exception as e:
        raise ValueError(f"output is not JSON: {e}")
    Draft202012Validator(schema).validate(obj)
    return obj


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init()
    app.state.providers = P.build_providers()
    app.state.router = Router(app.state.providers, ORDER)
    app.state.embedders, app.state.embed_order = E.build_embedders()
    yield


app = FastAPI(title="LLM Gateway V7 (OpenAI)", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(ROOT / "static")), name="static")


def _normalize_messages(req: ChatRequest):
    if req.messages:
        return list(req.messages)
    return [{"role": "user", "content": req.prompt or ""}]


def _system_blocks(req: ChatRequest):
    if req.system is None:
        return None
    if isinstance(req.system, str):
        if req.cache_system:
            return [{"text": req.system, "cache": True}]
        return req.system
    return [b.model_dump() if hasattr(b, "model_dump") else b for b in req.system]


def _est_tokens(messages, system_blocks, max_tokens):
    chars = sum(len(str(m.get("content", ""))) for m in messages)
    if isinstance(system_blocks, str):
        chars += len(system_blocks)
    elif isinstance(system_blocks, list):
        for b in system_blocks:
            chars += len(b.get("text", "") if isinstance(b, dict) else "")
    return chars // 4 + max_tokens


@app.post("/v1/chat")
async def chat(req: ChatRequest):
    router = app.state.router
    messages = _normalize_messages(req)
    system_blocks = _system_blocks(req)
    prompt_text = "".join(str(m.get("content", "")) for m in messages)
    est = _est_tokens(messages, system_blocks, req.max_tokens)
    explicit_override = bool(req.provider)
    required_caps = _required_caps(req)

    # auto_route is accepted for compatibility but not used for tier routing —
    # single provider means no routing decision is needed.
    candidates = router.candidates(req.provider) if req.provider else list(router.order)

    if req.provider and not candidates:
        raise HTTPException(400, f"unknown provider '{req.provider}'. Try: {list(router.providers)} or shortcuts {list(SHORTCUTS)}")

    all_attempts: list[dict] = []
    last_err = None

    for _ in range(len(candidates) + 1):
        name, atts = router.pick(est, candidates, required_caps=required_caps)
        all_attempts.extend(atts)
        if name is None:
            break

        provider = router.providers[name]
        t0 = time.time()
        router.state[name].record(0)

        try:
            if req.stream:
                async def gen():
                    try:
                        agg = []
                        async for chunk in provider.stream(
                            messages,
                            max_tokens=req.max_tokens,
                            temperature=req.temperature,
                            model=req.model,
                            tools=req.tools,
                            tool_choice=req.tool_choice,
                            reasoning=req.reasoning,
                            response_format=req.response_format,
                            system_blocks=system_blocks,
                            cache_system=bool(req.cache_system),
                        ):
                            agg.append(chunk)
                            if chunk.startswith("[[TOOL_CALL_DELTA]]"):
                                yield f"data: {json.dumps({'provider': name, 'tool_call_delta': chunk[len('[[TOOL_CALL_DELTA]] '):]})}\n\n"
                            else:
                                yield f"data: {json.dumps({'provider': name, 'delta': chunk})}\n\n"
                        text = "".join(agg)
                        latency = int((time.time() - t0) * 1000)
                        db.log_call(provider=name, model=req.model or provider.model,
                                    latency_ms=latency, status="ok",
                                    prompt_chars=len(prompt_text), response_chars=len(text))
                        yield f"data: {json.dumps({'done': True, 'provider': name})}\n\n"
                    except Exception as e:
                        db.log_call(provider=name, model=req.model or provider.model,
                                    status="error", error=str(e)[:500],
                                    latency_ms=int((time.time() - t0) * 1000),
                                    prompt_chars=len(prompt_text))
                        yield f"data: {json.dumps({'error': str(e)[:300]})}\n\n"
                return StreamingResponse(gen(), media_type="text/event-stream")

            result = await provider.chat(
                messages,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
                model=req.model,
                tools=req.tools,
                tool_choice=req.tool_choice,
                reasoning=req.reasoning,
                response_format=req.response_format,
                system_blocks=system_blocks,
                cache_system=bool(req.cache_system),
            )
            latency = int((time.time() - t0) * 1000)

            parsed = None
            if req.response_format and req.response_format.schema_ and not result["tool_calls"]:
                try:
                    parsed = _validate_structured(result["text"], req.response_format.schema_)
                except (ValueError, ValidationError) as ve:
                    fix_msgs = list(messages) + [
                        {"role": "assistant", "content": result["text"]},
                        {"role": "user", "content": (
                            f"Your previous reply did not match the required JSON schema: {ve}. "
                            "Reply ONLY with valid JSON conforming to the schema."
                        )},
                    ]
                    result = await provider.chat(
                        fix_msgs,
                        max_tokens=req.max_tokens,
                        temperature=0,
                        model=req.model,
                        response_format=req.response_format,
                        system_blocks=system_blocks,
                        cache_system=bool(req.cache_system),
                    )
                    try:
                        parsed = _validate_structured(result["text"], req.response_format.schema_)
                    except (ValueError, ValidationError) as ve2:
                        raise HTTPException(503, f"structured output failed validation: {ve2}")

            tokens = (result["input_tokens"] or 0) + (result["output_tokens"] or 0)
            router.state[name].tokens_today += tokens
            router.state[name].tokens_minute.append((time.time(), tokens))
            db.log_call(
                provider=name, model=result["model"],
                input_tokens=result["input_tokens"], output_tokens=result["output_tokens"],
                cache_read_tokens=result["cache_read_input_tokens"],
                latency_ms=latency, status="ok",
                prompt_chars=len(prompt_text), response_chars=len(result["text"]),
                override=req.provider, attempted=_attempts_str(all_attempts),
                tool_calls=len(result["tool_calls"]),
                reasoning_applied=result["reasoning_applied"],
                tool_dialect=result["tool_call_dialect"],
                call_role="worker",
            )
            return ChatResponse(
                provider=name,
                model=result["model"],
                text=result["text"],
                tool_calls=[ToolCall(**tc) for tc in result["tool_calls"]],
                stop_reason=result["stop_reason"],
                input_tokens=result["input_tokens"],
                output_tokens=result["output_tokens"],
                cache_creation_input_tokens=result["cache_creation_input_tokens"],
                cache_read_input_tokens=result["cache_read_input_tokens"],
                latency_ms=latency,
                tool_call_dialect=result["tool_call_dialect"],
                reasoning_applied=result["reasoning_applied"],
                parsed=parsed,
                attempted=all_attempts,
            ).model_dump()

        except P.ProviderError as e:
            last_err = str(e)
            secs, reason = _backoff_for(e, has_model_override=bool(req.model))
            if secs > 0:
                router.state[name].mark_unavailable(secs, reason)
            db.log_call(provider=name, model=req.model or provider.model,
                        status="error", error=str(e)[:500],
                        latency_ms=int((time.time() - t0) * 1000),
                        prompt_chars=len(prompt_text))
            if explicit_override:
                raise HTTPException(502, f"{name} failed: {e}")
            candidates = [c for c in candidates if c != name]
            continue
        except HTTPException:
            raise
        except Exception as e:
            last_err = str(e)
            secs, reason = _backoff_for(e, has_model_override=bool(req.model))
            if secs > 0:
                router.state[name].mark_unavailable(secs, reason)
            db.log_call(provider=name, model=req.model or provider.model,
                        status="error", error=str(e)[:500],
                        latency_ms=int((time.time() - t0) * 1000),
                        prompt_chars=len(prompt_text))
            if explicit_override:
                raise HTTPException(502, f"{name} failed: {e}")
            candidates = [c for c in candidates if c != name]
            continue

    raise HTTPException(503, f"all providers unavailable. attempts: {all_attempts}. last_error: {last_err}")


@app.post("/v1/embed")
async def embed(req: EmbedRequest):
    embedders = app.state.embedders
    if not embedders:
        raise HTTPException(503, "no embedding providers configured")

    if len(req.text) > E.MAX_INPUT_CHARS:
        raise HTTPException(
            413,
            f"text is {len(req.text)} chars; embed input is capped at "
            f"{E.MAX_INPUT_CHARS} chars. Chunk the input and embed each chunk.",
        )

    t0 = time.time()
    try:
        name, result, attempts, latency = await E.embed_with_failover(
            embedders, req.text, req.task_type, explicit=req.provider
        )
    except E.EmbedderError as e:
        latency = int((time.time() - t0) * 1000)
        db.log_call(
            provider=req.provider or "(any)", model="(none)",
            status="error", error=str(e)[:500], latency_ms=latency,
            prompt_chars=len(req.text), override=req.provider, call_role="embed",
        )
        if req.provider:
            if e.status == 429:
                raise HTTPException(429, f"{req.provider} rate-limited: {e}")
            if e.status == 400:
                raise HTTPException(400, str(e))
            raise HTTPException(502, f"{req.provider} embed failed: {e}")
        raise HTTPException(503, str(e))

    db.log_call(
        provider=name, model=result["model"], status="ok", latency_ms=latency,
        prompt_chars=len(req.text), override=req.provider,
        attempted=_attempts_str(attempts), call_role="embed", embed_dim=result["dim"],
    )
    return EmbedResponse(
        provider=name,
        model=result["model"],
        embedding=result["embedding"],
        dim=result["dim"],
        latency_ms=latency,
        attempted=attempts,
    ).model_dump()


@app.get("/v1/embedders")
async def list_embedders():
    return {
        "order": app.state.embed_order,
        "models": {e.name: e.model for e in app.state.embedders},
        "fixed_dim": E.EMBED_DIM,
        "max_input_chars": E.MAX_INPUT_CHARS,
        "backoff_steps_s": E.BACKOFF_STEPS,
        "live": {e.name: e.state.snapshot() for e in app.state.embedders},
        "today": db.aggregate(call_role="embed"),
    }


@app.get("/v1/providers")
async def list_providers():
    r = app.state.router
    return {
        "order": r.order,
        "providers": list(r.providers.keys()),
        "shortcuts": SHORTCUTS,
        "limits": LIMITS,
        "models": {n: p.model for n, p in r.providers.items()},
    }


@app.get("/v1/capabilities")
async def capabilities():
    r = app.state.router
    out = {}
    for name, p in r.providers.items():
        caps = dict(getattr(p, "capabilities", {}))
        caps = P.model_capabilities(name, p.model, caps)
        caps["model"] = p.model
        limits = LIMITS.get(name, LIMITS["openai"])
        caps.update({
            "max_ctx": limits["max_ctx"],
            "rpm": limits["rpm"],
            "rpd": limits["rpd"],
        })
        out[name] = caps
    return out


@app.get("/v1/status")
async def status():
    r = app.state.router
    return {
        "order": r.order,
        "live": r.all_status(),
        "today": db.aggregate(call_role="worker"),
        "limits": LIMITS,
    }


@app.get("/v1/routers")
async def routers():
    """Stub endpoint for gateway liveness check compatibility."""
    return {"order": [], "providers": {}, "note": "No router LLM in OpenAI-only mode"}


@app.get("/v1/calls")
async def calls(limit: int = 100, provider: Optional[str] = None, status: Optional[str] = None):
    return db.recent(limit=limit, provider=provider, status=status)


@app.get("/", response_class=HTMLResponse)
async def index():
    return FileResponse(str(ROOT / "static" / "dashboard.html"))


@app.get("/help", response_class=HTMLResponse)
async def help_page():
    return FileResponse(str(ROOT / "static" / "help.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=False)
