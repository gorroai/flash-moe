#!/usr/bin/env python3
"""
Flash-MoE Phase 3 HTTP Wrapper — persistent process edition
OpenAI-compatible streaming API on port 8084.

Architecture:
  One infer subprocess is kept alive for the lifetime of the server using the
  new --stdin flag.  The binary loads the model once, then loops:
    - reads {"prompt":"...","max_tokens":N}\\n from stdin
    - streams tokens to stdout (one fflush per token)
    - writes \\x00 as end-of-response marker

  Cold start happens once at server startup (~10-30 s to load model + Metal).
  Every subsequent request pays only generation time — no re-load.

  asyncio.Semaphore(1) serialises requests so only one generation runs at a
  time.  Waiting requests queue until the current generation finishes.

Stdin protocol:
  Request  →  one JSON line: {"prompt":"<full qwen text>","max_tokens":<N>}\\n
  Response ←  raw UTF-8 tokens, each flushed immediately
              \\x00  (null byte) marks end of response

Usage:
  cd ~/flash-moe && python3 server/server.py
"""

import asyncio
import json
import logging
import os
import signal
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

WRAPPER_PORT = 8084

HOME         = os.path.expanduser("~")
REPO_ROOT    = os.path.join(HOME, "flash-moe")
INFER_BIN    = os.path.join(REPO_ROOT, "metal_infer", "infer")
MODEL_DIR    = os.path.join(HOME, "Models", "flash_mlx_4bit")
GGUF_EMBED   = os.path.join(MODEL_DIR, "gguf", "embedding_q8_0.bin")
GGUF_LM_HEAD = os.path.join(MODEL_DIR, "gguf", "lm_head_q6.bin")
MODEL_ID     = "qwen3.5-397b-a17b"

# How long to wait for the model to finish loading (binary prints "[stdin] Ready")
READY_TIMEOUT = 300.0

# Bytes read from stdout per iteration — small for low first-token latency
READ_CHUNK = 32

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("flash-moe")

# ─────────────────────────────────────────────────────────────────────────────
# Global state
# ─────────────────────────────────────────────────────────────────────────────

_proc:    Optional[asyncio.subprocess.Process] = None
_sem:     Optional[asyncio.Semaphore] = None   # Semaphore(1) — one request at a time
_ready:   asyncio.Event                        # set when binary prints "[stdin] Ready"

# ─────────────────────────────────────────────────────────────────────────────
# Chat format
# ─────────────────────────────────────────────────────────────────────────────

_DEFAULT_SYSTEM = "You are a helpful assistant."


def build_qwen_prompt(messages: list[dict]) -> str:
    """
    Convert OpenAI messages array → Qwen3.5 im_start/im_end chat format.

    The trailing <|im_start|>assistant (no content) is the generation cue.
    """
    parts: list[str] = []
    has_system = any(m.get("role") == "system" for m in messages)
    if not has_system:
        parts.append(f"<|im_start|>system\n{_DEFAULT_SYSTEM}<|im_end|>")

    for msg in messages:
        role    = msg.get("role", "user")
        content = (msg.get("content") or "").rstrip()
        if role == "assistant":
            parts.append(f"<|im_start|>assistant\n{content}<|im_end|>")
        else:
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")

    parts.append("<|im_start|>assistant")
    return "\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Persistent process management
# ─────────────────────────────────────────────────────────────────────────────

def _infer_cmd() -> list[str]:
    missing = [p for p in (INFER_BIN, MODEL_DIR, GGUF_EMBED, GGUF_LM_HEAD)
               if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"Required paths missing: {missing}")
    return [
        INFER_BIN,
        "--model",          MODEL_DIR,
        "--q3-experts",
        "--gguf-embedding", GGUF_EMBED,
        "--gguf-lm-head",   GGUF_LM_HEAD,
        "--cache-io-split", "4",
        "--predict",
        "--stdin",
    ]


async def _start_process() -> asyncio.subprocess.Process:
    """Spawn infer --stdin and return the process handle."""
    cmd = _infer_cmd()
    log.info("Spawning: %s", " ".join(cmd))
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,   # captured so we can detect "Ready"
    )
    log.info("infer PID %d — loading model...", proc.pid)
    return proc


async def _wait_for_ready(proc: asyncio.subprocess.Process) -> None:
    """
    Tail stderr until we see the '[stdin] Ready' line, which the binary
    prints after Metal setup and model load are complete.
    """
    deadline = asyncio.get_event_loop().time() + READY_TIMEOUT
    async for line in proc.stderr:
        decoded = line.decode(errors="replace").rstrip()
        log.info("[infer] %s", decoded)
        if "[stdin] Ready" in decoded:
            return
        if asyncio.get_event_loop().time() > deadline:
            break
    raise RuntimeError("infer did not become ready within timeout")


async def _stderr_relay(proc: asyncio.subprocess.Process) -> None:
    """Background task: relay infer's stderr to our log so nothing is lost."""
    try:
        async for line in proc.stderr:
            log.debug("[infer] %s", line.decode(errors="replace").rstrip())
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Token streaming
# ─────────────────────────────────────────────────────────────────────────────

async def _send_request(prompt: str, max_tokens: int) -> None:
    """Write one JSON request to the infer process stdin."""
    # JSON-encode the prompt so newlines and quotes are escaped
    payload = json.dumps({"prompt": prompt, "max_tokens": max_tokens}) + "\n"
    _proc.stdin.write(payload.encode())
    await _proc.stdin.drain()


async def _read_tokens() -> AsyncIterator[str]:
    """
    Read stdout from the persistent infer process until the \\x00 end marker.
    Yields decoded text chunks as they arrive (one fflush per token from the binary).
    """
    raw_buf  = b""
    text_buf = ""

    while True:
        chunk = await _proc.stdout.read(READ_CHUNK)
        if not chunk:
            # Process died unexpectedly
            raise RuntimeError("infer process closed stdout unexpectedly")

        # Scan for null-byte end-of-response marker
        null_pos = chunk.find(b"\x00")
        if null_pos != -1:
            raw_buf += chunk[:null_pos]
            break
        raw_buf += chunk

        # Decode valid UTF-8 prefix; keep incomplete tail
        try:
            text = raw_buf.decode("utf-8")
            raw_buf = b""
        except UnicodeDecodeError:
            for cut in range(len(raw_buf) - 1, 0, -1):
                try:
                    text = raw_buf[:cut].decode("utf-8")
                    raw_buf = raw_buf[cut:]
                    break
                except UnicodeDecodeError:
                    continue
            else:
                continue

        text_buf += text
        # Strip EOS tokens that the binary may emit before stopping
        for eos in ("<|im_end|>", "<|endoftext|>"):
            text_buf = text_buf.replace(eos, "")
        if text_buf:
            yield text_buf
            text_buf = ""

    # Flush any remaining bytes after the null marker was found
    if raw_buf:
        try:
            text = raw_buf.decode("utf-8", errors="replace")
            for eos in ("<|im_end|>", "<|endoftext|>"):
                text = text.replace(eos, "")
            if text:
                yield text
        except Exception:
            pass


async def _stream_generation(
    prompt:     str,
    max_tokens: int,
    req_id:     str,
    created:    int,
    model:      str,
) -> AsyncIterator[str]:
    """
    Send request to infer, yield OpenAI SSE chunks as tokens arrive.
    Called inside the semaphore — only one active at a time.
    """
    await _send_request(prompt, max_tokens)

    # Role chunk (OpenAI convention)
    yield _chunk(req_id, created, model, {"role": "assistant", "content": ""}, None)

    try:
        async for text in _read_tokens():
            yield _chunk(req_id, created, model, {"content": text}, None)
    except Exception as exc:
        log.error("Stream error: %s", exc)
        yield f"data: {{\"error\": \"{exc}\"}}\n\n"

    yield _chunk(req_id, created, model, {}, "stop")
    yield "data: [DONE]\n\n"


def _chunk(req_id: str, created: int, model: str, delta: dict,
           finish_reason: Optional[str]) -> str:
    return "data: " + json.dumps({
        "id":      req_id,
        "object":  "chat.completion.chunk",
        "created": created,
        "model":   model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }) + "\n\n"


# ─────────────────────────────────────────────────────────────────────────────
# App lifecycle
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _proc, _sem, _ready

    _sem   = asyncio.Semaphore(1)
    _ready = asyncio.Event()

    _proc = await _start_process()

    # Wait for "[stdin] Ready" before accepting requests
    try:
        await _wait_for_ready(_proc)
        _ready.set()
    except Exception as exc:
        log.error("infer failed to start: %s", exc)
        _proc.kill()
        raise

    # Relay remaining stderr in the background
    asyncio.create_task(_stderr_relay(_proc))

    log.info("Flash-MoE wrapper ready on http://0.0.0.0:%d  (model hot)", WRAPPER_PORT)
    yield

    log.info("Shutting down infer (pid=%d)...", _proc.pid)
    try:
        _proc.stdin.close()
        await asyncio.wait_for(_proc.wait(), timeout=10)
    except Exception:
        _proc.kill()
        await _proc.wait()


app = FastAPI(title="Flash-MoE API", version="3.1.0", lifespan=lifespan)


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    alive = _proc is not None and _proc.returncode is None
    return {
        "status": "ok" if alive else "error",
        "model":  MODEL_ID,
        "pid":    _proc.pid if _proc else None,
        "busy":   _sem.locked() if _sem else False,
    }


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [{"id": MODEL_ID, "object": "model", "owned_by": "local", "created": 0}],
    }


@app.options("/v1/chat/completions")
async def cors_preflight():
    return Response(status_code=204, headers={
        "Access-Control-Allow-Origin":  "*",
        "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type, Authorization",
        "Access-Control-Max-Age":       "86400",
    })


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    if not _ready.is_set():
        raise HTTPException(503, "Model is still loading, please retry")
    if _proc is None or _proc.returncode is not None:
        raise HTTPException(503, "infer process is not running")

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(400, "Invalid JSON body")

    messages   = body.get("messages") or []
    model      = body.get("model", MODEL_ID)
    max_tokens = int(body.get("max_tokens") or body.get("max_completion_tokens") or 8192)
    do_stream  = body.get("stream", False)

    if not messages:
        raise HTTPException(400, "messages array is empty")
    if not any(m.get("role") == "user" for m in messages):
        raise HTTPException(400, "No user message in messages")

    prompt  = build_qwen_prompt(messages)
    req_id  = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())
    n_turns = sum(1 for m in messages if m.get("role") in ("user", "assistant"))

    log.info("POST /v1/chat/completions  id=%s  turns=%d  max_tokens=%d  stream=%s",
             req_id, n_turns, max_tokens, do_stream)

    if do_stream:
        async def generate():
            async with _sem:
                async for chunk in _stream_generation(
                    prompt, max_tokens, req_id, created, model
                ):
                    yield chunk

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control":               "no-cache",
                "Access-Control-Allow-Origin": "*",
                "X-Accel-Buffering":           "no",
            },
        )

    # Non-streaming: collect all tokens then return
    parts: list[str] = []
    finish_reason = "stop"

    async with _sem:
        async for raw in _stream_generation(
            prompt, max_tokens, req_id, created, model
        ):
            if not raw.startswith("data: "):
                continue
            payload = raw[6:].strip()
            if payload == "[DONE]":
                break
            try:
                obj = json.loads(payload)
                ch  = (obj.get("choices") or [{}])[0]
                tok = ch.get("delta", {}).get("content")
                if tok:
                    parts.append(tok)
                fr = ch.get("finish_reason")
                if fr:
                    finish_reason = fr
            except (json.JSONDecodeError, IndexError):
                pass

    return JSONResponse({
        "id":      req_id,
        "object":  "chat.completion",
        "created": created,
        "model":   model,
        "choices": [{
            "index":         0,
            "message":       {"role": "assistant", "content": "".join(parts)},
            "finish_reason": finish_reason,
        }],
        "usage": {
            "prompt_tokens":     len(prompt.split()),
            "completion_tokens": len(parts),
            "total_tokens":      len(prompt.split()) + len(parts),
        },
    })


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=WRAPPER_PORT,
        log_level="info",
        workers=1,
    )
