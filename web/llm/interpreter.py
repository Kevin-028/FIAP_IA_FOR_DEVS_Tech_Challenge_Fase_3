"""Serviço de interpretação: prompt, LLM e logging."""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone

from .client import get_client
from .config import INTERPRETATION_LOG, LOG_DIR
from .prompts import (
    CHAT_SYSTEM_PROMPT,
    COMPARISON_SYSTEM_EXTRA,
    FEW_SHOT_EXAMPLES,
    SYSTEM_PROMPT,
    build_case_context,
    build_user_prompt,
)

MAX_HISTORY_TURNS = 8


def interpret(payload: dict) -> dict:
    mode = payload.get("mode", "fase1")
    system = SYSTEM_PROMPT + FEW_SHOT_EXAMPLES
    if mode == "comparar":
        system += COMPARISON_SYSTEM_EXTRA

    user = build_user_prompt(payload)
    t0 = time.perf_counter()

    client = get_client()
    error = None
    text = ""
    try:
        text = client.complete(system, user)
    except Exception as e:
        error = str(e)
        from .client import FallbackClient
        fb = FallbackClient()
        text = fb.complete(system, user)
        client = fb

    latency_ms = round((time.perf_counter() - t0) * 1000, 2)
    sections = _parse_sections(text)

    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "provider": client.provider_name,
        "latency_ms": latency_ms,
        "diagnostico": _extract_diag(payload),
        "prob": _extract_prob(payload),
        "error": error,
        "text_len": len(text),
    }
    _log_interpretation(record)

    return {
        "text": text,
        "sections": sections,
        "provider": client.provider_name,
        "latency_ms": latency_ms,
        "mode": mode,
        "fallback_used": error is not None,
        "error": error,
    }


def chat(payload: dict) -> dict:
    question = (payload.get("question") or "").strip()
    if not question:
        raise ValueError("Pergunta vazia")

    mode = payload.get("mode", "fase2")
    grounding = "CONTEXTO DO CASO (use como base factual das respostas):\n" + build_case_context(payload)

    messages = [{"role": "user", "content": grounding}]
    for m in (payload.get("history") or [])[-MAX_HISTORY_TURNS:]:
        role = "assistant" if m.get("role") == "assistant" else "user"
        content = str(m.get("content", "")).strip()
        if content:
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": question})

    t0 = time.perf_counter()
    prefer_fallback = bool(payload.get("prefer_fallback"))
    error = None
    text = ""

    if prefer_fallback:
        from .client import FallbackClient
        client = FallbackClient()
        text = client.chat(CHAT_SYSTEM_PROMPT, messages)
    else:
        client = get_client()
        try:
            text = client.chat(CHAT_SYSTEM_PROMPT, messages)
        except Exception as e:
            error = str(e)
            from .client import FallbackClient
            fb = FallbackClient()
            text = fb.chat(CHAT_SYSTEM_PROMPT, messages)
            client = fb

    latency_ms = round((time.perf_counter() - t0) * 1000, 2)
    _log_interpretation({
        "ts": datetime.now(timezone.utc).isoformat(),
        "kind": "chat",
        "mode": mode,
        "provider": client.provider_name,
        "latency_ms": latency_ms,
        "question": question,
        "error": error,
        "text_len": len(text),
        "prefer_fallback": prefer_fallback,
    })

    return {
        "text": text,
        "provider": client.provider_name,
        "latency_ms": latency_ms,
        "mode": mode,
        "fallback_used": error is not None or prefer_fallback,
        "error": error,
    }


def _parse_sections(text: str) -> dict:
    sections = {}
    current = None
    buf = []
    for line in text.splitlines():
        if line.startswith("## "):
            if current:
                sections[current] = "\n".join(buf).strip()
            current = line[3:].strip()
            buf = []
        else:
            buf.append(line)
    if current:
        sections[current] = "\n".join(buf).strip()
    return sections


def _extract_diag(payload: dict) -> str:
    if payload.get("mode") == "comparar":
        r = payload.get("result", {})
        return f"{r.get('fase1', {}).get('diagnostico')}|{r.get('fase2', {}).get('diagnostico')}"
    return payload.get("result", {}).get("diagnostico", "")


def _extract_prob(payload: dict) -> float:
    if payload.get("mode") == "comparar":
        return payload.get("result", {}).get("fase2", {}).get("probabilidade", 0)
    return payload.get("result", {}).get("probabilidade", 0)


def _log_interpretation(record: dict):
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(INTERPRETATION_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def get_provider_info() -> dict:
    from .config import LLM_GGUF_PATH, LLM_PROVIDER, OLLAMA_BASE_URL, OLLAMA_MODEL

    if LLM_PROVIDER == "fallback":
        model_label = "fallback/regras-clinicas"
        using_real = False
    elif LLM_PROVIDER == "ollama":
        model_label = f"ollama/{OLLAMA_MODEL}"
        using_real = True
    else:
        model_label = "huggingface/local"
        using_real = True

    engine_status = {"loaded": False, "path": LLM_GGUF_PATH}
    try:
        from .engine import engine

        engine_status = engine.status()
    except Exception:
        pass

    return {
        "provider": LLM_PROVIDER,
        "provider_active": LLM_PROVIDER,
        "model": model_label,
        "ollama_model": OLLAMA_MODEL,
        "ollama_url": OLLAMA_BASE_URL,
        "using_real_llm": using_real,
        "engine": engine_status,
    }
