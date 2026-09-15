"""Ponto de entrada do assistente."""
from __future__ import annotations

import os
import sqlite3
import sys
import uuid

from langgraph.types import Command

_WEB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web")
if _WEB not in sys.path:
    sys.path.insert(0, _WEB)

from assistant.audit import log_turn
from assistant.graph import build_graph, is_interrupted, thread_config
from assistant import tools as tools_mod
from hospital.db import connect, init_db, upsert_hitl
from hospital.paths import CHECKPOINTS_PATH, DB_PATH


_graph = None
_saver = None

LLM_REQUIRED = (
    "LLM local não está carregada. Abra /modelo, clique em Carregar e tente de novo. "
    "Este assistente não responde sem o modelo fine-tuned."
)


def configure(db_path: str | None = None) -> str:
    path = db_path or DB_PATH
    tools_mod.DB_PATH = path
    init_db(connect(path))
    return path


def _checkpointer():
    global _saver
    if _saver is not None:
        return _saver
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver

        conn = sqlite3.connect(CHECKPOINTS_PATH, check_same_thread=False)
        _saver = SqliteSaver(conn)
    except Exception:
        from langgraph.checkpoint.memory import MemorySaver

        _saver = MemorySaver()
    return _saver


def get_graph(complete=None, checkpointer=None):
    global _graph
    if checkpointer is not None or complete is not None:
        return build_graph(checkpointer=checkpointer, complete=complete)
    if _graph is None:
        _graph = build_graph(checkpointer=_checkpointer(), complete=_llm)
    return _graph


def reset_graph() -> None:
    global _graph, _saver
    _graph = None
    _saver = None


def _history_block(history: list | None) -> str:
    lines = []
    for turn in (history or [])[-4:]:
        if turn.get("role") != "user":
            continue
        text = (turn.get("content") or "").strip()
        if text:
            lines.append(f"- {text}")
    if not lines:
        return ""
    return "Earlier questions:\n" + "\n".join(lines) + "\n\n"


def _llm(question: str, docs: list, record: dict) -> str:
    from llm.engine import engine

    if not engine.status()["loaded"]:
        raise RuntimeError(LLM_REQUIRED)

    top = docs[0] if docs else None
    evidence_block = "No protocol retrieved."
    if top:
        raw = (top.get("snippet") or "").replace("REQUIRES PHYSICIAN VALIDATION.", "").strip()
        evidence, _, conduct = raw.partition("Suggested conduct")
        evidence = evidence.strip() or raw
        conduct = ("Suggested conduct" + conduct).strip() if conduct else ""
        if len(evidence) > 280:
            evidence = evidence[:277].rsplit(" ", 1)[0] + "…"
        if len(conduct) > 200:
            conduct = conduct[:197].rsplit(" ", 1)[0] + "…"
        evidence_block = (
            f"{top['protocol_id']} (PMID {top['pmid']})\n"
            f"Evidence: {evidence}\n"
            f"Conduct: {conduct or 'none'}"
        )
    pending_names = [p["name"] for p in (record.get("pending") or []) if p.get("name")]
    pending = ", ".join(pending_names) if pending_names else "none"
    user = (
        "CHART (authoritative — never invent facts):\n"
        f"- Patient {record.get('id')}: {record.get('diagnosis')}\n"
        f"- Allergies: {record.get('allergies') or 'none'}\n"
        f"- Pending exams: {pending}\n\n"
        f"PROTOCOL:\n{evidence_block}\n\n"
        f"{_history_block(record.get('history'))}"
        f"Clinician question: {question}"
    )
    text = engine.complete(
        "You are a hospital clinical assistant. Reply in English, at most 3 short sentences.\n"
        "1) Answer the clinician question first (Yes/No/Mixed when it is a yes-no question).\n"
        "2) If they ask about pending labs/exams, answer ONLY from 'Pending exams' in CHART "
        "(say 'none' when it is none). Do not invent labs.\n"
        "3) If they ask what the protocol says for THIS patient, use Conduct + the chart diagnosis.\n"
        "4) Cite one PMID. Never prescribe a drug, dose or route. Do not paste the full protocol.",
        user,
        max_new_tokens=140,
    )
    text = (text or "").strip()
    if not text:
        raise RuntimeError("A LLM retornou resposta vazia. Tente de novo ou recarregue o modelo em /modelo.")
    return text


def ask(
    patient_id: str,
    question: str,
    thread_id: str | None = None,
    history: list | None = None,
    graph=None,
) -> dict:
    configure()
    from llm.engine import engine

    if not engine.status()["loaded"]:
        raise RuntimeError(LLM_REQUIRED)

    if graph is None:
        reset_graph()
    thread_id = thread_id or uuid.uuid4().hex[:12]
    compiled = graph or get_graph()
    state = {
        "thread_id": thread_id,
        "patient_id": patient_id,
        "question": question,
        "history": history or [],
    }
    result = compiled.invoke(state, thread_config(thread_id))
    interrupted = is_interrupted(result)
    if interrupted:
        upsert_hitl(
            connect(),
            thread_id=thread_id,
            patient_id=patient_id,
            question=question,
            draft=result.get("llm_draft") or "",
            status="waiting",
        )
    payload = _public(result, thread_id, interrupted)
    payload["via"] = "llm"
    log_turn({"kind": "ask", **payload})
    return payload


def resume(thread_id: str, decision: str, graph=None) -> dict:
    configure()
    compiled = graph or get_graph()
    result = compiled.invoke(Command(resume=decision), thread_config(thread_id))
    status = "approved" if decision == "approved" else "rejected"
    conn = connect()
    row = conn.execute("SELECT * FROM hitl WHERE thread_id = ?", (thread_id,)).fetchone()
    upsert_hitl(
        conn,
        thread_id=thread_id,
        patient_id=(row["patient_id"] if row else ""),
        question=(row["question"] if row else ""),
        draft=(row["draft"] if row else ""),
        status=status,
    )
    payload = _public(result, thread_id, False)
    log_turn({"kind": "resume", "decision": decision, **payload})
    return payload


def _public(result: dict, thread_id: str, interrupted: bool) -> dict:
    return {
        "thread_id": thread_id,
        "interrupted": interrupted,
        "needs_human": bool(result.get("needs_human")) and interrupted,
        "final_answer": result.get("final_answer") or result.get("llm_draft") or "",
        "sources": result.get("sources") or [],
        "pending_exams": result.get("pending_exams") or [],
        "alerts": result.get("alerts") or [],
        "safety_flags": result.get("safety_flags") or [],
        "patient_id": result.get("patient_id"),
    }
