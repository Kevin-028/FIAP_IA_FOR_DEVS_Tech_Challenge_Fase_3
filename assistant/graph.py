"""Fluxo clínico em LangGraph."""
from __future__ import annotations

from typing import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from assistant.guardrails import (
    DISCLAIMER,
    NO_PRESCRIPTION,
    OUT_OF_SCOPE,
    has_source,
    postfilter,
    prefilter,
)
from assistant.tools import (
    emit_alert_tool,
    get_patient_tool,
    get_pending_exams_tool,
    search_protocols_tool,
)


class ClinicalState(TypedDict, total=False):
    thread_id: str
    patient_id: str
    question: str
    history: list
    record: dict
    pending_exams: list
    retrieved_docs: list
    llm_draft: str
    sources: list
    alerts: list
    safety_flags: list
    needs_human: bool
    final_answer: str
    out_of_scope: bool
    prescription_request: bool


def build_graph(checkpointer=None, complete=None):
    """`complete(question, docs, record) -> str` é obrigatório."""
    if complete is None:
        raise ValueError("complete (LLM) é obrigatório.")
    llm = complete

    def classificar(state: ClinicalState) -> dict:
        flags = prefilter(state.get("question", ""))
        return {
            "out_of_scope": flags["out_of_scope"],
            "prescription_request": flags["prescription_request"],
            "safety_flags": [],
            "alerts": [],
            "sources": [],
        }

    def carregar(state: ClinicalState) -> dict:
        record = get_patient_tool.invoke({"patient_id": state["patient_id"]})
        pending = get_pending_exams_tool.invoke({"patient_id": state["patient_id"]})
        return {"record": record, "pending_exams": pending}

    def recuperar(state: ClinicalState) -> dict:
        record = state.get("record") or {}
        context = " ".join(
            part for part in (record.get("diagnosis"), record.get("allergies")) if part
        )
        docs = search_protocols_tool.invoke({
            "query": state.get("question", ""),
            "context": context,
        })
        docs = docs[:1]
        sources = [
            {"protocol_id": d["protocol_id"], "pmid": d["pmid"], "title": d["title"]}
            for d in docs
        ]
        return {"retrieved_docs": docs, "sources": sources}

    def gerar(state: ClinicalState) -> dict:
        record = dict(state.get("record") or {})
        record["pending"] = state.get("pending_exams") or []
        record["history"] = state.get("history") or []
        draft = llm(state.get("question", ""), state.get("retrieved_docs") or [], record)
        return {"llm_draft": draft}

    def validar(state: ClinicalState) -> dict:
        flags = list(state.get("safety_flags") or [])
        draft = state.get("llm_draft") or ""
        if state.get("prescription_request") or postfilter(draft):
            flags.append("prescription_blocked")
            draft = NO_PRESCRIPTION
        if not state.get("out_of_scope") and not has_source(state.get("sources") or []):
            flags.append("missing_source")
            draft = (
                "No PMID or protocol_id was retrieved. The assistant will not publish a clinical opinion. "
                "Always consult the attending physician."
            )
        return {"needs_human": False, "safety_flags": flags, "llm_draft": draft}

    def alertar(state: ClinicalState) -> dict:
        reasons = []
        if state.get("pending_exams"):
            reasons.append("pending_exams")
        if "prescription_blocked" in (state.get("safety_flags") or []):
            reasons.append("prescription_refused")
        created = []
        for reason in reasons:
            status = emit_alert_tool.invoke({
                "patient_id": state.get("patient_id", ""),
                "thread_id": state.get("thread_id", ""),
                "severity": "high" if reason == "prescription_refused" else "medium",
                "reason": reason,
            })
            created.append({"reason": reason, "status": status})
        return {"alerts": created}

    def responder(state: ClinicalState) -> dict:
        if state.get("out_of_scope"):
            return {"final_answer": OUT_OF_SCOPE, "needs_human": False}
        text = (state.get("llm_draft") or "").strip()
        if "consult" not in text.lower() and "physician" not in text.lower():
            text = f"{text}\n\n{DISCLAIMER}" if text else DISCLAIMER
        return {"final_answer": text, "needs_human": False}

    def _after_classificar(state: ClinicalState) -> str:
        return "responder" if state.get("out_of_scope") else "carregar"

    builder = StateGraph(ClinicalState)
    builder.add_node("classificar", classificar)
    builder.add_node("carregar", carregar)
    builder.add_node("recuperar", recuperar)
    builder.add_node("gerar", gerar)
    builder.add_node("validar", validar)
    builder.add_node("alertar", alertar)
    builder.add_node("responder", responder)

    builder.add_edge(START, "classificar")
    builder.add_conditional_edges("classificar", _after_classificar, {
        "carregar": "carregar",
        "responder": "responder",
    })
    builder.add_edge("carregar", "recuperar")
    builder.add_edge("recuperar", "gerar")
    builder.add_edge("gerar", "validar")
    builder.add_edge("validar", "alertar")
    builder.add_edge("alertar", "responder")
    builder.add_edge("responder", END)

    return builder.compile(checkpointer=checkpointer or MemorySaver())


def thread_config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def is_interrupted(result: dict) -> bool:
    return bool(result.get("__interrupt__"))
