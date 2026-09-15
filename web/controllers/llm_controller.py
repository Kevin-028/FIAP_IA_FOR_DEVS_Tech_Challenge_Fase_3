"""
Controller LLM — interpretação e avaliação.
"""
import os
import time

from flask import Blueprint, jsonify, render_template, request

from llm.evaluation import run_evaluation_suite
from llm.feedback import FEEDBACK_TAGS, load_feedback, record_feedback, summarize_feedback
from llm.interpreter import chat, get_provider_info, interpret
from monitoring.logger import log_event
from monitoring.metrics import metrics_store

llm_bp = Blueprint("llm", __name__)


@llm_bp.route("/api/llm/interpret", methods=["POST"])
def api_interpret():
    payload = request.get_json(silent=True) or {}
    if not payload.get("result"):
        return jsonify({"error": "Campo 'result' obrigatório"}), 400

    t0 = time.perf_counter()
    try:
        out = interpret(payload)
    except Exception as e:
        log_event("error", "llm_interpret_failed", error=str(e))
        return jsonify({"error": str(e)}), 500

    latency = (time.perf_counter() - t0) * 1000
    metrics_store.record_request("/api/llm/interpret", "POST", 200, latency)
    log_event(
        "info", "llm_interpret",
        mode=payload.get("mode"),
        provider=out.get("provider"),
        latency_ms=out.get("latency_ms"),
    )
    return jsonify(out)


@llm_bp.route("/api/llm/chat", methods=["POST"])
def api_chat():
    payload = request.get_json(silent=True) or {}
    if not payload.get("result"):
        return jsonify({"error": "Campo 'result' obrigatório"}), 400
    if not (payload.get("question") or "").strip():
        return jsonify({"error": "Campo 'question' obrigatório"}), 400

    t0 = time.perf_counter()
    try:
        out = chat(payload)
    except Exception as e:
        log_event("error", "llm_chat_failed", error=str(e))
        return jsonify({"error": str(e)}), 500

    latency = (time.perf_counter() - t0) * 1000
    metrics_store.record_request("/api/llm/chat", "POST", 200, latency)
    log_event("info", "llm_chat", mode=payload.get("mode"),
              provider=out.get("provider"), latency_ms=out.get("latency_ms"))
    return jsonify(out)


@llm_bp.route("/api/llm/feedback", methods=["POST"])
def api_feedback():
    payload = request.get_json(silent=True) or {}
    try:
        record = record_feedback(payload)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        log_event("error", "llm_feedback_failed", error=str(e))
        return jsonify({"error": str(e)}), 500

    log_event("info", "llm_feedback", rating=record["rating"],
              tags=record["tags"], mode=record["mode"])
    return jsonify({"ok": True})


@llm_bp.route("/api/llm/status")
def api_status():
    return jsonify(get_provider_info())


@llm_bp.route("/llm/avaliacao")
def avaliacao_page():
    examples_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "examples_data.json",
    )
    try:
        rows = run_evaluation_suite(examples_path)
    except Exception as e:
        rows = []
        log_event("error", "llm_eval_failed", error=str(e))

    avg = 0.0
    if rows:
        avg = sum(r["score"]["overall"] for r in rows) / len(rows)

    return render_template(
        "llm/avaliacao.html",
        results=rows,
        avg_score=round(avg, 3),
        provider_info=get_provider_info(),
    )


@llm_bp.route("/llm/auditoria")
def auditoria_page():
    records = list(reversed(load_feedback()))
    return render_template(
        "llm/auditoria.html",
        summary=summarize_feedback(),
        records=records,
        tags=FEEDBACK_TAGS,
        provider_info=get_provider_info(),
    )
