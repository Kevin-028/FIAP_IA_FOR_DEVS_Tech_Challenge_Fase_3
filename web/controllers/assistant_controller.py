"""Rotas da Fase 3."""
from __future__ import annotations

import os
import sys

from flask import Blueprint, jsonify, redirect, render_template, request, url_for

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from assistant.service import ask, configure, resume
from hospital.db import (
    connect,
    list_alerts,
    list_exams,
    list_hitl,
    list_patients,
    open_alert_count,
    pending_exams,
    record_exam_result,
    waiting_hitl_count,
)

assistant_bp = Blueprint("fase3", __name__)

LABELS = ("yes", "no", "maybe")


def _db():
    configure()
    return connect()


def _pct(value) -> str | None:
    if value is None:
        return None
    return f"{float(value) * 100:.1f}%"


def _confusion_table(matrix: dict) -> list[dict]:
    rows = []
    for gold in LABELS:
        row = {"gold": gold}
        for pred in LABELS:
            row[pred] = int((matrix.get(gold) or {}).get(pred) or 0)
        rows.append(row)
    return rows


def desk_context() -> dict:
    from hospital.artifacts import snapshot
    from llm.engine import engine

    sft = snapshot()
    conn = _db()
    try:
        patients = list_patients(conn)
        waiting = waiting_hitl_count(conn)
        alerts = open_alert_count(conn)
        pending = 0
        for patient in patients:
            pending += len(pending_exams(conn, patient["id"]))
    finally:
        conn.close()
    return {
        "sft": sft,
        "engine": engine.status(),
        "n_patients": len(patients),
        "n_waiting": waiting,
        "n_alerts": alerts,
        "n_pending_exams": pending,
        "base_acc": _pct(sft.get("base_accuracy")),
        "ft_acc": _pct(sft.get("ft_accuracy")),
        "base_f1": _pct(sft.get("base_macro_f1")),
        "ft_f1": _pct(sft.get("ft_macro_f1")),
    }


@assistant_bp.route("/pacientes")
def pacientes():
    conn = _db()
    rows = list_patients(conn)
    conn.close()
    return render_template("fase3/pacientes.html", patients=rows)


@assistant_bp.route("/pacientes/<patient_id>")
def prontuario(patient_id):
    conn = _db()
    from hospital.db import get_patient

    patient = get_patient(conn, patient_id)
    exams = list_exams(conn, patient_id) if patient else []
    pending = [e for e in exams if e.get("status") == "pending"]
    conn.close()
    if not patient:
        return redirect(url_for("fase3.pacientes"))
    return render_template(
        "fase3/prontuario.html",
        patient=patient,
        exams=exams,
        pending=pending,
    )


@assistant_bp.route("/pacientes/<patient_id>/exames/<int:exam_id>/resultado", methods=["POST"])
def resultado_exame(patient_id, exam_id):
    result = (request.form.get("result") or "").strip()
    if result:
        conn = _db()
        try:
            record_exam_result(conn, exam_id, result)
        finally:
            conn.close()
    return redirect(url_for("fase3.prontuario", patient_id=patient_id))


@assistant_bp.route("/assistente")
def assistente():
    conn = _db()
    patients = list_patients(conn)
    selected = request.args.get("patient_id") or (patients[0]["id"] if patients else "")
    pending = pending_exams(conn, selected) if selected else []
    conn.close()
    from llm.engine import engine
    from hospital.artifacts import snapshot

    sft = snapshot()
    resp = render_template(
        "fase3/assistente.html",
        patients=patients,
        selected=selected,
        pending=pending,
        engine=engine.status(),
        adapter_ok=sft["adapter_ok"],
    )
    from flask import make_response

    out = make_response(resp)
    out.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
    return out


@assistant_bp.route("/modelo")
def modelo():
    ctx = desk_context()
    ctx["base_matrix"] = _confusion_table(ctx["sft"].get("base_confusion") or {})
    ctx["ft_matrix"] = _confusion_table(ctx["sft"].get("ft_confusion") or {})
    return render_template("fase3/modelo.html", **ctx)


@assistant_bp.route("/auditoria")
def auditoria():
    from hospital.artifacts import audit_turns

    return render_template("fase3/auditoria.html", turns=audit_turns())


def _history(data) -> list[dict]:
    raw = data.get("history") or []
    if not isinstance(raw, list):
        return []
    turns = []
    for item in raw[-8:]:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        content = (item.get("content") or "").strip()
        if role in ("user", "assistant") and content:
            turns.append({"role": role, "content": content[:2000]})
    return turns


@assistant_bp.route("/api/assistente/perguntar", methods=["POST"])
def perguntar():
    data = request.get_json(silent=True) or request.form
    patient_id = (data.get("patient_id") or "").strip()
    question = (data.get("question") or "").strip()
    if not patient_id or not question:
        return jsonify({"error": "patient_id e question são obrigatórios"}), 400
    try:
        payload = ask(patient_id, question, history=_history(data))
    except RuntimeError as exc:
        msg = str(exc)
        code = 503 if "não está carregada" in msg or "não carregada" in msg else 500
        return jsonify({"error": msg, "via": None}), code
    except Exception as exc:
        return jsonify({"error": str(exc), "via": None}), 500
    return jsonify(payload)


@assistant_bp.route("/validacao")
def validacao():
    conn = _db()
    rows = list_hitl(conn, "waiting")
    conn.close()
    return render_template("fase3/validacao.html", items=rows)


@assistant_bp.route("/api/validacao/<thread_id>", methods=["POST"])
def decidir(thread_id):
    data = request.get_json(silent=True) or request.form
    decision = (data.get("decision") or "").strip().lower()
    if decision not in ("approved", "rejected"):
        return jsonify({"error": "decision deve ser approved ou rejected"}), 400
    try:
        payload = resume(thread_id, decision)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
    if request.is_json:
        return jsonify(payload)
    return redirect(url_for("fase3.validacao"))


@assistant_bp.route("/alertas")
def alertas():
    conn = _db()
    rows = list_alerts(conn)
    conn.close()
    return render_template("fase3/alertas.html", alerts=rows)


