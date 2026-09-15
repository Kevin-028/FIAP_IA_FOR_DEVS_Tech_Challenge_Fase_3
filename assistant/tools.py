"""Tools LangChain usadas pelos nós do grafo."""
from __future__ import annotations

import sqlite3

from langchain_core.tools import tool

from hospital.db import (
    emit_alert,
    get_patient,
    pending_exams,
    record_exam_result,
)
from rag.retrieve import search_protocols

DB_PATH: str | None = None


def _conn() -> sqlite3.Connection:
    if not DB_PATH:
        raise RuntimeError("DB_PATH não configurado")
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


@tool
def get_patient_tool(patient_id: str) -> dict:
    """Load the current anonymized chart. Always reads the database."""
    conn = _conn()
    try:
        return get_patient(conn, patient_id) or {}
    finally:
        conn.close()


@tool
def get_pending_exams_tool(patient_id: str) -> list:
    """List exams still pending for the patient."""
    conn = _conn()
    try:
        return pending_exams(conn, patient_id)
    finally:
        conn.close()


@tool
def search_protocols_tool(query: str, context: str = "") -> list:
    """Retrieve hospital protocols and PMIDs relevant to the question."""
    conn = _conn()
    try:
        return search_protocols(conn, query, context=context, k=3)
    finally:
        conn.close()


@tool
def emit_alert_tool(patient_id: str, thread_id: str, severity: str, reason: str) -> str:
    """Persist a team alert. Idempotent per thread and reason."""
    conn = _conn()
    try:
        created = emit_alert(
            conn,
            patient_id=patient_id,
            thread_id=thread_id,
            severity=severity,
            reason=reason,
            source="langgraph",
        )
        return "created" if created else "exists"
    finally:
        conn.close()


@tool
def registrar_resultado_exame(exam_id: int, result: str) -> dict:
    """Write a new exam result so the next question sees updated context."""
    conn = _conn()
    try:
        return record_exam_result(conn, int(exam_id), result) or {}
    finally:
        conn.close()
