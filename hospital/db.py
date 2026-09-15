"""SQLite do hospital."""
from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timezone

from .paths import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS patients (
    id TEXT PRIMARY KEY,
    display_name TEXT NOT NULL,
    chart_hash TEXT NOT NULL,
    cpf_hash TEXT NOT NULL,
    birth_year INTEGER,
    sex TEXT,
    allergies TEXT,
    diagnosis TEXT,
    notes TEXT
);
CREATE TABLE IF NOT EXISTS exams (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id TEXT NOT NULL,
    name TEXT NOT NULL,
    status TEXT NOT NULL,
    result TEXT,
    ordered_at TEXT,
    resulted_at TEXT,
    FOREIGN KEY (patient_id) REFERENCES patients(id)
);
CREATE TABLE IF NOT EXISTS protocols (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    body TEXT NOT NULL,
    pmid TEXT NOT NULL,
    keywords TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id TEXT,
    thread_id TEXT,
    severity TEXT NOT NULL,
    reason TEXT NOT NULL,
    source TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'open',
    created_at TEXT NOT NULL,
    UNIQUE(thread_id, reason)
);
CREATE TABLE IF NOT EXISTS hitl (
    thread_id TEXT PRIMARY KEY,
    patient_id TEXT,
    question TEXT,
    draft TEXT,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    resolved_at TEXT
);
"""


def connect(db_path: str | None = None) -> sqlite3.Connection:
    path = db_path or DB_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    conn.commit()


def utcnow() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def get_patient(conn: sqlite3.Connection, patient_id: str) -> dict | None:
    row = conn.execute("SELECT * FROM patients WHERE id = ?", (patient_id,)).fetchone()
    return dict(row) if row else None


def list_patients(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute("SELECT * FROM patients ORDER BY id").fetchall()
    return [dict(r) for r in rows]


def pending_exams(conn: sqlite3.Connection, patient_id: str) -> list[dict]:
    rows = conn.execute(
        "SELECT * FROM exams WHERE patient_id = ? AND status = 'pending' ORDER BY id",
        (patient_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def list_exams(conn: sqlite3.Connection, patient_id: str) -> list[dict]:
    rows = conn.execute(
        "SELECT * FROM exams WHERE patient_id = ? ORDER BY id",
        (patient_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def record_exam_result(
    conn: sqlite3.Connection, exam_id: int, result: str, resulted_at: str | None = None
) -> dict | None:
    row = conn.execute("SELECT * FROM exams WHERE id = ?", (exam_id,)).fetchone()
    if not row:
        return None
    conn.execute(
        """
        UPDATE exams
           SET status = 'completed', result = ?, resulted_at = ?
         WHERE id = ?
        """,
        (result, resulted_at or utcnow()[:10], exam_id),
    )
    conn.commit()
    updated = conn.execute("SELECT * FROM exams WHERE id = ?", (exam_id,)).fetchone()
    return dict(updated)


def list_protocols(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute("SELECT * FROM protocols ORDER BY id").fetchall()
    return [dict(r) for r in rows]


def emit_alert(
    conn: sqlite3.Connection,
    *,
    patient_id: str,
    thread_id: str,
    severity: str,
    reason: str,
    source: str,
) -> bool:
    cur = conn.execute(
        """
        INSERT OR IGNORE INTO alerts
            (patient_id, thread_id, severity, reason, source, status, created_at)
        VALUES (?, ?, ?, ?, ?, 'open', ?)
        """,
        (patient_id, thread_id, severity, reason, source, utcnow()),
    )
    conn.commit()
    return cur.rowcount == 1


def list_alerts(conn: sqlite3.Connection, status: str | None = None) -> list[dict]:
    if status:
        rows = conn.execute(
            "SELECT * FROM alerts WHERE status = ? ORDER BY id DESC", (status,)
        ).fetchall()
    else:
        rows = conn.execute("SELECT * FROM alerts ORDER BY id DESC").fetchall()
    return [dict(r) for r in rows]


def open_alert_count(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT COUNT(*) AS n FROM alerts WHERE status = 'open'").fetchone()
    return int(row["n"])


def upsert_hitl(
    conn: sqlite3.Connection,
    *,
    thread_id: str,
    patient_id: str,
    question: str,
    draft: str,
    status: str,
) -> None:
    existing = conn.execute(
        "SELECT thread_id FROM hitl WHERE thread_id = ?", (thread_id,)
    ).fetchone()
    if existing:
        conn.execute(
            """
            UPDATE hitl
               SET status = ?,
                   resolved_at = CASE WHEN ? IN ('approved', 'rejected') THEN ? ELSE resolved_at END
             WHERE thread_id = ?
            """,
            (status, status, utcnow(), thread_id),
        )
    else:
        conn.execute(
            """
            INSERT INTO hitl (thread_id, patient_id, question, draft, status, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (thread_id, patient_id, question, draft, status, utcnow()),
        )
    conn.commit()


def waiting_hitl_count(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT COUNT(*) AS n FROM hitl WHERE status = 'waiting'").fetchone()
    return int(row["n"])


def list_hitl(conn: sqlite3.Connection, status: str = "waiting") -> list[dict]:
    rows = conn.execute(
        "SELECT * FROM hitl WHERE status = ? ORDER BY created_at", (status,)
    ).fetchall()
    return [dict(r) for r in rows]
