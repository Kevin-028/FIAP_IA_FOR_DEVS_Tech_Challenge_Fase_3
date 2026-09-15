"""Gera o SQLite anonimizado e o relatório de de-identificação."""
from __future__ import annotations

import argparse
import os

from .anonymize import anonymize_patient, residual_phi
from .catalog import PROTOCOLS, RAW_PATIENTS
from .db import connect, init_db
from .paths import ANON_REPORT, HOSPITAL_DATA, PROTOCOLS_DIR


def build(db_path: str | None = None) -> dict:
    os.makedirs(HOSPITAL_DATA, exist_ok=True)
    os.makedirs(PROTOCOLS_DIR, exist_ok=True)
    conn = connect(db_path)
    init_db(conn)
    conn.execute("DELETE FROM exams")
    conn.execute("DELETE FROM patients")
    conn.execute("DELETE FROM protocols")
    conn.commit()

    previews = []
    leftover = []
    for i, raw in enumerate(RAW_PATIENTS, start=1):
        clean, preview = anonymize_patient(raw, i)
        previews.append(preview)
        blob = " ".join([
            clean["display_name"], clean["notes"], clean["diagnosis"], clean["allergies"],
        ])
        leftover.extend(residual_phi(blob))
        conn.execute(
            """
            INSERT INTO patients
                (id, display_name, chart_hash, cpf_hash, birth_year, sex, allergies, diagnosis, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                clean["id"], clean["display_name"], clean["chart_hash"], clean["cpf_hash"],
                clean["birth_year"], clean["sex"], clean["allergies"], clean["diagnosis"],
                clean["notes"],
            ),
        )
        for exam in clean["exams"]:
            conn.execute(
                """
                INSERT INTO exams (patient_id, name, status, result, ordered_at, resulted_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    clean["id"], exam["name"], exam["status"], exam["result"],
                    exam["ordered_at"], exam["resulted_at"] or None,
                ),
            )

    for proto in PROTOCOLS:
        conn.execute(
            """
            INSERT INTO protocols (id, title, body, pmid, keywords)
            VALUES (?, ?, ?, ?, ?)
            """,
            (proto["id"], proto["title"], proto["body"], proto["pmid"], proto["keywords"]),
        )
        md = (
            f"# {proto['title']}\n\n"
            f"- protocol_id: {proto['id']}\n"
            f"- PMID: {proto['pmid']}\n\n"
            f"{proto['body']}\n"
        )
        with open(os.path.join(PROTOCOLS_DIR, f"{proto['id']}.md"), "w", encoding="utf-8") as f:
            f.write(md)

    conn.commit()
    n_patients = conn.execute("SELECT COUNT(*) AS n FROM patients").fetchone()["n"]
    n_pending = conn.execute(
        "SELECT COUNT(*) AS n FROM exams WHERE status = 'pending'"
    ).fetchone()["n"]
    conn.close()

    _write_report(previews, leftover)
    return {
        "patients": n_patients,
        "pending_exams": n_pending,
        "protocols": len(PROTOCOLS),
        "residual_phi": leftover,
        "report": ANON_REPORT,
    }


def _write_report(previews: list[dict], leftover: list[str]) -> None:
    lines = [
        "# Relatório de anonimização",
        "",
        "Dados sintéticos gerados com PHI proposital (nome, CPF, telefone, e-mail, endereço)",
        "e persistidos somente após de-identificação.",
        "",
        "| Técnica | Aplicação |",
        "| --- | --- |",
        "| Pseudônimo | nome → `PAC-0001` |",
        "| Hash com salt | CPF e número de prontuário |",
        "| Date shifting | offset fixo por paciente (intervalos clínicos preservados) |",
        "| Scrubbing | e-mail, telefone, CPF e endereço em texto livre |",
        "",
        f"PHI residual após scrubbing: **{len(leftover)}**.",
        "",
        "## Amostras antes / depois",
        "",
    ]
    for preview in previews[:3]:
        lines.extend([
            f"- Nome: `{preview['before_name']}` → `{preview['after_name']}`",
            f"- CPF: `{preview['before_cpf']}` → `{preview['after_cpf']}`",
            f"- Prontuário: `{preview['before_chart']}` → `{preview['after_chart']}`",
            f"- Offset de datas: {preview['date_offset_days']} dias",
            f"- Hits de PHI no texto: {preview['phi_hits']}",
            "",
        ])
    lines.append(
        "O conjunto bruto com PHI não é versionado. Só o SQLite anonimizado e este relatório vão ao repositório."
    )
    with open(ANON_REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed hospitalar anonimizado")
    parser.add_argument("--db", default=None)
    args = parser.parse_args()
    summary = build(args.db)
    print(summary)


if __name__ == "__main__":
    main()
