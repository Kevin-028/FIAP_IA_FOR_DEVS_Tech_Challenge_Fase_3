"""De-identificação de prontuários sintéticos."""
from __future__ import annotations

import hashlib
import re
from datetime import datetime, timedelta

SALT = "fiap-fase3-hospital-salt"

_EMAIL = re.compile(r"[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}", re.I)
_PHONE = re.compile(r"(?:\+?55\s?)?(?:\(?\d{2}\)?\s?)?(?:9\s?)?\d{4,5}[-.\s]?\d{4}")
_CPF = re.compile(r"\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b")
_ADDRESS = re.compile(
    r"\b(?:rua|av\.?|avenida|travessa|alameda)\s+[^,\n]{3,80}",
    re.I,
)


def hash_id(value: str) -> str:
    digest = hashlib.sha256(f"{SALT}:{value}".encode("utf-8")).hexdigest()
    return digest[:12]


def scrub_text(text: str) -> tuple[str, dict[str, int]]:
    counts = {"email": 0, "phone": 0, "cpf": 0, "address": 0}
    if not text:
        return text, counts

    def _sub(pattern, label, replacement):
        nonlocal text
        found = pattern.findall(text)
        counts[label] += len(found)
        text = pattern.sub(replacement, text)

    _sub(_EMAIL, "email", "[EMAIL]")
    _sub(_CPF, "cpf", "[CPF]")
    _sub(_PHONE, "phone", "[TELEFONE]")
    _sub(_ADDRESS, "address", "[ENDERECO]")
    return text, counts


def shift_date(iso_date: str, offset_days: int) -> str:
    dt = datetime.strptime(iso_date[:10], "%Y-%m-%d")
    return (dt + timedelta(days=offset_days)).date().isoformat()


def anonymize_patient(raw: dict, index: int) -> tuple[dict, dict]:
    offset = 17 + (index * 11) % 40
    notes, note_counts = scrub_text(raw.get("notes", ""))
    diagnosis, diag_counts = scrub_text(raw.get("diagnosis", ""))
    counts = {k: note_counts.get(k, 0) + diag_counts.get(k, 0) for k in note_counts}

    exams = []
    for exam in raw.get("exams", []):
        ordered = shift_date(exam["ordered_at"], offset) if exam.get("ordered_at") else None
        resulted = shift_date(exam["resulted_at"], offset) if exam.get("resulted_at") else None
        result_text, result_counts = scrub_text(exam.get("result") or "")
        for k, v in result_counts.items():
            counts[k] = counts.get(k, 0) + v
        exams.append({
            "name": exam["name"],
            "status": exam["status"],
            "result": result_text,
            "ordered_at": ordered,
            "resulted_at": resulted,
        })

    clean = {
        "id": f"PAC-{index:04d}",
        "display_name": f"PAC-{index:04d}",
        "chart_hash": hash_id(raw["chart_number"]),
        "cpf_hash": hash_id(raw["cpf"]),
        "birth_year": int(raw["birth_date"][:4]) - (offset // 30),
        "sex": raw.get("sex", ""),
        "allergies": raw.get("allergies", ""),
        "diagnosis": diagnosis,
        "notes": notes,
        "exams": exams,
    }
    preview = {
        "before_name": raw["name"],
        "after_name": clean["display_name"],
        "before_cpf": raw["cpf"],
        "after_cpf": clean["cpf_hash"],
        "before_chart": raw["chart_number"],
        "after_chart": clean["chart_hash"],
        "date_offset_days": offset,
        "phi_hits": counts,
    }
    return clean, preview


def residual_phi(text: str) -> list[str]:
    hits = []
    if _EMAIL.search(text or ""):
        hits.append("email")
    if _CPF.search(text or ""):
        hits.append("cpf")
    if _PHONE.search(text or ""):
        hits.append("phone")
    if _ADDRESS.search(text or ""):
        hits.append("address")
    return hits
