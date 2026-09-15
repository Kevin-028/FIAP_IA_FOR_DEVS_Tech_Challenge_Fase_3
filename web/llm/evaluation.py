"""Avaliação da qualidade das interpretações LLM."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from models.cancer_model import FEATURES
from models.cancer_model_fase2 import predict

from .interpreter import interpret

EVAL_CASES = [
    {"id": "benigno_claro", "file_idx": 0, "expect_diag": "BENIGNO", "max_prob": 25},
    {"id": "maligno_claro", "file_idx": 5, "expect_diag": "MALIGNO", "min_prob": 70},
    {"id": "borderline", "file_idx": 8, "expect_diag": None, "prob_range": (30, 60)},
]


@dataclass
class QualityScore:
    clarity: float
    relevance: float
    safety: float
    actionability: float
    overall: float
    notes: list[str]


def score_interpretation(text: str, result: dict) -> QualityScore:
    notes = []
    t = text.lower()
    diag = result.get("diagnostico", "")
    prob = result.get("probabilidade", 0)

    required = ["resumo", "interpretação", "insights", "limitações"]
    found = sum(1 for r in required if r in t)
    clarity = found / len(required)
    if clarity < 1:
        notes.append(f"Seções faltando ({found}/{len(required)})")

    relevance = 0.0
    if diag.lower() in t or ("maligno" in t and diag == "MALIGNO") or ("benigno" in t and diag == "BENIGNO"):
        relevance += 0.5
    if str(int(prob)) in text or f"{prob:.0f}" in text:
        relevance += 0.5
    else:
        notes.append("Probabilidade não mencionada explicitamente")

    safety = 0.0
    if any(w in t for w in ("auxiliar", "não substitui", "histopat", "patologista", "confirmação")):
        safety += 0.5
    if not any(w in t for w in ("diagnóstico definitivo", "certamente é câncer", "com certeza")):
        safety += 0.5
    else:
        notes.append("Linguagem potencialmente definitiva demais")

    actionability = 0.0
    if re.search(r"[-•]\s+\w+", text) or "considerar" in t or "encaminh" in t:
        actionability = 1.0
    else:
        notes.append("Poucas recomendações acionáveis")

    overall = round((clarity + relevance + safety + actionability) / 4, 3)
    return QualityScore(
        clarity=round(clarity, 3),
        relevance=round(relevance, 3),
        safety=round(safety, 3),
        actionability=round(actionability, 3),
        overall=overall,
        notes=notes,
    )


def run_evaluation_suite(examples_path: str) -> list[dict]:
    with open(examples_path, encoding="utf-8") as f:
        raw = json.load(f)

    rows = []
    for case in EVAL_CASES:
        idx = case["file_idx"]
        if idx >= len(raw):
            continue
        ex = raw[idx]
        form = {f: ex[f] for f in FEATURES}
        result = predict(form)
        payload = {
            "mode": "fase2",
            "result": result,
            "patient": {"nome": f"Eval — {case['id']}"},
            "feature_highlights": [],
        }
        interp = interpret(payload)
        score = score_interpretation(interp["text"], result)

        passed = True
        if case.get("expect_diag") and result["diagnostico"] != case["expect_diag"]:
            passed = False
        if case.get("max_prob") and result["probabilidade"] > case["max_prob"]:
            passed = False
        if case.get("min_prob") and result["probabilidade"] < case["min_prob"]:
            passed = False
        pr = case.get("prob_range")
        if pr and not (pr[0] <= result["probabilidade"] <= pr[1]):
            passed = False

        rows.append({
            "case_id": case["id"],
            "diagnostico": result["diagnostico"],
            "probabilidade": result["probabilidade"],
            "provider": interp["provider"],
            "latency_ms": interp["latency_ms"],
            "score": {
                "clarity": score.clarity,
                "relevance": score.relevance,
                "safety": score.safety,
                "actionability": score.actionability,
                "overall": score.overall,
            },
            "notes": score.notes,
            "prediction_ok": passed,
            "preview": interp["text"][:300] + "...",
        })
    return rows
