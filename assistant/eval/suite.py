"""Suíte de avaliação do assistente."""
from __future__ import annotations

import time

from assistant.guardrails import adversarial_cases, has_source, prefilter
from rag.retrieve import expected_hit, search_protocols


RAG_PROBES = [
    ("preoperative statin atrial fibrillation CABG", "17625060"),
    ("single negative troponin ACS", "25173350"),
    ("glucocorticoid croup", "10591357"),
    ("uncomplicated diverticulitis antibiotics", "22290281"),
    ("sepsis lactate", "26903338"),
]


def run_guardrails() -> dict:
    cases = adversarial_cases()
    ok = 0
    rows = []
    for case in cases:
        flags = prefilter(case["q"])
        if case["expect"] == "block":
            passed = flags["prescription_request"] and not flags["out_of_scope"]
        elif case["expect"] == "scope":
            passed = flags["out_of_scope"]
        else:
            passed = not flags["prescription_request"] and not flags["out_of_scope"]
        ok += int(passed)
        rows.append({**case, "passed": passed, "flags": flags})
    return {"n": len(cases), "passed": ok, "rate": round(ok / len(cases), 3), "rows": rows}


def run_rag(conn, k: int = 3) -> dict:
    hits = 0
    rows = []
    for query, pmid in RAG_PROBES:
        docs = search_protocols(conn, query, k=k)
        hit = expected_hit(docs, pmid, k=k)
        hits += int(hit)
        rows.append({"query": query, "pmid": pmid, "hit": hit, "top": docs[:1]})
    return {"n": len(RAG_PROBES), "hits": hits, "recall_at_k": round(hits / len(RAG_PROBES), 3), "rows": rows}


def citation_rate(answers: list[dict]) -> float:
    if not answers:
        return 0.0
    ok = sum(1 for a in answers if has_source(a.get("sources") or []))
    return round(ok / len(answers), 3)


def time_call(fn) -> tuple[object, float]:
    t0 = time.perf_counter()
    value = fn()
    return value, round((time.perf_counter() - t0) * 1000, 2)
