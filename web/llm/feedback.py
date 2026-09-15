"""Feedback do médico sobre as respostas da LLM."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone

from .config import LOG_DIR

FEEDBACK_LOG = os.path.join(LOG_DIR, "llm_feedback.jsonl")

FEEDBACK_TAGS = [
    "Impreciso ou incorreto",
    "Genérico ou vago",
    "Não fundamentado nos dados",
    "Conduta clínica inadequada",
    "Muito longo ou prolixo",
    "Fora do contexto clínico",
    "Inventou informação (alucinação)",
    "Linguagem inadequada",
]

_MAX_TEXT = 1200


def record_feedback(data: dict) -> dict:
    rating = data.get("rating")
    if rating not in ("like", "dislike"):
        raise ValueError("rating deve ser 'like' ou 'dislike'")

    tags = [t for t in (data.get("tags") or []) if t in FEEDBACK_TAGS]

    context = data.get("context") or {}
    if not isinstance(context, dict):
        context = {}

    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "rating": rating,
        "mode": str(data.get("mode", ""))[:40],
        "provider": str(data.get("provider", ""))[:80],
        "question": str(data.get("question", ""))[:_MAX_TEXT],
        "answer": str(data.get("answer", ""))[:_MAX_TEXT],
        "reason": str(data.get("reason", ""))[:_MAX_TEXT] if rating == "dislike" else "",
        "tags": tags if rating == "dislike" else [],
        "diagnostico": str(data.get("diagnostico", ""))[:60],
        "prob": data.get("prob"),
        "context": context,
    }

    os.makedirs(LOG_DIR, exist_ok=True)
    with open(FEEDBACK_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


def load_feedback() -> list[dict]:
    if not os.path.isfile(FEEDBACK_LOG):
        return []
    records = []
    with open(FEEDBACK_LOG, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def summarize_feedback() -> dict:
    records = load_feedback()
    total = len(records)
    likes = sum(1 for r in records if r.get("rating") == "like")
    dislikes = sum(1 for r in records if r.get("rating") == "dislike")
    satisfaction = round(likes / total * 100, 1) if total else 0.0

    tag_counts: dict[str, int] = {}
    for r in records:
        if r.get("rating") == "dislike":
            for t in r.get("tags", []):
                tag_counts[t] = tag_counts.get(t, 0) + 1
    tag_counts = dict(sorted(tag_counts.items(), key=lambda kv: kv[1], reverse=True))

    recent_dislikes = [r for r in records if r.get("rating") == "dislike"]
    recent_dislikes = list(reversed(recent_dislikes))[:25]

    return {
        "total": total,
        "likes": likes,
        "dislikes": dislikes,
        "satisfaction": satisfaction,
        "tag_counts": tag_counts,
        "recent_dislikes": recent_dislikes,
    }
