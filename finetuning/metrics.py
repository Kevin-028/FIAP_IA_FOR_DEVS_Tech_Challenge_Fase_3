"""Métricas de yes/no/maybe."""
from __future__ import annotations

LABELS = ("yes", "no", "maybe")


def normalize_label(text: str) -> str | None:
    head = (text or "").strip().lower()
    for label in LABELS:
        if head.startswith(label):
            return label
    for label in LABELS:
        if label in head[:40]:
            return label
    return None


def confusion(gold: list[str], pred: list[str]) -> dict[str, dict[str, int]]:
    matrix = {g: {p: 0 for p in LABELS} for g in LABELS}
    for g, p in zip(gold, pred):
        if g in matrix and p in matrix[g]:
            matrix[g][p] += 1
    return matrix


def accuracy(gold: list[str], pred: list[str]) -> float:
    if not gold:
        return 0.0
    hit = sum(1 for g, p in zip(gold, pred) if g == p)
    return round(hit / len(gold), 4)


def macro_f1(gold: list[str], pred: list[str]) -> float:
    scores = []
    for label in LABELS:
        tp = sum(1 for g, p in zip(gold, pred) if g == label and p == label)
        fp = sum(1 for g, p in zip(gold, pred) if g != label and p == label)
        fn = sum(1 for g, p in zip(gold, pred) if g == label and p != label)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        scores.append(0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec))
    return round(sum(scores) / len(scores), 4)
