"""Leitura dos artefatos gravados pelo treino."""
from __future__ import annotations

import json
import os

from hospital.paths import AUDIT_LOG, MODELS_DIR, ROOT, SFT_DIR


def _read_json(path: str) -> dict | None:
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _loss_ends(history: list) -> tuple[float | None, float | None]:
    if not history:
        return None, None
    first = history[0].get("loss")
    last = history[-1].get("loss")
    return first, last


def snapshot() -> dict:
    train = _read_json(os.path.join(SFT_DIR, "train_metrics.json")) or {}
    eval_report = _read_json(os.path.join(SFT_DIR, "eval_model.json")) or {}
    stats = _read_json(os.path.join(SFT_DIR, "stats.json")) or {}
    history = train.get("history") or []
    loss_0, loss_n = _loss_ends(history)
    base = eval_report.get("base") or {}
    tuned = eval_report.get("fine_tuned") or {}
    adapter_dir = os.environ.get("LLM_ADAPTER_DIR", os.path.join(ROOT, "models", "adapter"))
    base_dir = os.environ.get(
        "LLM_MODEL_DIR",
        os.path.join(os.environ.get("LLM_MODELS_DIR", r"C:\workspace\Modelos"), "Qwen2.5-3B-Instruct"),
    )
    adapter_ok = os.path.isfile(os.path.join(adapter_dir, "adapter_config.json"))
    return {
        "ready": bool(train) and bool(eval_report),
        "adapter_ok": adapter_ok,
        "adapter_dir": adapter_dir,
        "base_model_dir": base_dir,
        "weights_dir": MODELS_DIR,
        "n": train.get("n"),
        "hospital": train.get("hospital"),
        "hospital_unique": train.get("hospital_unique"),
        "pubmed": train.get("pubmed"),
        "epochs": train.get("epochs"),
        "lora_r": train.get("lora_r"),
        "max_seq": train.get("max_seq"),
        "loss_start": loss_0,
        "loss_end": loss_n,
        "steps": len(history),
        "history": history,
        "train_examples": stats.get("train_examples"),
        "test_labeled": stats.get("test_labeled"),
        "template": stats.get("template"),
        "label_counts": stats.get("train_label_counts") or {},
        "base_accuracy": base.get("accuracy"),
        "base_macro_f1": base.get("macro_f1"),
        "base_n": base.get("n"),
        "ft_accuracy": tuned.get("accuracy"),
        "ft_macro_f1": tuned.get("macro_f1"),
        "ft_n": tuned.get("n"),
        "base_confusion": base.get("confusion") or {},
        "ft_confusion": tuned.get("confusion") or {},
    }


def audit_turns(limit: int = 40) -> list[dict]:
    if not os.path.isfile(AUDIT_LOG):
        return []
    rows = []
    with open(AUDIT_LOG, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows[-limit:][::-1]
