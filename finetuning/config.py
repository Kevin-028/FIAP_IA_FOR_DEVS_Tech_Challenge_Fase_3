"""Perfis de treino por VRAM."""
from __future__ import annotations

import os

SEED = 42
BASE_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
FALLBACK_MODEL = "Qwen/Qwen2.5-3B-Instruct"
LOCAL_QWEN = os.environ.get(
    "LLM_MODEL_DIR",
    r"C:\workspace\Modelos\Qwen2.5-3B-Instruct",
)


def resolve_base_model() -> str:
    token = (os.environ.get("HF_TOKEN") or "").strip()
    if token and os.environ.get("FIAP_FORCE_LLAMA") == "1":
        return BASE_MODEL
    if os.path.isdir(LOCAL_QWEN):
        return LOCAL_QWEN
    return FALLBACK_MODEL

SYSTEM_PROMPT = (
    "You are a hospital clinical assistant. Answer in English. "
    "Cite PMID or protocol_id. Never prescribe a drug, dose, or route. "
    "End drafts that suggest a conduct with REQUIRES PHYSICIAN VALIDATION."
)


def detect_profile() -> dict:
    vram_gb = None
    name = None
    try:
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            vram_gb = round(props.total_memory / (1024 ** 3), 1)
            name = props.name
    except Exception:
        vram_gb = None

    if vram_gb is None:
        chosen = 12.0
    else:
        chosen = vram_gb

    sample = 5500
    if chosen <= 8:
        profile = {"max_seq": 512, "batch": 1, "grad_accum": 8, "sample": sample, "label": "<=8GB"}
    elif chosen < 16:
        profile = {"max_seq": 512, "batch": 4, "grad_accum": 2, "sample": sample, "label": "12-16GB"}
    else:
        profile = {"max_seq": 512, "batch": 4, "grad_accum": 2, "sample": sample, "label": ">=16GB"}

    profile.update({
        "vram_gb": vram_gb,
        "device_name": name,
        "assumed_gb": None if vram_gb is not None else 12.0,
        "base_model": resolve_base_model(),
        "lora_r": 32,
        "lora_alpha": 64,
        "epochs": 1,
        "lr": 1.5e-4,
        "seed": SEED,
        "grad_checkpoint": chosen <= 8,
    })
    return profile
