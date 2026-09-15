"""Caminhos do projeto."""
from __future__ import annotations

import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HOSPITAL_DATA = os.path.join(ROOT, "hospital_data")
DB_PATH = os.path.join(HOSPITAL_DATA, "hospital.db")
CHECKPOINTS_PATH = os.path.join(HOSPITAL_DATA, "checkpoints.sqlite")
PROTOCOLS_DIR = os.path.join(HOSPITAL_DATA, "protocols")
PUBMEDQA_RAW = os.path.join(HOSPITAL_DATA, "pubmedqa_raw")
SFT_DIR = os.path.join(HOSPITAL_DATA, "sft")
CHROMA_DIR = os.path.join(HOSPITAL_DATA, "chroma")
MODELS_DIR = os.path.join(ROOT, "models")
ANON_REPORT = os.path.join(HOSPITAL_DATA, "anonimizacao_report.md")
LOG_DIR = os.path.join(ROOT, "web", "logs")
AUDIT_LOG = os.path.join(LOG_DIR, "assistant_audit.jsonl")
