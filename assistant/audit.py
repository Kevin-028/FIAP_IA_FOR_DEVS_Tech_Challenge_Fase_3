"""Log JSONL do grafo."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone

from hospital.paths import AUDIT_LOG


def log_turn(record: dict) -> None:
    os.makedirs(os.path.dirname(AUDIT_LOG), exist_ok=True)
    record = {"ts": datetime.now(timezone.utc).isoformat(), **record}
    with open(AUDIT_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
