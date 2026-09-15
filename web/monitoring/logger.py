"""
Logging estruturado para tracking de desempenho (Fase 2).
"""
import json
import logging
import os
from datetime import datetime, timezone

_LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
_LOG_FILE = os.path.join(_LOG_DIR, "app.log")


class JsonFormatter(logging.Formatter):
  def format(self, record):
    payload = {
      "ts": datetime.now(timezone.utc).isoformat(),
      "level": record.levelname,
      "msg": record.getMessage(),
      "module": record.module,
    }
    if hasattr(record, "extra_data"):
      payload.update(record.extra_data)
    return json.dumps(payload, ensure_ascii=False)


def setup_logging():
  os.makedirs(_LOG_DIR, exist_ok=True)
  root = logging.getLogger("fiap_tc")
  if root.handlers:
    return root
  root.setLevel(logging.INFO)
  fh = logging.FileHandler(_LOG_FILE, encoding="utf-8")
  fh.setFormatter(JsonFormatter())
  sh = logging.StreamHandler()
  sh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
  root.addHandler(fh)
  root.addHandler(sh)
  return root


def log_event(level: str, msg: str, **kwargs):
  logger = setup_logging()
  record = logger.makeRecord(
    logger.name, getattr(logging, level.upper(), logging.INFO),
    "", 0, msg, (), None,
  )
  record.extra_data = kwargs
  logger.handle(record)
