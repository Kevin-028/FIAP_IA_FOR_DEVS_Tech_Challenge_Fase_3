"""
Coletor de métricas em memória + auto-scaling por demanda.
"""
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field

MIN_WORKERS = int(os.environ.get("FIAP_MIN_WORKERS", "2"))
MAX_WORKERS = int(os.environ.get("FIAP_MAX_WORKERS", "8"))
SCALE_UP_THRESHOLD = int(os.environ.get("FIAP_SCALE_UP_RPS", "5"))
SCALE_DOWN_THRESHOLD = int(os.environ.get("FIAP_SCALE_DOWN_RPS", "2"))


@dataclass
class MetricsStore:
  request_count: int = 0
  error_count: int = 0
  prediction_count: int = 0
  total_latency_ms: float = 0.0
  active_requests: int = 0
  workers: int = MIN_WORKERS
  latencies: deque = field(default_factory=lambda: deque(maxlen=200))
  predictions_by_fase: dict = field(default_factory=lambda: {"fase1": 0, "fase2": 0, "comparar": 0})
  _lock: threading.Lock = field(default_factory=threading.Lock)

  def record_request(self, path: str, method: str, status: int, latency_ms: float):
    with self._lock:
      self.request_count += 1
      self.total_latency_ms += latency_ms
      self.latencies.append(latency_ms)
      if status >= 400:
        self.error_count += 1
      self._auto_scale()

  def record_prediction(self, fase: str, latency_ms: float):
    with self._lock:
      self.prediction_count += 1
      key = fase if fase in self.predictions_by_fase else "fase1"
      self.predictions_by_fase[key] += 1
      self.latencies.append(latency_ms)

  def request_started(self):
    with self._lock:
      self.active_requests += 1
      self._auto_scale()

  def request_finished(self):
    with self._lock:
      self.active_requests = max(0, self.active_requests - 1)

  def _auto_scale(self):
    """Escala workers conforme demanda (requisições ativas + taxa recente)."""
    load = self.active_requests
    if load >= SCALE_UP_THRESHOLD and self.workers < MAX_WORKERS:
      self.workers += 1
    elif load <= SCALE_DOWN_THRESHOLD and self.workers > MIN_WORKERS:
      self.workers -= 1

  def snapshot(self) -> dict:
    with self._lock:
      lats = list(self.latencies)
      avg = sum(lats) / len(lats) if lats else 0
      p95 = sorted(lats)[int(len(lats) * 0.95)] if len(lats) >= 5 else avg
      return {
        "request_count": self.request_count,
        "error_count": self.error_count,
        "prediction_count": self.prediction_count,
        "predictions_by_fase": dict(self.predictions_by_fase),
        "active_requests": self.active_requests,
        "workers": self.workers,
        "min_workers": MIN_WORKERS,
        "max_workers": MAX_WORKERS,
        "avg_latency_ms": round(avg, 2),
        "p95_latency_ms": round(p95, 2),
        "error_rate": round(self.error_count / max(self.request_count, 1) * 100, 2),
        "timestamp": time.time(),
      }


metrics_store = MetricsStore()
