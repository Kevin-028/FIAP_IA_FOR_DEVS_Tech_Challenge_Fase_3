"""Testes do coletor de métricas."""
from monitoring.metrics import MAX_WORKERS, MIN_WORKERS, MetricsStore


def test_snapshot_inicial_zerado():
    store = MetricsStore()
    snap = store.snapshot()
    assert snap["request_count"] == 0
    assert snap["prediction_count"] == 0
    assert snap["error_rate"] == 0
    assert snap["workers"] == MIN_WORKERS


def test_record_request_incrementa_contadores():
    store = MetricsStore()
    store.record_request("/fase2/predict", "POST", 200, 12.5)
    store.record_request("/fase2/predict", "POST", 500, 20.0)
    snap = store.snapshot()
    assert snap["request_count"] == 2
    assert snap["error_count"] == 1
    assert snap["error_rate"] == 50.0


def test_record_prediction_contabiliza_por_fase():
    store = MetricsStore()
    store.record_prediction("fase1", 5.0)
    store.record_prediction("fase2", 7.0)
    store.record_prediction("fase2", 9.0)
    snap = store.snapshot()
    assert snap["prediction_count"] == 3
    assert snap["predictions_by_fase"]["fase2"] == 2
    assert snap["predictions_by_fase"]["fase1"] == 1


def test_fase_desconhecida_cai_em_fase1():
    store = MetricsStore()
    store.record_prediction("inexistente", 5.0)
    assert store.snapshot()["predictions_by_fase"]["fase1"] == 1


def test_auto_scale_up_e_down():
    store = MetricsStore()
    for _ in range(20):
        store.request_started()
    assert store.snapshot()["workers"] <= MAX_WORKERS
    assert store.snapshot()["workers"] > MIN_WORKERS
    for _ in range(20):
        store.request_finished()
    for _ in range(10):
        store.record_request("/", "GET", 200, 1.0)
    assert store.snapshot()["workers"] >= MIN_WORKERS


def test_active_requests_nunca_negativo():
    store = MetricsStore()
    store.request_finished()
    assert store.snapshot()["active_requests"] == 0
