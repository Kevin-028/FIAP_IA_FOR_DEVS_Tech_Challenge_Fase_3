"""Testes de feedback da LLM."""
import importlib

import pytest

import llm.feedback as feedback


@pytest.fixture()
def temp_log(tmp_path, monkeypatch):
    log_file = tmp_path / "llm_feedback.jsonl"
    monkeypatch.setattr(feedback, "FEEDBACK_LOG", str(log_file))
    monkeypatch.setattr(feedback, "LOG_DIR", str(tmp_path))
    return log_file


def test_record_like_minimo(temp_log):
    rec = feedback.record_feedback({"rating": "like", "answer": "boa resposta"})
    assert rec["rating"] == "like"
    assert rec["tags"] == []
    assert rec["reason"] == ""
    assert temp_log.exists()


def test_record_dislike_com_tags_e_contexto(temp_log):
    rec = feedback.record_feedback({
        "rating": "dislike",
        "answer": "resposta ruim",
        "reason": "genérica demais",
        "tags": ["Genérico ou vago", "tag_invalida"],
        "diagnostico": "MALIGNO",
        "prob": 80,
        "context": {"mode": "fase2", "result": {"nome_modelo": "Regressao Logistica (Fase 2 — AG)"}},
    })
    assert rec["tags"] == ["Genérico ou vago"]
    assert rec["reason"] == "genérica demais"
    assert rec["context"]["result"]["nome_modelo"] == "Regressao Logistica (Fase 2 — AG)"


def test_rating_invalido_gera_erro(temp_log):
    with pytest.raises(ValueError):
        feedback.record_feedback({"rating": "talvez"})


def test_context_nao_dict_e_ignorado(temp_log):
    rec = feedback.record_feedback({"rating": "like", "context": "não é dict"})
    assert rec["context"] == {}


def test_load_ignora_linha_corrompida(temp_log):
    feedback.record_feedback({"rating": "like", "answer": "ok"})
    with open(temp_log, "a", encoding="utf-8") as f:
        f.write("{linha corrompida}\n")
    records = feedback.load_feedback()
    assert len(records) == 1


def test_summarize_agrega_estatisticas(temp_log):
    feedback.record_feedback({"rating": "like", "answer": "a"})
    feedback.record_feedback({"rating": "like", "answer": "b"})
    feedback.record_feedback({
        "rating": "dislike", "answer": "c",
        "tags": ["Impreciso ou incorreto"], "reason": "x",
    })
    summary = feedback.summarize_feedback()
    assert summary["total"] == 3
    assert summary["likes"] == 2
    assert summary["dislikes"] == 1
    assert summary["satisfaction"] == pytest.approx(66.7, abs=0.1)
    assert summary["tag_counts"]["Impreciso ou incorreto"] == 1


def test_feedback_tags_nao_vazio():
    assert isinstance(feedback.FEEDBACK_TAGS, list) and len(feedback.FEEDBACK_TAGS) >= 3
