"""Testes da camada LLM em modo offline."""
import json

import pytest

from llm.client import FallbackClient, OllamaClient, get_client
from llm.interpreter import chat, get_provider_info, interpret
from llm.prompts import (
    SYSTEM_PROMPT,
    build_case_context,
    build_user_prompt,
)


@pytest.fixture()
def result_fase2():
    return {
        "nome_modelo": "Regressao Logistica (Fase 2 — AG)",
        "diagnostico": "MALIGNO",
        "probabilidade": 82.0,
        "nivel": "Muito Alto",
        "limiar": 50.0,
        "metricas": {"Acuracia": 0.97, "Recall": 0.95, "ROC-AUC": 0.99},
    }


def test_get_client_usa_fallback_no_ambiente_de_teste():
    assert isinstance(get_client(), FallbackClient)


def test_provider_info_fallback():
    info = get_provider_info()
    assert info["provider"] == "fallback"
    assert info["using_real_llm"] is False
    assert info["model"] == "fallback/regras-clinicas"


def test_system_prompt_tem_regras_de_seguranca():
    assert "histopat" in SYSTEM_PROMPT.lower()
    assert "## Resumo executivo" in SYSTEM_PROMPT


def test_build_case_context_inclui_dados(result_fase2):
    ctx = build_case_context({"mode": "fase2", "result": result_fase2})
    assert "MALIGNO" in ctx
    assert "82" in ctx


def test_build_user_prompt_estende_contexto(result_fase2):
    payload = {"mode": "fase2", "result": result_fase2}
    assert build_case_context(payload) in build_user_prompt(payload)


def test_interpret_retorna_secoes_estruturadas(result_fase2):
    out = interpret({"mode": "fase2", "result": result_fase2})
    assert out["provider"] == "fallback/regras-clinicas"
    assert "## Resumo executivo" in out["text"]
    assert "Interpretação do resultado" in out["text"]
    assert isinstance(out["sections"], dict) and out["sections"]


def test_interpret_reflete_probabilidade(result_fase2):
    out = interpret({"mode": "fase2", "result": result_fase2})
    assert "82" in out["text"]


def test_fallback_client_complete_direto(result_fase2):
    text = FallbackClient().complete("sys", build_user_prompt({"mode": "fase2", "result": result_fase2}))
    assert "## Resumo executivo" in text


def test_chat_fallback_responde_pergunta(result_fase2):
    out = chat({
        "mode": "fase2",
        "result": result_fase2,
        "question": "Quais condutas você recomenda?",
    })
    assert "Ollama" not in out["text"] or "modo local" in out["text"].lower()
    assert "biópsia" in out["text"].lower() or "especialista" in out["text"].lower()
    assert out["provider"] == "fallback/regras-clinicas"


def test_chat_fallback_sobre_sensibilidade(result_fase2):
    out = chat({
        "mode": "fase2",
        "result": result_fase2,
        "feature_highlights": [
            {
                "label": "Variação do Raio",
                "value": 1.2,
                "min": 0.1,
                "max": 0.5,
                "status": "acima",
            }
        ],
        "question": "O que mais influenciou este resultado?",
    })
    assert "Variação do Raio" in out["text"] or "achados" in out["text"].lower()


def test_chat_prefer_fallback_nao_chama_ollama(result_fase2, monkeypatch):
    called = {"ollama": False}

    class Boom:
        provider_name = "ollama/fake"

        def chat(self, *a, **k):
            called["ollama"] = True
            raise RuntimeError("não deveria chamar ollama")

    monkeypatch.setattr("llm.interpreter.get_client", lambda: Boom())
    out = chat({
        "mode": "fase2",
        "result": result_fase2,
        "question": "Resumo do resultado",
        "prefer_fallback": True,
    })
    assert called["ollama"] is False
    assert out["provider"] == "fallback/regras-clinicas"
    assert "MALIGNO" in out["text"]


def test_chat_sem_pergunta_gera_erro(result_fase2):
    with pytest.raises(ValueError):
        chat({"mode": "fase2", "result": result_fase2, "question": "   "})


def test_ollama_extract_prefere_content():
    data = {"message": {"content": "resposta", "thinking": "raciocínio"}}
    assert OllamaClient._extract(data) == "resposta"


def test_ollama_extract_usa_thinking_se_content_vazio():
    data = {"message": {"content": "", "thinking": "só raciocínio"}}
    assert OllamaClient._extract(data) == "só raciocínio"


def test_ollama_extract_vazio_gera_erro():
    with pytest.raises(RuntimeError):
        OllamaClient._extract({"message": {"content": "", "thinking": ""}})
