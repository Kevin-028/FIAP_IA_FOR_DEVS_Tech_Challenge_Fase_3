"""Testes das camadas de modelo (Fase 1 e Fase 2)."""
import pytest

from models.cancer_model import FEATURES, predict as predict_f1
from models.cancer_model_fase2 import predict as predict_f2, predict_compare

REQUIRED_KEYS = {
    "probabilidade", "nivel", "diagnostico", "limiar", "nome_modelo", "metricas", "fase",
}


def _assert_contrato(result, fase):
    assert REQUIRED_KEYS.issubset(result.keys())
    assert 0.0 <= result["probabilidade"] <= 100.0
    assert result["diagnostico"] in ("MALIGNO", "BENIGNO")
    assert result["fase"] == fase


def test_features_tem_20_itens():
    assert len(FEATURES) == 20


def test_predict_fase1_contrato(benigno_form):
    _assert_contrato(predict_f1(benigno_form), fase=1)


def test_predict_fase2_contrato(maligno_form):
    _assert_contrato(predict_f2(maligno_form), fase=2)


def test_caso_benigno_classificado_benigno(benigno_form):
    assert predict_f1(benigno_form)["diagnostico"] == "BENIGNO"


def test_caso_maligno_classificado_maligno(maligno_form):
    assert predict_f2(maligno_form)["diagnostico"] == "MALIGNO"


def test_maligno_tem_prob_maior_que_benigno(benigno_form, maligno_form):
    assert predict_f2(maligno_form)["probabilidade"] > predict_f2(benigno_form)["probabilidade"]


def test_predict_compare_estrutura(maligno_form):
    comp = predict_compare(maligno_form)
    assert set(comp.keys()) >= {"fase1", "fase2", "diff_prob", "concordancia", "concordancia_label"}
    assert comp["concordancia_label"] in ("Concordam", "Divergem")


def test_predict_valor_invalido_gera_erro():
    with pytest.raises((ValueError, KeyError, TypeError)):
        predict_f1({f: "texto" for f in FEATURES})
