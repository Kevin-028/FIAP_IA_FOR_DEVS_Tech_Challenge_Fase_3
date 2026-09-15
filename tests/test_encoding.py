"""Testes da codificação de cromossomos do AG."""
import numpy as np
import pytest

from otimizacao_genetica.config import MODEL_NAMES
from otimizacao_genetica.encoding import (
    GENE_COUNTS,
    decode_chromosome,
    get_gene_count,
    random_chromosome,
)


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_gene_count_positivo(model_name):
    assert get_gene_count(model_name) == GENE_COUNTS[model_name]
    assert get_gene_count(model_name) > 0


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_decode_com_genes_zero_e_um(model_name):
    n = get_gene_count(model_name)
    for genes in (np.zeros(n), np.ones(n)):
        params = decode_chromosome(model_name, genes)
        assert isinstance(params, dict) and params


def test_decode_tamanho_invalido_gera_erro():
    with pytest.raises(ValueError):
        decode_chromosome("KNN", np.array([0.5, 0.5]))


def test_decode_knn_dentro_das_faixas():
    params = decode_chromosome("KNN", np.array([0.0, 0.0, 0.0, 0.0]))
    assert 3 <= params["n_neighbors"] <= 25
    assert params["weights"] in ("uniform", "distance")
    assert params["metric"] in ("euclidean", "manhattan", "minkowski")


def test_decode_logistica_l1_usa_liblinear():
    params = decode_chromosome("Regressao Logistica", np.array([0.5, 0.99, 0.0, 0.0]))
    assert params["penalty"] == "l1"
    assert params["solver"] == "liblinear"


def test_random_chromosome_no_intervalo_unitario():
    rng = np.random.default_rng(42)
    genes = random_chromosome("Random Forest", rng)
    assert len(genes) == get_gene_count("Random Forest")
    assert np.all(genes >= 0) and np.all(genes <= 1)


def test_random_chromosome_deterministico_com_seed():
    a = random_chromosome("SVM", np.random.default_rng(7))
    b = random_chromosome("SVM", np.random.default_rng(7))
    assert np.allclose(a, b)
