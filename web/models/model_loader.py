"""
Carregamento lazy dos modelos Fase 1 (baseline) e Fase 2 (otimizado pelo AG).
Fase 1 = Regressão Logística padrão (analise_cancer).
Fase 2 = mesma família, hiperparâmetros encontrados pelo AG + limiar ajustado.
"""
import os
import sys
import threading

import joblib

_WEB_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PROJECT_DIR = os.path.dirname(_WEB_DIR)
_FASE1_PATH = os.path.join(_PROJECT_DIR, "modelo", "modelo_fase1.pkl")
_FASE2_PATH = os.path.join(
    _PROJECT_DIR, "resultados", "otimizacao_genetica", "notebook", "modelo_otimizado_ag.pkl"
)

_lock = threading.Lock()
_cache: dict = {}


def _train_baseline_fase1() -> dict:
    """Treina LR baseline (Fase 1) com limiar ajustado por CV — igual ao analise_cancer."""
    sys.path.insert(0, _PROJECT_DIR)
    from otimizacao_genetica.config import BASELINE_HYPERPARAMS, MODELO_FASE1
    from otimizacao_genetica.data import prepare_data
    from otimizacao_genetica.fitness import build_estimator, evaluate_on_test
    from otimizacao_genetica.save_web_model import evaluate_at_threshold, tune_threshold

    data = prepare_data()
    params = BASELINE_HYPERPARAMS[MODELO_FASE1].copy()
    params.setdefault("random_state", 42)
    model = build_estimator(MODELO_FASE1, params)
    model.fit(data["X_train_proc"], data["y_train"])

    limiar = tune_threshold(model, data["X_train_proc"], data["y_train"])
    test_metrics = evaluate_at_threshold(
        model, data["X_test_proc"], data["y_test"], limiar
    )
    metricas = {
        "Acuracia": test_metrics["accuracy"],
        "Recall": test_metrics["recall"],
        "Precisao": test_metrics["accuracy"],
        "F1-Score": test_metrics["f1"],
        "ROC-AUC": test_metrics["roc_auc"],
    }

    return {
        "modelo": model,
        "preprocessor": data["preprocessor"],
        "features": data["features"],
        "limiar": limiar,
        "nome_modelo": f"{MODELO_FASE1} (Fase 1)",
        "modelo_nome": MODELO_FASE1,
        "hyperparams": params,
        "metricas": metricas,
        "fase": 1,
        "origem": "treino_automatico_lr",
    }


def load_fase1() -> dict:
    with _lock:
        if "fase1" not in _cache:
            if os.path.isfile(_FASE1_PATH):
                pkg = joblib.load(_FASE1_PATH)
                pkg["fase"] = 1
                pkg.setdefault("origem", "modelo_fase1.pkl")
            else:
                pkg = _train_baseline_fase1()
            _cache["fase1"] = pkg
        return _cache["fase1"]


def load_fase2() -> dict:
    with _lock:
        if "fase2" not in _cache:
            if not os.path.isfile(_FASE2_PATH):
                raise FileNotFoundError(
                    f"Modelo Fase 2 não encontrado: {_FASE2_PATH}\n"
                    "Execute: python -m otimizacao_genetica.save_web_model"
                )
            pkg = joblib.load(_FASE2_PATH)
            pkg["fase"] = 2
            pkg["origem"] = "modelo_otimizado_ag.pkl"
            pkg.setdefault("limiar", 0.50)
            pkg.setdefault("modelo_nome", "Regressao Logistica")
            pkg.setdefault("nome_modelo", "Regressao Logistica (AG)")
            if "metricas_otimizado" in pkg:
                m = pkg["metricas_otimizado"]
                pkg["metricas"] = {
                    "Acuracia": m.get("accuracy", 0),
                    "Recall": m.get("recall", 0),
                    "Precisao": m.get("accuracy", 0),
                    "F1-Score": m.get("f1", 0),
                    "ROC-AUC": m.get("roc_auc", 0),
                }
            _cache["fase2"] = pkg
        return _cache["fase2"]


def _normalize_metricas(raw: dict) -> dict:
    if not raw:
        return {"Acuracia": 0, "Recall": 0, "Precisao": 0, "F1-Score": 0, "ROC-AUC": 0}
    return {
        "Acuracia": raw.get("Acuracia", raw.get("accuracy", 0)),
        "Recall": raw.get("Recall", raw.get("recall", 0)),
        "Precisao": raw.get("Precisao", raw.get("precision", raw.get("accuracy", 0))),
        "F1-Score": raw.get("F1-Score", raw.get("f1", 0)),
        "ROC-AUC": raw.get("ROC-AUC", raw.get("roc_auc", 0)),
    }


def get_comparison_data() -> dict:
    """Métricas de comparação baseline vs AG para exibição."""
    f1 = load_fase1()
    base_m = _normalize_metricas(f1["metricas"])

    f2_info = {"disponivel": False, "erro": None}
    try:
        f2 = load_fase2()
        opt_m2 = _normalize_metricas(f2.get("metricas", {}))
        f2_info = {
            "disponivel": True,
            "nome": f2.get("nome_modelo"),
            "metricas": opt_m2,
            "hyperparams": f2.get("hyperparams", {}),
            "ga_config": f2.get("ga_config", "exp1"),
            "origem": f2.get("origem"),
        }
    except FileNotFoundError as e:
        f2_info["erro"] = str(e)

    csv_path = os.path.join(
        _PROJECT_DIR, "resultados", "otimizacao_genetica", "notebook",
        "comparacao_anterior_vs_otimizado.csv",
    )
    tabela = []
    if os.path.isfile(csv_path):
        import pandas as pd
        df = pd.read_csv(csv_path)
        tabela = df.to_dict("records")

    return {
        "fase1": {"nome": f1.get("nome_modelo"), "metricas": base_m, "origem": f1.get("origem")},
        "fase2": f2_info,
        "tabela_modelos": tabela,
    }
