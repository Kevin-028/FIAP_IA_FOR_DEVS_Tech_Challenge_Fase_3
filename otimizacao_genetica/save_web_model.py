#!/usr/bin/env python
"""
Treina o modelo foco (vencedor do AG) e salva o pacote para a web,
com ajuste de limiar de decisão (mesmo procedimento da Fase 1).

Uso:
    python -m otimizacao_genetica.save_web_model
"""
import os

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import cross_val_predict

from otimizacao_genetica.config import BASELINE_HYPERPARAMS, EXPERIMENT_CONFIGS, MODELO_FOCO
from otimizacao_genetica.data import prepare_data
from otimizacao_genetica.experiments import _serialize
from otimizacao_genetica.fitness import build_estimator
from otimizacao_genetica.genetic_algorithm import GeneticAlgorithm

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(_PROJECT_ROOT, "resultados", "otimizacao_genetica", "notebook")
PKL_PATH = os.path.join(OUTPUT_DIR, "modelo_otimizado_ag.pkl")
CSV_PATH = os.path.join(OUTPUT_DIR, "comparacao_anterior_vs_otimizado.csv")


def tune_threshold(model, X_train, y_train, cv: int = 5) -> float:
    """Escolhe o limiar que maximiza F1 nas predições de CV (sem tocar no teste)."""
    proba = cross_val_predict(
        model, X_train, y_train, cv=cv, method="predict_proba"
    )[:, 1]
    thresholds = np.arange(0.20, 0.71, 0.01)
    f1s = [f1_score(y_train, (proba >= t).astype(int)) for t in thresholds]
    return float(thresholds[int(np.argmax(f1s))])


def evaluate_at_threshold(model, X_test, y_test, limiar: float) -> dict:
    """Métricas no teste usando o limiar ajustado."""
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= limiar).astype(int)
    return {
        "accuracy": float(accuracy_score(y_test, pred)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
    }


def save_web_model(exp_config=None, verbose: bool = True) -> str:
    exp_config = exp_config or EXPERIMENT_CONFIGS[0]
    data = prepare_data()

    if verbose:
        print(f"Modelo foco: {MODELO_FOCO}")
        print(f"Experimento: {exp_config.name}")

    baseline_params = BASELINE_HYPERPARAMS[MODELO_FOCO].copy()
    baseline_params.setdefault("random_state", 42)
    baseline_model = build_estimator(MODELO_FOCO, baseline_params)
    baseline_model.fit(data["X_train_proc"], data["y_train"])
    limiar_base = tune_threshold(baseline_model, data["X_train_proc"], data["y_train"])
    test_base = evaluate_at_threshold(
        baseline_model, data["X_test_proc"], data["y_test"], limiar_base
    )
    baseline = {"hyperparams": baseline_params, "test": test_base, "limiar": limiar_base}

    ga = GeneticAlgorithm(
        MODELO_FOCO,
        exp_config,
        data["X_train_proc"],
        data["y_train"],
    )
    ga_result = ga.run(verbose=verbose)
    best = ga_result.best_individual

    # Limiar ajustado por CV no treino (mesmo procedimento da Fase 1)
    limiar = tune_threshold(
        build_estimator(MODELO_FOCO, best.hyperparams),
        data["X_train_proc"],
        data["y_train"],
    )

    modelo_final = build_estimator(MODELO_FOCO, best.hyperparams)
    modelo_final.fit(data["X_train_proc"], data["y_train"])

    test_opt = evaluate_at_threshold(
        modelo_final, data["X_test_proc"], data["y_test"], limiar
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pacote = {
        "modelo": modelo_final,
        "preprocessor": data["preprocessor"],
        "features": data["features"],
        "limiar": limiar,
        "ga_config": exp_config.name,
        "modelo_nome": MODELO_FOCO,
        "nome_modelo": f"{MODELO_FOCO} (Fase 2 — AG)",
        "hyperparams": best.hyperparams,
        "hyperparams_baseline": BASELINE_HYPERPARAMS[MODELO_FOCO],
        "metricas_otimizado": test_opt,
        "metricas_baseline": baseline["test"],
    }
    joblib.dump(pacote, PKL_PATH)

    # CSV: uma linha — baseline vs otimizado (modelo único do AG)
    import pandas as pd

    base = baseline["test"]
    row = {
        "Modelo": MODELO_FOCO,
        "Recall Anterior": base["recall"],
        "Recall Otimizado": test_opt["recall"],
        "F1 Anterior": base["f1"],
        "F1 Otimizado": test_opt["f1"],
        "AUC Anterior": base["roc_auc"],
        "AUC Otimizado": test_opt["roc_auc"],
        "Melhorou?": "Sim" if test_opt["recall"] > base["recall"] or test_opt["f1"] > base["f1"] else "Nao",
    }
    pd.DataFrame([row]).to_csv(CSV_PATH, index=False)

    if verbose:
        print(f"\nSalvo: {PKL_PATH}")
        print(f"Baseline  — acc={base['accuracy']:.4f} recall={base['recall']:.4f} f1={base['f1']:.4f}")
        print(f"Otimizado — acc={test_opt['accuracy']:.4f} recall={test_opt['recall']:.4f} f1={test_opt['f1']:.4f}")
        print(f"Limiar baseline={limiar_base:.2f} | otimizado={limiar:.2f}")
        print(f"Hiperparâmetros AG: {_serialize(best.hyperparams)}")

    return PKL_PATH


if __name__ == "__main__":
    save_web_model()
