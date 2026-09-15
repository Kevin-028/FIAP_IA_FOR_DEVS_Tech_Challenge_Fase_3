"""
Função fitness e avaliação de modelos com validação cruzada.
Suporta GPU (XGBoost CUDA, PyTorch) e paralelismo CPU (joblib).
"""
from __future__ import annotations

import warnings

import numpy as np
from joblib import Parallel, delayed
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from .config import BASELINE_HYPERPARAMS, FITNESS_WEIGHTS
from .device import N_JOBS, USE_GPU, detect_devices
from .encoding import decode_chromosome
from .gpu_models import build_gpu_estimator

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Modelos que usam GPU — avaliados sequencialmente na placa
GPU_MODELS = {"Regressao Logistica", "Gradient Boosting"}


def uses_gpu(model_name: str) -> bool:
    if not USE_GPU:
        return False
    if model_name not in GPU_MODELS:
        return False
    info = detect_devices()
    if model_name == "Regressao Logistica":
        return info["torch_cuda"]
    if model_name == "Gradient Boosting":
        return info["xgboost_gpu"]
    return False


def build_estimator(model_name: str, hyperparams: dict):
    """Instancia classificador — GPU quando disponível, senão sklearn."""
    gpu_model = build_gpu_estimator(model_name, hyperparams)
    if gpu_model is not None:
        return gpu_model

    if model_name == "Regressao Logistica":
        return LogisticRegression(**hyperparams)
    if model_name == "Random Forest":
        return RandomForestClassifier(**hyperparams)
    if model_name == "Gradient Boosting":
        return GradientBoostingClassifier(**hyperparams)
    if model_name == "KNN":
        return KNeighborsClassifier(**hyperparams)
    if model_name == "SVM":
        return SVC(**hyperparams)
    raise ValueError(f"Modelo desconhecido: {model_name}")


def compute_fitness(metrics: dict) -> float:
    return sum(FITNESS_WEIGHTS[k] * metrics[k] for k in FITNESS_WEIGHTS)


def _cv_metrics(
    model_name: str,
    model,
    X_train,
    y_train,
    hyperparams: dict,
    cv: int,
) -> dict:
    """Validação cruzada — sklearn paralelo (CPU) ou manual (GPU)."""
    model_cls = type(model).__name__

    if model_cls in ("TorchLogisticRegression", "XGBClassifier"):
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        scores = {"accuracy": [], "recall": [], "f1": [], "roc_auc": []}
        for train_idx, val_idx in skf.split(X_train, y_train):
            m = build_estimator(model_name, hyperparams)
            m.fit(X_train[train_idx], y_train[train_idx])
            y_pred = m.predict(X_train[val_idx])
            y_proba = m.predict_proba(X_train[val_idx])[:, 1]
            scores["accuracy"].append(accuracy_score(y_train[val_idx], y_pred))
            scores["recall"].append(recall_score(y_train[val_idx], y_pred, zero_division=0))
            scores["f1"].append(f1_score(y_train[val_idx], y_pred, zero_division=0))
            scores["roc_auc"].append(roc_auc_score(y_train[val_idx], y_proba))
        return {k: float(np.mean(v)) for k, v in scores.items()}

    scoring = {
        "accuracy": "accuracy",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }
    cv_results = cross_validate(
        model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=N_JOBS
    )
    return {
        "accuracy": float(cv_results["test_accuracy"].mean()),
        "recall": float(cv_results["test_recall"].mean()),
        "f1": float(cv_results["test_f1"].mean()),
        "roc_auc": float(cv_results["test_roc_auc"].mean()),
    }


def evaluate_hyperparams(
    model_name: str,
    hyperparams: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv: int = 5,
) -> dict:
    try:
        model = build_estimator(model_name, hyperparams)
        metrics = _cv_metrics(model_name, model, X_train, y_train, hyperparams, cv)
        metrics["fitness"] = compute_fitness(metrics)
        return metrics
    except Exception:
        return {
            "accuracy": 0.0, "recall": 0.0, "f1": 0.0,
            "roc_auc": 0.0, "fitness": 0.0,
        }


def evaluate_chromosome(
    model_name: str,
    genes: np.ndarray,
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv: int = 5,
) -> tuple[dict, dict]:
    hyperparams = decode_chromosome(model_name, genes)
    metrics = evaluate_hyperparams(model_name, hyperparams, X_train, y_train, cv=cv)
    return metrics, hyperparams


def evaluate_chromosomes_batch(
    model_name: str,
    genes_list: list[np.ndarray],
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv: int = 5,
) -> list[tuple[dict, dict]]:
    """
    Avalia vários cromossomos de uma vez.
    GPU: sequencial na placa | CPU: paralelo com joblib.
    """
    if uses_gpu(model_name):
        return [
            evaluate_chromosome(model_name, g, X_train, y_train, cv)
            for g in genes_list
        ]

    results = Parallel(n_jobs=N_JOBS, prefer="processes")(
        delayed(evaluate_chromosome)(model_name, g, X_train, y_train, cv)
        for g in genes_list
    )
    return results


def evaluate_on_test(
    model_name: str,
    hyperparams: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    model = build_estimator(model_name, hyperparams)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
    }
    metrics["fitness"] = compute_fitness(metrics)
    return metrics


def evaluate_baseline(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    cv: int = 5,
) -> dict:
    params = BASELINE_HYPERPARAMS[model_name].copy()
    if model_name == "Regressao Logistica":
        params.setdefault("random_state", 42)
    cv_metrics = evaluate_hyperparams(model_name, params, X_train, y_train, cv=cv)
    test_metrics = evaluate_on_test(
        model_name, params, X_train, y_train, X_test, y_test
    )
    return {
        "hyperparams": params,
        "cv": cv_metrics,
        "test": test_metrics,
    }
