"""
Codificação de cromossomos (genes) ↔ hiperparâmetros dos modelos.
Cada indivíduo é um vetor de floats em [0, 1], um gene por hiperparâmetro.
"""
import numpy as np

from .config import MODEL_NAMES

# Número de genes por modelo (ordem fixa na decodificação)
GENE_COUNTS = {
    "Regressao Logistica": 4,   # C, penalty, class_weight, solver_idx
    "Random Forest": 5,         # n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features
    "Gradient Boosting": 5,     # n_estimators, learning_rate, max_depth, min_samples_split, subsample
    "KNN": 4,                   # n_neighbors, weights, metric, p
    "SVM": 4,                   # C, kernel, gamma, class_weight
}


def get_gene_count(model_name: str) -> int:
    return GENE_COUNTS[model_name]


def _lerp(g: float, lo: float, hi: float) -> float:
    return lo + float(np.clip(g, 0, 1)) * (hi - lo)


def _log_lerp(g: float, lo: float, hi: float) -> float:
    return float(np.exp(_lerp(g, np.log(lo), np.log(hi))))


def _choice(g: float, options: list) -> object:
    idx = int(np.clip(g, 0, 0.9999) * len(options))
    return options[idx]


def decode_chromosome(model_name: str, genes: np.ndarray) -> dict:
    """Decodifica vetor de genes [0,1] em dicionário de hiperparâmetros."""
    g = np.asarray(genes, dtype=float)
    expected = get_gene_count(model_name)
    if len(g) != expected:
        raise ValueError(f"{model_name}: esperado {expected} genes, recebido {len(g)}")

    if model_name == "Regressao Logistica":
        penalty = _choice(g[1], ["l2", "l1"])
        class_weight = _choice(g[2], [None, "balanced"])
        solver = "liblinear" if penalty == "l1" else _choice(g[3], ["lbfgs", "saga"])
        return {
            "C": _log_lerp(g[0], 1e-3, 100.0),
            "penalty": penalty,
            "class_weight": class_weight,
            "solver": solver,
            "max_iter": 2000,
            "random_state": 42,
        }

    if model_name == "Random Forest":
        max_depth_raw = _lerp(g[1], 2, 30)
        max_depth = int(round(max_depth_raw)) if g[1] < 0.95 else None
        max_feat_opts = ["sqrt", "log2", 0.5, 0.8]
        return {
            "n_estimators": int(round(_lerp(g[0], 50, 300))),
            "max_depth": max_depth,
            "min_samples_split": int(round(_lerp(g[2], 2, 20))),
            "min_samples_leaf": int(round(_lerp(g[3], 1, 10))),
            "max_features": _choice(g[4], max_feat_opts),
            "random_state": 42,
            "n_jobs": -1,
        }

    if model_name == "Gradient Boosting":
        return {
            "n_estimators": int(round(_lerp(g[0], 50, 300))),
            "learning_rate": _log_lerp(g[1], 0.01, 0.3),
            "max_depth": int(round(_lerp(g[2], 2, 8))),
            "min_samples_split": int(round(_lerp(g[3], 2, 20))),
            "subsample": _lerp(g[4], 0.6, 1.0),
            "random_state": 42,
        }

    if model_name == "KNN":
        metric = _choice(g[2], ["euclidean", "manhattan", "minkowski"])
        p = 1 if metric == "manhattan" else int(round(_lerp(g[3], 1, 4)))
        return {
            "n_neighbors": int(round(_lerp(g[0], 3, 25))),
            "weights": _choice(g[1], ["uniform", "distance"]),
            "metric": metric,
            "p": p,
            "n_jobs": -1,
        }

    if model_name == "SVM":
        kernel = _choice(g[1], ["rbf", "linear", "poly"])
        gamma_opts = ["scale", "auto", 0.001, 0.01, 0.1]
        return {
            "C": _log_lerp(g[0], 1e-2, 100.0),
            "kernel": kernel,
            "gamma": _choice(g[2], gamma_opts),
            "class_weight": _choice(g[3], [None, "balanced"]),
            "probability": True,
            "random_state": 42,
        }

    raise ValueError(f"Modelo desconhecido: {model_name}")


def random_chromosome(model_name: str, rng: np.random.Generator) -> np.ndarray:
    """Gera cromossomo aleatório com genes em [0, 1]."""
    return rng.uniform(0, 1, size=get_gene_count(model_name))


def describe_encoding(model_name: str) -> list[str]:
    """Descrição textual dos genes para documentação."""
    descriptions = {
        "Regressao Logistica": [
            "g0: C (log-uniforme 1e-3 a 100)",
            "g1: penalty (l2 | l1)",
            "g2: class_weight (None | balanced)",
            "g3: solver (lbfgs | saga, ou liblinear se l1)",
        ],
        "Random Forest": [
            "g0: n_estimators (50–300)",
            "g1: max_depth (2–30, ou None se >0.95)",
            "g2: min_samples_split (2–20)",
            "g3: min_samples_leaf (1–10)",
            "g4: max_features (sqrt | log2 | 0.5 | 0.8)",
        ],
        "Gradient Boosting": [
            "g0: n_estimators (50–300)",
            "g1: learning_rate (log 0.01–0.3)",
            "g2: max_depth (2–8)",
            "g3: min_samples_split (2–20)",
            "g4: subsample (0.6–1.0)",
        ],
        "KNN": [
            "g0: n_neighbors (3–25)",
            "g1: weights (uniform | distance)",
            "g2: metric (euclidean | manhattan | minkowski)",
            "g3: p (1–4, para minkowski)",
        ],
        "SVM": [
            "g0: C (log-uniforme 0.01–100)",
            "g1: kernel (rbf | linear | poly)",
            "g2: gamma (scale | auto | 0.001 | 0.01 | 0.1)",
            "g3: class_weight (None | balanced)",
        ],
    }
    return descriptions.get(model_name, [])
