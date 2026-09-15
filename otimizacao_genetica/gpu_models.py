"""
Modelos com suporte a GPU (XGBoost CUDA, Regressão Logística PyTorch).
Interface compatível com sklearn: fit / predict / predict_proba.
"""
from __future__ import annotations

import numpy as np

from .device import USE_GPU, detect_devices


class TorchLogisticRegression:
    """Regressão logística treinada na GPU via PyTorch (quando CUDA disponível)."""

    def __init__(self, C=1.0, class_weight=None, max_iter=500, lr=0.05, random_state=42):
        self.C = C
        self.class_weight = class_weight
        self.max_iter = max_iter
        self.lr = lr
        self.random_state = random_state
        self._device = "cpu"
        self._weights = None
        self._bias = None

    def fit(self, X, y):
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        info = detect_devices()
        self._device = "cuda" if info["torch_cuda"] and USE_GPU else "cpu"

        torch.manual_seed(self.random_state)
        X_np = np.asarray(X, dtype=np.float32)
        y_np = np.asarray(y, dtype=np.float32)

        X_t = torch.tensor(X_np, device=self._device)
        y_t = torch.tensor(y_np, device=self._device).unsqueeze(1)

        n_features = X_np.shape[1]
        w = torch.zeros(n_features, 1, device=self._device, requires_grad=True)
        b = torch.zeros(1, device=self._device, requires_grad=True)

        if self.class_weight == "balanced":
            n_pos = y_np.sum()
            n_neg = len(y_np) - n_pos
            pos_w = len(y_np) / (2 * max(n_pos, 1))
            neg_w = len(y_np) / (2 * max(n_neg, 1))
            weights = torch.where(y_t > 0.5, pos_w, neg_w)
        else:
            weights = torch.ones_like(y_t)

        optimizer = torch.optim.Adam([w, b], lr=self.lr)
        l2 = 1.0 / (self.C * len(y_np))

        # mini-batches aceleram na GPU
        dataset = TensorDataset(X_t, y_t, weights)
        loader = DataLoader(dataset, batch_size=min(128, len(y_np)), shuffle=True)

        for _ in range(self.max_iter):
            for xb, yb, wb in loader:
                logits = xb @ w + b
                loss = (nn.functional.binary_cross_entropy_with_logits(
                    logits, yb, reduction="none"
                ) * wb).mean()
                loss = loss + l2 * (w ** 2).sum()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        self._weights = w.detach()
        self._bias = b.detach()
        return self

    def _proba_pos(self, X):
        import torch

        X_t = torch.tensor(np.asarray(X, dtype=np.float32), device=self._device)
        logits = X_t @ self._weights + self._bias
        proba = torch.sigmoid(logits).squeeze(1)
        return proba.cpu().numpy()

    def predict_proba(self, X):
        p1 = self._proba_pos(X)
        return np.column_stack([1 - p1, p1])

    def predict(self, X):
        return (self._proba_pos(X) >= 0.5).astype(int)


def _map_gb_to_xgb(hyperparams: dict, use_cuda: bool) -> dict:
    """Mapeia hiperparâmetros do Gradient Boosting sklearn → XGBoost."""
    max_depth = hyperparams.get("max_depth", 3)
    if max_depth is None:
        max_depth = 6
    return {
        "n_estimators": hyperparams.get("n_estimators", 100),
        "learning_rate": hyperparams.get("learning_rate", 0.1),
        "max_depth": int(max_depth),
        "min_child_weight": max(1, hyperparams.get("min_samples_split", 2) // 2),
        "subsample": hyperparams.get("subsample", 1.0),
        "tree_method": "hist",
        "device": "cuda" if use_cuda else "cpu",
        "eval_metric": "logloss",
        "verbosity": 0,
        "random_state": hyperparams.get("random_state", 42),
    }


def build_gpu_estimator(model_name: str, hyperparams: dict):
    """
    Retorna estimador com GPU quando possível.
    None = usar sklearn padrão.
    """
    if not USE_GPU:
        return None

    info = detect_devices()

    if model_name == "Regressao Logistica" and info["torch_cuda"]:
        return TorchLogisticRegression(
            C=hyperparams.get("C", 1.0),
            class_weight=hyperparams.get("class_weight"),
            max_iter=400,
            random_state=hyperparams.get("random_state", 42),
        )

    if model_name == "Gradient Boosting":
        import xgboost as xgb

        params = _map_gb_to_xgb(hyperparams, use_cuda=info["xgboost_gpu"])
        return xgb.XGBClassifier(**params)

    return None
