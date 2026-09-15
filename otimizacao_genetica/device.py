"""
Detecção de hardware e configuração de aceleração (GPU / CPU paralelo).
"""
from __future__ import annotations

import os

# True = usa GPU onde suportado; False = força CPU/sklearn
USE_GPU = os.environ.get("OTIMIZACAO_USE_GPU", "1") != "0"

# Workers para avaliação paralela da população (modelos CPU)
N_JOBS = int(os.environ.get("OTIMIZACAO_N_JOBS", "-1"))


def detect_devices() -> dict:
    """Detecta GPU NVIDIA e backends disponíveis."""
    info = {
        "use_gpu": USE_GPU,
        "xgboost_gpu": False,
        "torch_cuda": False,
        "device_name": None,
        "n_jobs": N_JOBS,
        "backends": [],
    }

    if not USE_GPU:
        info["backends"].append("sklearn (CPU, paralelo)")
        return info

    # XGBoost + CUDA
    try:
        import numpy as np
        import xgboost as xgb

        X = np.random.randn(32, 8).astype(np.float32)
        y = (np.random.rand(32) > 0.5).astype(int)
        m = xgb.XGBClassifier(
            tree_method="hist", device="cuda",
            n_estimators=5, verbosity=0,
        )
        m.fit(X, y)
        info["xgboost_gpu"] = True
        info["backends"].append("XGBoost (CUDA)")
    except Exception:
        info["backends"].append("XGBoost (CPU)")

    # PyTorch CUDA
    try:
        import torch

        if torch.cuda.is_available():
            info["torch_cuda"] = True
            info["device_name"] = torch.cuda.get_device_name(0)
            info["backends"].append(f"PyTorch ({info['device_name']})")
        else:
            info["backends"].append("PyTorch (CPU)")
    except ImportError:
        info["backends"].append("PyTorch (nao instalado)")

    # sklearn paralelo para RF, KNN, SVM
    info["backends"].append(f"sklearn (CPU, n_jobs={N_JOBS})")
    return info


def print_device_info():
    info = detect_devices()
    print("Aceleracao:")
    print(f"  GPU ativa: {info['use_gpu']}")
    if info["device_name"]:
        print(f"  Placa: {info['device_name']}")
    print(f"  XGBoost CUDA: {info['xgboost_gpu']}")
    print(f"  PyTorch CUDA: {info['torch_cuda']}")
    print(f"  Backends: {', '.join(info['backends'])}")
    return info
