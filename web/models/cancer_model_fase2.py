"""
Model layer — Fase 2 (Algoritmo Genético).
"""
import numpy as np
import pandas as pd

from models.model_loader import load_fase2
from models.cancer_model import predict as predict_fase1_base


def _risk_band(prob: float) -> tuple:
    if prob < 0.20:
        return "Baixíssimo", "🟢", "success"
    if prob < 0.50:
        return "Moderado", "🟡", "warning"
    if prob < 0.75:
        return "Alto", "🟠", "orange"
    return "Muito Alto", "🔴", "danger"


def predict(form_data: dict) -> dict:
    pkg = load_fase2()
    features = pkg["features"]
    model = pkg["modelo"]
    preprocessor = pkg["preprocessor"]
    limiar = pkg.get("limiar", 0.5)

    values = {f: float(form_data.get(f, np.nan)) for f in features}
    df = pd.DataFrame([values])[features]
    prob = model.predict_proba(preprocessor.transform(df))[0, 1]
    nivel, nivel_icon, nivel_class = _risk_band(prob)
    diagnostico = "MALIGNO" if prob >= limiar else "BENIGNO"

    metricas_raw = pkg.get("metricas", {})
    metricas = {
        "Acuracia": metricas_raw.get("Acuracia", metricas_raw.get("accuracy", 0)),
        "Recall": metricas_raw.get("Recall", metricas_raw.get("recall", 0)),
        "Precisao": metricas_raw.get("Precisao", metricas_raw.get("accuracy", 0)),
        "F1-Score": metricas_raw.get("F1-Score", metricas_raw.get("f1", 0)),
        "ROC-AUC": metricas_raw.get("ROC-AUC", metricas_raw.get("roc_auc", 0)),
    }
    return {
        "probabilidade": round(prob * 100, 1),
        "nivel": nivel,
        "nivel_icon": nivel_icon,
        "nivel_class": nivel_class,
        "diagnostico": diagnostico,
        "limiar": round(limiar * 100, 1),
        "nome_modelo": pkg.get("nome_modelo", "Regressao Logistica (AG)"),
        "metricas": metricas,
        "fase": 2,
        "modulo": "Fase 2 — Algoritmo Genético",
        "badge_class": "fase2-badge",
        "ga_config": pkg.get("ga_config", ""),
        "hyperparams": pkg.get("hyperparams", {}),
    }


def predict_compare(form_data: dict) -> dict:
    """Executa ambos os modelos e retorna comparação."""
    r1 = predict_fase1_base(form_data, fase=1)
    r2 = predict(form_data)
    diff_prob = round(r2["probabilidade"] - r1["probabilidade"], 1)
    concordo = r1["diagnostico"] == r2["diagnostico"]
    return {
        "fase1": r1,
        "fase2": r2,
        "diff_prob": diff_prob,
        "concordancia": concordo,
        "concordancia_label": "Concordam" if concordo else "Divergem",
    }
