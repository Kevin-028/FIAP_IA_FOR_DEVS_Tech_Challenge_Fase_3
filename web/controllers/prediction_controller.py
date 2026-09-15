"""
Controller layer — Fase 1 (baseline) — rotas /fase1
"""
import os
import json
import time

from flask import Blueprint, render_template, request, redirect, url_for

from models.cancer_model import (
    predict, FEATURES, FEATURE_META, FEATURE_GROUPS,
    build_form_groups, build_form_groups_empty, get_model_info,
)
from monitoring.logger import log_event
from monitoring.metrics import metrics_store

prediction_bp = Blueprint("prediction", __name__, url_prefix="/fase1")


@prediction_bp.route("/", methods=["GET"])
def index():
    info = get_model_info()
    return render_template(
        "fase1/index.html",
        groups=build_form_groups_empty(),
        nome_modelo=info["nome_modelo"],
        metricas=info["metricas"],
    )


@prediction_bp.route("/predict", methods=["POST"])
def predict_view():
    form_data = request.form.to_dict()
    t0 = time.perf_counter()
    try:
        result = predict(form_data)
    except Exception as e:
        log_event("error", "fase1_predict_failed", error=str(e))
        info = get_model_info()
        return render_template(
            "fase1/index.html",
            groups=build_form_groups_empty(),
            nome_modelo=info["nome_modelo"],
            metricas=info["metricas"],
            error=str(e),
        )

    latency = (time.perf_counter() - t0) * 1000
    metrics_store.record_prediction("fase1", latency)
    log_event("info", "fase1_predict", latency_ms=round(latency, 2),
              diagnostico=result["diagnostico"], prob=result["probabilidade"])

    patient = {
        "nome": form_data.get("paciente_nome", "").strip(),
        "data": form_data.get("paciente_data", "").strip(),
        "prontuario": form_data.get("paciente_prontuario", "").strip(),
    }
    groups = build_form_groups(form_data)
    return render_template(
        "fase1/result.html",
        result=result,
        groups=groups,
        patient=patient,
    )


@prediction_bp.route("/exemplos", methods=["GET"])
def exemplos():
    _dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(_dir, "examples_data.json"), encoding="utf-8") as fp:
        raw = json.load(fp)

    CAT_META = {
        "benigno_claro":   {"label": "Benigno Evidente",        "color": "success",  "icon": "🟢"},
        "benigno_limiar":  {"label": "Benigno com Atipia Leve", "color": "info",     "icon": "🔵"},
        "maligno_claro":   {"label": "Maligno Evidente",        "color": "danger",   "icon": "🔴"},
        "maligno_limiar":  {"label": "Maligno Borderline",      "color": "warning",  "icon": "🟡"},
        "falso_positivo":  {"label": "Falso Positivo ⚠️",       "color": "fp",       "icon": "⚠️"},
        "falso_negativo":  {"label": "Falso Negativo ⚠️",       "color": "fn",       "icon": "⚠️"},
    }

    examples = []
    for i, ex in enumerate(raw):
        meta = CAT_META.get(ex["cat"], {})
        dados = {f: ex[f] for f in FEATURES}
        form_data = {**dados, "paciente_nome": f"Exemplo #{i + 1}"}
        try:
            result = predict(form_data)
            pred = result["diagnostico"]
            prob = result["probabilidade"]
        except Exception:
            pred = ex["pred_s"]
            prob = ex["prob"]
        examples.append({
            "id":      i,
            "cat":     ex["cat"],
            "label":   meta.get("label", ex["cat"]),
            "color":   meta.get("color", "secondary"),
            "icon":    meta.get("icon", ""),
            "real":    ex["real_s"],
            "pred":    pred,
            "prob":    prob,
            "dados":   dados,
        })

    info = get_model_info()
    return render_template(
        "exemplos.html",
        examples=examples,
        features=FEATURES,
        feature_meta=FEATURE_META,
        nome_modelo=info["nome_modelo"],
    )

