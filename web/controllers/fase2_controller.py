"""
Controller Fase 2 — diagnóstico com modelo otimizado por Algoritmo Genético.
"""
import time

from flask import Blueprint, render_template, request

from models.cancer_model import (
    FEATURES, FEATURE_META, FEATURE_GROUPS,
    build_form_groups, build_form_groups_empty,
)
from models.cancer_model_fase2 import predict
from models.model_loader import get_comparison_data, load_fase2, _normalize_metricas
from llm.context import build_llm_template_context
from llm.interpreter import get_provider_info
from monitoring.logger import log_event
from monitoring.metrics import metrics_store

fase2_bp = Blueprint("fase2", __name__, url_prefix="/fase2")


@fase2_bp.route("/", methods=["GET"])
def index():
    pkg = load_fase2()
    comp = get_comparison_data()
    return render_template(
        "fase2/index.html",
        groups=build_form_groups_empty(),
        nome_modelo=pkg.get("nome_modelo", "Regressao Logistica (Fase 2 — AG)"),
        metricas=_normalize_metricas(pkg.get("metricas", {})),
        ga_config=pkg.get("ga_config", ""),
        hyperparams=pkg.get("hyperparams", {}),
        comparacao=comp,
    )


@fase2_bp.route("/predict", methods=["POST"])
def predict_view():
    form_data = request.form.to_dict()
    t0 = time.perf_counter()
    try:
        result = predict(form_data)
    except Exception as e:
        log_event("error", "fase2_predict_failed", error=str(e))
        return render_template(
            "fase2/index.html",
            groups=build_form_groups_empty(),
            nome_modelo="Regressao Logistica (Fase 2 — AG)",
            metricas=_normalize_metricas({}),
            ga_config="",
            error=str(e),
        )

    latency = (time.perf_counter() - t0) * 1000
    metrics_store.record_prediction("fase2", latency)
    log_event("info", "fase2_predict", latency_ms=round(latency, 2),
              diagnostico=result["diagnostico"], prob=result["probabilidade"])

    patient = {
        "nome": form_data.get("paciente_nome", "").strip(),
        "data": form_data.get("paciente_data", "").strip(),
        "prontuario": form_data.get("paciente_prontuario", "").strip(),
    }
    groups = build_form_groups(form_data)
    return render_template(
        "fase2/result.html",
        result=result,
        groups=groups,
        patient=patient,
        **build_llm_template_context("fase2", result, groups, patient),
        llm_config=get_provider_info(),
    )
