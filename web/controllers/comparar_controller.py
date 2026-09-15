"""
Controller — comparação lado a lado Fase 1 vs Fase 2.
"""
import time

from flask import Blueprint, render_template, request

from models.cancer_model import (
    FEATURES, FEATURE_META, FEATURE_GROUPS,
    build_form_groups, build_form_groups_empty, get_model_info,
)
from models.cancer_model_fase2 import predict_compare
from models.model_loader import get_comparison_data
from monitoring.logger import log_event
from monitoring.metrics import metrics_store

comparar_bp = Blueprint("comparar", __name__, url_prefix="/comparar")


@comparar_bp.route("/", methods=["GET"])
def index():
    info = get_model_info()
    comp = get_comparison_data()
    return render_template(
        "comparar/index.html",
        groups=build_form_groups_empty(),
        comparacao=comp,
        nome_modelo_f1=info["nome_modelo"],
    )


@comparar_bp.route("/predict", methods=["POST"])
def predict_view():
    form_data = request.form.to_dict()
    t0 = time.perf_counter()
    try:
        result = predict_compare(form_data)
    except Exception as e:
        log_event("error", "comparar_failed", error=str(e))
        return render_template(
            "comparar/index.html",
            groups=build_form_groups_empty(),
            comparacao=get_comparison_data(),
            error=str(e),
        )

    latency = (time.perf_counter() - t0) * 1000
    metrics_store.record_prediction("comparar", latency)
    log_event("info", "comparar_predict", latency_ms=round(latency, 2),
              concordancia=result["concordancia"])

    patient = {
        "nome": form_data.get("paciente_nome", "").strip(),
        "data": form_data.get("paciente_data", "").strip(),
        "prontuario": form_data.get("paciente_prontuario", "").strip(),
    }
    comp = get_comparison_data()
    groups = build_form_groups(form_data)
    dados = {f: form_data.get(f, "") for f in FEATURES}
    return render_template(
        "comparar/result.html",
        result=result,
        groups=groups,
        patient=patient,
        comparacao=comp,
        dados=dados,
    )
