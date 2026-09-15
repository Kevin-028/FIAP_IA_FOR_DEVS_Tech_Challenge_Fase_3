"""Controller ops — health, metrics, monitoramento."""
import time

from flask import Blueprint, jsonify, render_template, request

from monitoring.logger import setup_logging
from monitoring.metrics import metrics_store

ops_bp = Blueprint("ops", __name__)


@ops_bp.route("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "service": "fiap-tech-challenge",
        "fase": 3,
        "timestamp": time.time(),
    })


@ops_bp.route("/api/metrics")
def metrics():
    return jsonify(metrics_store.snapshot())


@ops_bp.route("/ops/monitoramento")
def monitoramento():
    from controllers.assistant_controller import _confusion_table, desk_context

    ctx = desk_context()
    ctx["base_matrix"] = _confusion_table(ctx["sft"].get("base_confusion") or {})
    ctx["ft_matrix"] = _confusion_table(ctx["sft"].get("ft_confusion") or {})
    return render_template("fase3/modelo.html", **ctx)


@ops_bp.route("/api/llm/engine", methods=["GET"])
def engine_status():
    from llm.engine import engine
    return jsonify(engine.status())


@ops_bp.route("/api/llm/engine/load", methods=["POST"])
def engine_load():
    from llm.engine import engine
    try:
        return jsonify(engine.load())
    except Exception as exc:
        return jsonify({"loaded": False, "error": str(exc)}), 400


@ops_bp.route("/api/llm/engine/unload", methods=["POST"])
def engine_unload():
    from llm.engine import engine
    return jsonify(engine.unload())
