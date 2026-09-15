"""Entry point Flask — assistente clínico (Fase 3)."""
import sys
import os
import time

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from flask import Flask, g, request, render_template

from controllers.prediction_controller import prediction_bp
from controllers.visao_controller import visao_bp
from controllers.fase2_controller import fase2_bp
from controllers.comparar_controller import comparar_bp
from controllers.llm_controller import llm_bp
from controllers.ops_controller import ops_bp
from controllers.settings_controller import settings_bp
from controllers.assistant_controller import assistant_bp, desk_context
from monitoring.logger import setup_logging, log_event
from monitoring.metrics import metrics_store

app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = "fiap-tech-challenge-fase3"

setup_logging()


@app.before_request
def _track_start():
    g._t0 = time.perf_counter()
    metrics_store.request_started()


@app.after_request
def _track_end(response):
    t0 = getattr(g, "_t0", None)
    if t0 is not None:
        latency = (time.perf_counter() - t0) * 1000
        metrics_store.record_request(
            request.path, request.method, response.status_code, latency
        )
        if request.path.endswith("/predict"):
            log_event(
                "info", "http_request",
                path=request.path, method=request.method,
                status=response.status_code, latency_ms=round(latency, 2),
            )
    metrics_store.request_finished()
    return response


app.register_blueprint(prediction_bp)
app.register_blueprint(fase2_bp)
app.register_blueprint(comparar_bp)
app.register_blueprint(ops_bp)
app.register_blueprint(settings_bp)
app.register_blueprint(llm_bp)
app.register_blueprint(visao_bp)
app.register_blueprint(assistant_bp)


@app.route("/")
def home():
    return render_template("home.html", **desk_context())


@app.context_processor
def inject_globals():
    p = request.path
    waiting, alerts = _desk_badges()
    return {
        "app_fase": 3,
        "nav_home": p == "/",
        "nav_pacientes": p.startswith("/pacientes"),
        "nav_assistente": p.startswith("/assistente"),
        "nav_validacao": p.startswith("/validacao"),
        "nav_alertas": p.startswith("/alertas"),
        "nav_modelo": p.startswith("/modelo") or p.startswith("/ops"),
        "nav_auditoria": p.startswith("/auditoria"),
        "nav_settings": p.startswith("/configuracoes"),
        "open_alerts": alerts,
        "waiting_hitl": waiting,
    }


def _desk_badges() -> tuple[int, int]:
    try:
        from hospital.db import connect, open_alert_count, waiting_hitl_count
        from hospital.paths import DB_PATH

        if not os.path.isfile(DB_PATH):
            return 0, 0
        conn = connect()
        waiting = waiting_hitl_count(conn)
        alerts = open_alert_count(conn)
        conn.close()
        return waiting, alerts
    except Exception:
        return 0, 0


if __name__ == "__main__":
    app.run(debug=True, port=5000, threaded=True)
