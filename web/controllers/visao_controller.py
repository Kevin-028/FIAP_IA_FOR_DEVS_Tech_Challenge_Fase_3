"""Controller de visão computacional."""
import os
from flask import Blueprint, render_template, request, jsonify, url_for

from models.visao_model import get_status, predict, save_sample, train, CM_PATH

visao_bp = Blueprint("visao_bp", __name__, url_prefix="/visao")


@visao_bp.route("/")
def index():
    status = get_status()
    return render_template("visao/index.html", status=status)


@visao_bp.route("/coletar")
def coletar():
    status = get_status()
    return render_template("visao/coletar.html", status=status)


@visao_bp.route("/treinar")
def treinar():
    status = get_status()
    has_cm = os.path.isfile(CM_PATH)
    cm_url = url_for("static", filename="visao_cm.png") if has_cm else None
    if has_cm:
        import shutil
        dst = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "static", "visao_cm.png")
        shutil.copy2(CM_PATH, dst)
    return render_template("visao/treinar.html", status=status, cm_url=cm_url)


@visao_bp.route("/api/status")
def api_status():
    return jsonify(get_status())


@visao_bp.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(force=True)
    landmarks = data.get("landmarks", [])
    if len(landmarks) != 63:
        return jsonify({"error": f"Esperado 63 landmarks, recebido {len(landmarks)}"}), 400
    result = predict(landmarks)
    if result is None:
        return jsonify({"error": "Modelo não carregado. Treine o modelo primeiro."}), 503
    return jsonify(result)


@visao_bp.route("/api/save_sample", methods=["POST"])
def api_save_sample():
    data   = request.get_json(force=True)
    digit  = int(data.get("digit", -1))
    landmarks = data.get("landmarks", [])
    if not (0 <= digit <= 9):
        return jsonify({"error": "Dígito inválido"}), 400
    if len(landmarks) != 63:
        return jsonify({"error": f"Esperado 63 landmarks, recebido {len(landmarks)}"}), 400
    count = save_sample(digit, landmarks)
    return jsonify({"digit": digit, "count": count})


@visao_bp.route("/api/train", methods=["POST"])
def api_train():
    result = train()
    return jsonify(result)
