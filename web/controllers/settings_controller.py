"""Controller de configurações da UI."""
from flask import Blueprint, jsonify, render_template

settings_bp = Blueprint("settings", __name__, url_prefix="/configuracoes")

SETTINGS_SCHEMA = [
    {
        "group": "aparencia",
        "title": "Aparência",
        "description": "Tema visual da aplicação.",
        "fields": [
            {
                "key": "theme",
                "label": "Tema",
                "type": "theme",
                "default": "system",
                "options": [
                    {"value": "light", "label": "Claro", "hint": "Fundo claro"},
                    {"value": "dark", "label": "Escuro", "hint": "Fundo escuro"},
                    {"value": "system", "label": "Sistema", "hint": "Segue o SO"},
                ],
            },
        ],
    },
]


@settings_bp.route("/")
def index():
    return render_template(
        "settings/index.html",
        schema=SETTINGS_SCHEMA,
    )


@settings_bp.route("/api/schema")
def api_schema():
    return jsonify({"version": 1, "groups": SETTINGS_SCHEMA})
