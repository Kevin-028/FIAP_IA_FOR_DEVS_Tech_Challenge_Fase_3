"""
Model layer — Fase 1 (baseline). Carrega via model_loader com fallback automático.
"""
import os
import numpy as np
import pandas as pd

from models.model_loader import load_fase1

# Metadados clínicos — exportados para templates (inalterados)
#   group   → chave do grupo clínico (ver FEATURE_GROUPS abaixo)
#   label   → rótulo exibido no formulário
#   tooltip → explicação clínica resumida para o profissional de saúde
# ---------------------------------------------------------------------------
FEATURE_META = {
    # ── Tamanho e Estrutura ──────────────────────────────────────────────
    "radius_se": {
        "label": "Variação do Raio dos Núcleos",
        "tooltip": "Desvio-padrão do raio médio dos núcleos celulares na amostra. "
                   "Valores altos indicam células de tamanhos muito distintos (heterogeneidade).",
        "group": "tamanho", "step": "0.001", "min": "0.10", "max": "3.0",  "placeholder": "0.405",
    },
    "perimeter_worst": {
        "label": "Maior Perímetro do Núcleo (µm)",
        "tooltip": "Perímetro do maior núcleo celular identificado na imagem. "
                   "Núcleos com perímetro elevado sugerem crescimento celular desordenado.",
        "group": "tamanho", "step": "0.1", "min": "50.0", "max": "252.0", "placeholder": "107.3",
    },
    "texture_se": {
        "label": "Variação da Textura dos Núcleos",
        "tooltip": "Desvio-padrão da escala de cinza dos núcleos. "
                   "Alta variação indica cromatina distribuída de forma irregular.",
        "group": "tamanho", "step": "0.01", "min": "0.30", "max": "5.0",  "placeholder": "1.217",
    },
    "texture_worst": {
        "label": "Textura Máxima do Núcleo",
        "tooltip": "Maior valor de textura (desvio da escala de cinza) encontrado. "
                   "Cromatina grosseira e irregular é sinal comum em células malignas.",
        "group": "tamanho", "step": "0.1", "min": "12.0", "max": "50.0",  "placeholder": "25.7",
    },

    # ── Regularidade da Superfície ───────────────────────────────────────
    "smoothness_mean": {
        "label": "Suavidade da Borda (média)",
        "tooltip": "Variação local no comprimento dos raios da borda celular. "
                   "Bordas irregulares (baixa suavidade) são típicas de células malignas.",
        "group": "superficie", "step": "0.001", "min": "0.05", "max": "0.17", "placeholder": "0.096",
    },
    "smoothness_se": {
        "label": "Variação da Suavidade entre Células",
        "tooltip": "Desvio-padrão da suavidade de borda na população de células. "
                   "Alta variação indica população celular heterogênea.",
        "group": "superficie", "step": "0.0001", "min": "0.001", "max": "0.032", "placeholder": "0.007",
    },
    "smoothness_worst": {
        "label": "Pior Suavidade de Borda",
        "tooltip": "Menor suavidade (borda mais irregular) observada na amostra. "
                   "Valor baixo = borda altamente irregular.",
        "group": "superficie", "step": "0.001", "min": "0.07", "max": "0.23", "placeholder": "0.132",
    },
    "compactness_mean": {
        "label": "Compacidade Celular (média)",
        "tooltip": "Calculada como (perímetro² / área − 1). "
                   "Células compactas e arredondadas têm valores próximos de zero.",
        "group": "superficie", "step": "0.001", "min": "0.01", "max": "0.35", "placeholder": "0.104",
    },
    "compactness_se": {
        "label": "Variação da Compacidade",
        "tooltip": "Desvio-padrão da compacidade entre os núcleos da amostra. "
                   "Alta variação sugere diversidade morfológica celular.",
        "group": "superficie", "step": "0.001", "min": "0.002", "max": "0.14", "placeholder": "0.025",
    },
    "compactness_worst": {
        "label": "Maior Compacidade Observada",
        "tooltip": "Valor máximo de compacidade na amostra. "
                   "Células muito compactas indicam alta relação perímetro/área.",
        "group": "superficie", "step": "0.001", "min": "0.02", "max": "1.06", "placeholder": "0.254",
    },

    # ── Concavidade das Bordas ───────────────────────────────────────────
    "concavity_se": {
        "label": "Variação das Reentrâncias da Borda",
        "tooltip": "Desvio-padrão da profundidade das partes côncavas do contorno celular. "
                   "Reentrâncias variáveis indicam bordas malformadas.",
        "group": "concavidade", "step": "0.001", "min": "0.0", "max": "0.40", "placeholder": "0.032",
    },
    "concave_points_se": {
        "label": "Variação dos Pontos Côncavos",
        "tooltip": "Desvio-padrão do número de pontos côncavos do contorno. "
                   "Muitos pontos côncavos variáveis são frequentes em tumores malignos.",
        "group": "concavidade", "step": "0.001", "min": "0.0", "max": "0.054", "placeholder": "0.012",
    },
    "concavity_worst": {
        "label": "Maior Profundidade de Reentrâncias",
        "tooltip": "Profundidade máxima de concavidade observada no contorno celular. "
                   "Indica o quão recortada é a borda do núcleo mais irregular.",
        "group": "concavidade", "step": "0.001", "min": "0.0", "max": "1.26", "placeholder": "0.272",
    },
    "concave_points_worst": {
        "label": "Maior Número de Pontos Côncavos",
        "tooltip": "Número máximo de porções côncavas no contorno celular. "
                   "Contornos muito recortados são característicos de malignidade.",
        "group": "concavidade", "step": "0.001", "min": "0.0", "max": "0.30", "placeholder": "0.115",
    },

    # ── Simetria e Irregularidade ────────────────────────────────────────
    "symmetry_mean": {
        "label": "Simetria do Núcleo (média)",
        "tooltip": "Medida de simetria do formato do núcleo celular. "
                   "Células normais tendem a ser simétricas; malignidade causa assimetria.",
        "group": "simetria", "step": "0.001", "min": "0.10", "max": "0.31", "placeholder": "0.181",
    },
    "symmetry_se": {
        "label": "Variação da Simetria entre Células",
        "tooltip": "Desvio-padrão da simetria entre os núcleos. "
                   "Alta variação indica diversidade morfológica na população celular.",
        "group": "simetria", "step": "0.001", "min": "0.007", "max": "0.080", "placeholder": "0.021",
    },
    "symmetry_worst": {
        "label": "Pior Simetria Observada",
        "tooltip": "Maior assimetria encontrada na amostra. "
                   "Núcleos muito assimétricos são indicadores de atipia celular.",
        "group": "simetria", "step": "0.001", "min": "0.15", "max": "0.67", "placeholder": "0.290",
    },
    "fractal_dimension_mean": {
        "label": "Irregularidade da Borda (média)",
        "tooltip": "Dimensão fractal média da borda celular. "
                   "Valores maiores indicam contorno mais complexo e irregular ('aproximação da costeira').",
        "group": "simetria", "step": "0.001", "min": "0.04", "max": "0.10", "placeholder": "0.063",
    },
    "fractal_dimension_se": {
        "label": "Variação da Irregularidade da Borda",
        "tooltip": "Desvio-padrão da dimensão fractal entre os núcleos. "
                   "Alta variação sugere bordas de complexidade muito distinta entre células.",
        "group": "simetria", "step": "0.0001", "min": "0.0008", "max": "0.030", "placeholder": "0.004",
    },
    "fractal_dimension_worst": {
        "label": "Maior Irregularidade da Borda",
        "tooltip": "Dimensão fractal máxima observada. "
                   "Indica o núcleo com borda de maior complexidade geométrica na amostra.",
        "group": "simetria", "step": "0.001", "min": "0.05", "max": "0.21", "placeholder": "0.084",
    },
}

# Grupos clínicos — ordem e metadados de exibição
FEATURE_GROUPS = {
    "tamanho": {
        "title": "Tamanho e Estrutura dos Núcleos",
        "subtitle": "Medidas de tamanho, perímetro e textura dos núcleos celulares na amostra de biópsia.",
        "icon": "📐",
    },
    "superficie": {
        "title": "Regularidade da Superfície Celular",
        "subtitle": "Avalia o quão suave e compacta é a borda dos núcleos. Bordas irregulares são sinal de alerta.",
        "icon": "🔲",
    },
    "concavidade": {
        "title": "Concavidade das Bordas",
        "subtitle": "Quantidade e profundidade das reentrâncias (partes côncavas) no contorno dos núcleos.",
        "icon": "🌀",
    },
    "simetria": {
        "title": "Simetria e Irregularidade da Borda",
        "subtitle": "Grau de simetria e complexidade geométrica dos núcleos. Assimetria elevada é indicador de atipia.",
        "icon": "⚖️",
    },
}


def _get_pkg():
    return load_fase1()


def get_features():
    return _get_pkg()["features"]


FEATURES = list(FEATURE_META.keys())


def build_form_groups(form_data: dict):
    groups = {}
    for gkey, gmeta in FEATURE_GROUPS.items():
        groups[gkey] = {**gmeta, "inputs": []}
    for f in FEATURES:
        meta = FEATURE_META[f]
        groups[meta["group"]]["inputs"].append({
            "label": meta["label"],
            "value": form_data.get(f, "—"),
        })
    return groups


def build_form_groups_empty():
    groups = {}
    for gkey, gmeta in FEATURE_GROUPS.items():
        groups[gkey] = {**gmeta, "features": []}
    for feat in FEATURES:
        meta = FEATURE_META[feat]
        groups[meta["group"]]["features"].append({"name": feat, **meta})
    return groups


def get_model_info() -> dict:
    pkg = _get_pkg()
    return {
        "nome_modelo": pkg.get("nome_modelo", "Regressao Logistica (baseline)"),
        "metricas": pkg.get("metricas", {}),
        "limiar": pkg.get("limiar", 0.5),
    }


def predict(form_data: dict, fase: int = 1) -> dict:
    """Predição Fase 1 (baseline)."""
    pkg = _get_pkg()
    features = pkg["features"]
    model = pkg["modelo"]
    preprocessor = pkg["preprocessor"]
    limiar = pkg.get("limiar", 0.5)
    nome_modelo = pkg.get("nome_modelo", "Regressao Logistica (baseline)")
    metricas = pkg.get("metricas", {})

    values = {f: float(form_data.get(f, np.nan)) for f in features}
    df = pd.DataFrame([values])[features]
    prob = model.predict_proba(preprocessor.transform(df))[0, 1]

    if prob < 0.20:
        nivel, nivel_icon, nivel_class = "Baixíssimo", "🟢", "success"
    elif prob < 0.50:
        nivel, nivel_icon, nivel_class = "Moderado", "🟡", "warning"
    elif prob < 0.75:
        nivel, nivel_icon, nivel_class = "Alto", "🟠", "orange"
    else:
        nivel, nivel_icon, nivel_class = "Muito Alto", "🔴", "danger"

    diagnostico = "MALIGNO" if prob >= limiar else "BENIGNO"

    return {
        "probabilidade": round(prob * 100, 1),
        "nivel": nivel,
        "nivel_icon": nivel_icon,
        "nivel_class": nivel_class,
        "diagnostico": diagnostico,
        "limiar": round(limiar * 100, 1),
        "nome_modelo": nome_modelo,
        "metricas": metricas,
        "fase": fase,
        "modulo": "Fase 1 — Baseline",
        "badge_class": "fase1-badge",
    }
