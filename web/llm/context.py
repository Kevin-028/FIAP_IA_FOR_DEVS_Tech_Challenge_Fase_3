"""Contexto clínico a partir do formulário e das features."""
from models.cancer_model import FEATURE_META


def highlight_features(form_data: dict, features: list[str]) -> list[dict]:
    highlights = []
    for name in features:
        if name not in form_data or form_data.get(name) in ("", "—", None):
            continue
        try:
            val = float(form_data[name])
        except (TypeError, ValueError):
            continue
        meta = FEATURE_META.get(name, {})
        try:
            lo = float(meta.get("min", 0))
            hi = float(meta.get("max", 1))
        except (TypeError, ValueError):
            continue
        span = hi - lo if hi > lo else 1.0
        if val < lo:
            status = "abaixo do típico"
        elif val > hi:
            status = "acima do típico"
        elif val > lo + 0.75 * span:
            status = "tendência alta"
        elif val < lo + 0.25 * span:
            status = "tendência baixa"
        else:
            status = "dentro da faixa"
        if status != "dentro da faixa":
            highlights.append({
                "name": name,
                "label": meta.get("label", name),
                "value": round(val, 4),
                "min": lo,
                "max": hi,
                "status": status,
                "tooltip": meta.get("tooltip", ""),
            })
    order = {"acima do típico": 0, "abaixo do típico": 1, "tendência alta": 2, "tendência baixa": 3}
    highlights.sort(key=lambda x: order.get(x["status"], 9))
    return highlights


def build_llm_template_context(mode: str, result, groups: dict, patient: dict) -> dict:
    from models.cancer_model import FEATURES

    from .feedback import FEEDBACK_TAGS

    fv = feature_values_from_groups(groups)
    if not fv and isinstance(result, dict) and mode != "comparar":
        pass
    return {
        "llm_mode": mode,
        "llm_payload": {
            "mode": mode,
            "result": result,
            "patient": patient or {},
            "feature_highlights": highlight_features(fv, FEATURES),
        },
        "llm_feedback_tags": FEEDBACK_TAGS,
    }


def feature_values_from_groups(groups: dict) -> dict:
    label_to_name = {m["label"]: n for n, m in FEATURE_META.items()}
    data = {}
    for g in groups.values():
        for inp in g.get("inputs", []):
            name = label_to_name.get(inp.get("label"))
            if not name:
                continue
            v = inp.get("value")
            if v in ("—", None, ""):
                continue
            try:
                data[name] = float(v)
            except (TypeError, ValueError):
                data[name] = v
    return data
