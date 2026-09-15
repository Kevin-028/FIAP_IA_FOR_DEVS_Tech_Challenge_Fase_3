"""
Modelo de visão computacional — gestos de mão (1 a 5).
Wrapper para carregar/salvar landmarks, treinar e predizer.
"""
import csv
import os
import threading

import joblib
import numpy as np

# ── Configuração ──────────────────────────────────────────────────────────────
# Apenas dígitos representáveis com uma única mão (contagem de dedos).
VALID_DIGITS = list(range(1, 6))  # [1, 2, 3, 4, 5]

# ── Caminhos ──────────────────────────────────────────────────────────────────
_WEB_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PROJECT_DIR = os.path.dirname(_WEB_DIR)
DATA_PATH    = os.path.join(_PROJECT_DIR, "visao", "data", "landmarks.csv")
MODEL_PATH   = os.path.join(_PROJECT_DIR, "visao", "modelo", "modelo_visao.pkl")
CM_PATH      = os.path.join(_PROJECT_DIR, "visao", "modelo", "confusion_matrix.png")

_HEADER = ["label"] + [f"{axis}{i}" for i in range(21) for axis in ("x", "y", "z")]

_model_cache = None
_cache_lock  = threading.Lock()
_csv_lock    = threading.Lock()  # evita escrita simultânea no CSV


def _load_model():
    global _model_cache
    with _cache_lock:
        if _model_cache is None and os.path.isfile(MODEL_PATH):
            _model_cache = joblib.load(MODEL_PATH)
    return _model_cache


def get_status() -> dict:
    """Retorna status do modelo e quantidade de amostras por dígito."""
    payload = _load_model()
    samples_per_digit = _count_samples()
    return {
        "model_loaded": payload is not None,
        "samples_per_digit": samples_per_digit,
        "total_samples": sum(samples_per_digit.values()),
        "metricas": payload.get("metricas", {}) if payload else {},
        "classes": payload.get("classes", []) if payload else [],
    }


def _count_samples() -> dict:
    counts = {str(i): 0 for i in VALID_DIGITS}
    if not os.path.isfile(DATA_PATH):
        return counts
    with open(DATA_PATH, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = str(row.get("label", ""))
            if label in counts:
                counts[label] += 1
    return counts


def save_sample(digit: int, landmarks: list) -> int:
    """Salva uma amostra de landmark no CSV. Retorna total para esse dígito."""
    if int(digit) not in VALID_DIGITS:
        raise ValueError(f"Dígito {digit} não suportado. Use {VALID_DIGITS}.")
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    with _csv_lock:
        file_exists = os.path.isfile(DATA_PATH)
        with open(DATA_PATH, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(_HEADER)
            writer.writerow([digit] + landmarks)
    counts = _count_samples()
    return counts.get(str(digit), 0)


def predict(landmarks: list) -> dict | None:
    """Prediz o dígito a partir de 63 landmark features normalizadas."""
    payload = _load_model()
    if payload is None:
        return None
    model = payload["modelo"]
    le    = payload["label_encoder"]
    proba = model.predict_proba([landmarks])[0]
    idx   = int(np.argmax(proba))
    digit = int(le.inverse_transform([idx])[0])
    return {"digit": digit, "confidence": float(proba[idx])}


def train() -> dict:
    """Treina o modelo com os dados em DATA_PATH. Retorna métricas."""
    global _model_cache

    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import classification_report, confusion_matrix
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    if not os.path.isfile(DATA_PATH):
        return {"success": False, "error": "Nenhum dado coletado ainda."}

    df = pd.read_csv(DATA_PATH)
    # Filtra apenas dígitos suportados (1–5), descartando legados (0, 6–9)
    df = df[df["label"].isin(VALID_DIGITS)].reset_index(drop=True)
    if len(df) < 50:
        return {"success": False, "error": f"Poucos dados válidos (1–5): {len(df)} amostras. Colete pelo menos 50."}

    X     = df.drop("label", axis=1).values
    y     = df["label"].values
    le    = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    cv_scores   = cross_val_score(model, X, y_enc, cv=min(5, len(set(y_enc))), scoring="accuracy")
    y_pred      = model.predict(X_test)
    test_acc    = float((y_pred == y_test).mean())
    labels      = le.inverse_transform(sorted(set(y_enc)))
    report_str  = classification_report(y_test, y_pred, target_names=[str(l) for l in labels])

    # Salva matriz de confusão
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Reds",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predito"); ax.set_ylabel("Real")
    ax.set_title("Matriz de Confusão — Gestos 1 a 5")
    plt.tight_layout()
    plt.savefig(CM_PATH, dpi=150)
    plt.close()

    metricas = {
        "cv_accuracy_mean": float(cv_scores.mean()),
        "cv_accuracy_std":  float(cv_scores.std()),
        "test_accuracy":    test_acc,
        "n_samples":        len(df),
        "n_classes":        len(labels),
    }
    payload = {
        "modelo": model, "label_encoder": le,
        "n_features": X.shape[1],
        "classes": [int(c) for c in le.classes_],
        "metricas": metricas,
    }
    joblib.dump(payload, MODEL_PATH)

    with _cache_lock:
        _model_cache = payload

    return {
        "success": True,
        "metricas": metricas,
        "report": report_str,
        "classes": [int(c) for c in labels],
    }
