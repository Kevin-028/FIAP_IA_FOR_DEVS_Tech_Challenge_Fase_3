"""
train_model.py
--------------
Treina um classificador de gestos (0-9) com os landmarks coletados.

Como usar:
  python train_model.py

Gera: modelo/modelo_visao.pkl
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

DATA_PATH  = os.path.join(os.path.dirname(__file__), "data", "landmarks.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "modelo", "modelo_visao.pkl")


def main():
    if not os.path.isfile(DATA_PATH):
        print("ERRO: arquivo de dados não encontrado.")
        print("Execute collect_data.py primeiro.")
        return

    print("Carregando dados...")
    df = pd.read_csv(DATA_PATH)
    print(f"  Amostras: {len(df)}")
    print(f"  Distribuição:\n{df['label'].value_counts().sort_index().to_string()}\n")

    X = df.drop("label", axis=1).values
    y = df["label"].values

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    print("Treinando Random Forest...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # Cross-validation
    cv_scores = cross_val_score(model, X, y_enc, cv=5, scoring="accuracy")
    print(f"  CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Avaliação no test set
    y_pred = model.predict(X_test)
    acc = (y_pred == y_test).mean()
    print(f"  Test Accuracy: {acc:.4f}\n")

    labels = le.inverse_transform(sorted(set(y_enc)))
    print(classification_report(
        y_test, y_pred,
        target_names=[str(l) for l in labels]
    ))

    # Matriz de confusão
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predito")
    ax.set_ylabel("Real")
    ax.set_title("Matriz de Confusão — Gestos 0 a 9")
    plt.tight_layout()
    cm_path = os.path.join(os.path.dirname(__file__), "modelo", "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"  Matriz de confusão salva em: {cm_path}")

    # Salva modelo + encoder
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    payload = {
        "modelo": model,
        "label_encoder": le,
        "n_features": X.shape[1],
        "classes": list(le.classes_),
        "metricas": {
            "cv_accuracy_mean": float(cv_scores.mean()),
            "cv_accuracy_std": float(cv_scores.std()),
            "test_accuracy": float(acc),
        },
    }
    joblib.dump(payload, MODEL_PATH)
    print(f"\nModelo salvo em: {MODEL_PATH}")


if __name__ == "__main__":
    main()
