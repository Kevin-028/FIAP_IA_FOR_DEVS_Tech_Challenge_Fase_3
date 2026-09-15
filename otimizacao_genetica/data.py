"""
Carregamento e pré-processamento dos dados — espelha o pipeline do Módulo 1.
"""
import os

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, PowerTransformer, StandardScaler

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(_PROJECT_ROOT, "DATA", "data.csv")

# Features removidas por multicolinearidade no notebook (correlação > 0.90)
DROPPED_FEATURES = {
    "area_mean", "area_se", "area_worst",
    "concave_points_mean", "concavity_mean",
    "perimeter_mean", "perimeter_se",
    "radius_mean", "radius_worst", "texture_mean",
}


def load_dataset(path: str = DATA_PATH) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """Carrega CSV, codifica alvo e retorna features selecionadas."""
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    df = df.loc[:, ~df.columns.str.startswith("unnamed")]

    le = LabelEncoder()
    df["diagnosis_enc"] = le.fit_transform(df["diagnosis"])  # B=0, M=1

    X_num = df.drop(columns=["diagnosis", "diagnosis_enc", "id"], errors="ignore")
    features = [c for c in X_num.columns if c not in DROPPED_FEATURES]

    X = df[features]
    y = df["diagnosis_enc"].values
    return X, y, features


def build_preprocessor() -> Pipeline:
    """Pipeline idêntico ao notebook: Imputer → PowerTransformer → StandardScaler."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("transformer", PowerTransformer(method="yeo-johnson")),
        ("scaler", StandardScaler()),
    ])


def prepare_data(
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    """
    Prepara conjuntos de treino/teste com pré-processamento.
    Retorna dict com X_train, X_test, y_train, y_test, preprocessor, features.
    """
    X, y, features = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    preprocessor = build_preprocessor()
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "X_train_proc": X_train_proc,
        "X_test_proc": X_test_proc,
        "preprocessor": preprocessor,
        "features": features,
    }
