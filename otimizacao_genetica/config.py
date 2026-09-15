"""
Configurações do AG e espaços de busca de hiperparâmetros por modelo.
"""
from dataclasses import dataclass

# ── Espaços de hiperparâmetros (Módulo 1) ────────────────────────────────────
# Cada gene é um float em [0, 1] decodificado em encoding.py

MODEL_NAMES = [
    "Regressao Logistica",
    "Random Forest",
    "Gradient Boosting",
    "KNN",
    "SVM",
]

# Campeão da Fase 1 (analise_cancer.ipynb) — único modelo otimizado pelo AG.
MODELO_FASE1 = "Regressao Logistica"
MODELO_FOCO = MODELO_FASE1

# Hiperparâmetros padrão usados no notebook analise_cancer.ipynb (baseline)
BASELINE_HYPERPARAMS = {
    "Regressao Logistica": {
        "C": 1.0,
        "penalty": "l2",
        "solver": "lbfgs",
        "class_weight": None,
        "max_iter": 1000,
    },
    "Random Forest": {
        "n_estimators": 100,
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
    },
    "Gradient Boosting": {
        "n_estimators": 100,
        "learning_rate": 0.1,
        "max_depth": 3,
        "min_samples_split": 2,
        "subsample": 1.0,
    },
    "KNN": {
        "n_neighbors": 5,
        "weights": "uniform",
        "metric": "minkowski",
        "p": 2,
    },
    "SVM": {
        "C": 1.0,
        "kernel": "rbf",
        "gamma": "scale",
        "class_weight": None,
        "probability": True,
    },
}


@dataclass(frozen=True)
class GAConfig:
    """Configuração de um experimento com algoritmo genético."""
    name: str
    population_size: int
    generations: int
    mutation_rate: float
    crossover_rate: float
    tournament_size: int = 3
    elitism: int = 2
    random_state: int = 42
    # Anti-estagnação: indivíduos aleatórios injetados por geração
    immigrants: int = 2
    # Para se o melhor fitness não melhorar por N gerações (0 = desligado)
    patience: int = 8
    # Mutação gaussiana padrão (genes B/C)
    mutation_sigma: float = 0.12
    # Mutação bruta nos genes A e D (índices 0 e 3)
    strong_mutation_sigma: float = 0.50
    # Chance de reset total do gene (salto bruta) quando gene A/D muta
    strong_reset_prob: float = 0.40


# Genes A (índice 0) e D (índice 3) — mutação mais agressiva (C e solver na LR)
STRONG_MUTATION_GENES = (0, 3)

# Com GPU disponível: sobe para 100 gerações e mais paciência (espaço pra melhorar)
GPU_GENERATIONS = 100
GPU_PATIENCE = 25
GPU_IMMIGRANTS = 4


# Pelo menos 3 experimentos com configurações distintas (requisito Fase 2)
EXPERIMENT_CONFIGS = [
    GAConfig(
        name="exp1_pop30_mut_bruta",
        population_size=30,
        generations=40,
        mutation_rate=0.25,
        crossover_rate=0.70,
        tournament_size=3,
        immigrants=3,
        patience=12,
        mutation_sigma=0.12,
        strong_mutation_sigma=0.50,
        strong_reset_prob=0.40,
    ),
    GAConfig(
        name="exp2_pop50_mut_forte",
        population_size=50,
        generations=50,
        mutation_rate=0.30,
        crossover_rate=0.80,
        tournament_size=5,
        immigrants=4,
        patience=15,
        mutation_sigma=0.15,
        strong_mutation_sigma=0.55,
        strong_reset_prob=0.45,
    ),
    GAConfig(
        name="exp3_pop20_mut_explor",
        population_size=20,
        generations=60,
        mutation_rate=0.20,
        crossover_rate=0.60,
        tournament_size=3,
        immigrants=3,
        patience=18,
        mutation_sigma=0.10,
        strong_mutation_sigma=0.45,
        strong_reset_prob=0.35,
    ),
]

# Pesos da função fitness (prioriza recall — contexto clínico do Módulo 1)
FITNESS_WEIGHTS = {
    "recall": 0.35,
    "f1": 0.30,
    "accuracy": 0.20,
    "roc_auc": 0.15,
}
