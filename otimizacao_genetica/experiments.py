"""
Execução dos experimentos e comparação modelos otimizados vs. originais.
"""
import json
import os
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd

from .config import EXPERIMENT_CONFIGS, MODEL_NAMES
from .data import prepare_data
from .encoding import describe_encoding
from .fitness import evaluate_baseline, evaluate_on_test
from .genetic_algorithm import GeneticAlgorithm

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(_PROJECT_ROOT, "resultados", "otimizacao_genetica")


def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def run_single_experiment(
    exp_config,
    data: dict,
    models: list[str] | None = None,
    verbose: bool = True,
) -> dict:
    """Executa um experimento AG completo para todos os modelos."""
    models = models or MODEL_NAMES
    exp_dir = _ensure_dir(os.path.join(RESULTS_DIR, exp_config.name))
    results = {
        "experiment": exp_config.name,
        "config": {
            "population_size": exp_config.population_size,
            "generations": exp_config.generations,
            "mutation_rate": exp_config.mutation_rate,
            "crossover_rate": exp_config.crossover_rate,
            "tournament_size": exp_config.tournament_size,
            "elitism": exp_config.elitism,
        },
        "models": {},
        "timestamp": datetime.now().isoformat(),
    }

    if verbose:
        print(f"\n{'='*60}")
        print(f"Experimento: {exp_config.name}")
        print(
            f"  pop={exp_config.population_size} | gen={exp_config.generations} | "
            f"mut={exp_config.mutation_rate} | cross={exp_config.crossover_rate}"
        )
        print(f"{'='*60}")

    for model_name in models:
        if verbose:
            print(f"\n>> Otimizando: {model_name}")

        ga = GeneticAlgorithm(
            model_name=model_name,
            config=exp_config,
            X_train=data["X_train_proc"],
            y_train=data["y_train"],
        )
        ga_result = ga.run(verbose=verbose)

        best = ga_result.best_individual
        test_metrics = evaluate_on_test(
            model_name,
            best.hyperparams,
            data["X_train_proc"],
            data["y_train"],
            data["X_test_proc"],
            data["y_test"],
        )

        baseline = evaluate_baseline(
            model_name,
            data["X_train_proc"],
            data["y_train"],
            data["X_test_proc"],
            data["y_test"],
        )

        improvement = {
            metric: round(test_metrics[metric] - baseline["test"][metric], 4)
            for metric in ["accuracy", "recall", "f1", "roc_auc", "fitness"]
        }

        model_result = {
            "encoding": describe_encoding(model_name),
            "best_hyperparams": _serialize(best.hyperparams),
            "cv_metrics": {k: round(v, 4) for k, v in best.metrics.items()},
            "test_metrics": {k: round(v, 4) for k, v in test_metrics.items()},
            "baseline": {
                "hyperparams": _serialize(baseline["hyperparams"]),
                "cv_metrics": {k: round(v, 4) for k, v in baseline["cv"].items()},
                "test_metrics": {k: round(v, 4) for k, v in baseline["test"].items()},
            },
            "improvement_vs_baseline": improvement,
            "evolution_history": ga_result.history,
        }
        results["models"][model_name] = model_result

        if verbose:
            print(
                f"  Baseline test F1={baseline['test']['f1']:.4f} | "
                f"Otimizado test F1={test_metrics['f1']:.4f} | "
                f"delta={improvement['f1']:+.4f}"
            )

        _plot_evolution(ga_result.history, exp_dir, model_name)

    out_path = os.path.join(exp_dir, "resultados.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    _save_comparison_table(results, exp_dir)
    return results


def run_all_experiments(
    experiments=None,
    models: list[str] | None = None,
    verbose: bool = True,
) -> dict:
    """Executa os 3 experimentos e gera relatório consolidado."""
    experiments = experiments or EXPERIMENT_CONFIGS
    data = prepare_data()

    if verbose:
        print("Dados carregados:")
        print(f"  Features: {len(data['features'])}")
        print(f"  Treino: {len(data['y_train'])} | Teste: {len(data['y_test'])}")

    all_results = {}
    for exp_config in experiments:
        all_results[exp_config.name] = run_single_experiment(
            exp_config, data, models=models, verbose=verbose
        )

    _save_consolidated_report(all_results)
    return all_results


def _serialize(obj):
    """Converte valores não-JSON-serializáveis."""
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    return str(obj)


def _plot_evolution(history: list[dict], exp_dir: str, model_name: str):
    """Gráfico de evolução do fitness ao longo das gerações."""
    df = pd.DataFrame(history)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df["generation"], df["best_fitness"], label="Melhor fitness", lw=2)
    ax.plot(df["generation"], df["mean_fitness"], label="Fitness médio", lw=1, alpha=0.7)
    ax.set_xlabel("Geração")
    ax.set_ylabel("Fitness")
    ax.set_title(f"Evolução AG — {model_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    safe_name = model_name.replace(" ", "_").lower()
    plt.savefig(os.path.join(exp_dir, f"evolucao_{safe_name}.png"), dpi=120)
    plt.close()


def _save_comparison_table(results: dict, exp_dir: str):
    """Tabela comparativa baseline vs. otimizado para um experimento."""
    rows = []
    for model_name, model_data in results["models"].items():
        base = model_data["baseline"]["test_metrics"]
        opt = model_data["test_metrics"]
        delta = model_data["improvement_vs_baseline"]
        rows.append({
            "Modelo": model_name,
            "Baseline_Acc": base["accuracy"],
            "Otimizado_Acc": opt["accuracy"],
            "Delta_Acc": delta["accuracy"],
            "Baseline_Recall": base["recall"],
            "Otimizado_Recall": opt["recall"],
            "Delta_Recall": delta["recall"],
            "Baseline_F1": base["f1"],
            "Otimizado_F1": opt["f1"],
            "Delta_F1": delta["f1"],
            "Baseline_AUC": base["roc_auc"],
            "Otimizado_AUC": opt["roc_auc"],
            "Delta_AUC": delta["roc_auc"],
        })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(exp_dir, "comparacao_baseline_vs_otimizado.csv"), index=False)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    metrics = ["recall", "f1", "roc_auc"]
    titles = ["Recall", "F1-Score", "ROC-AUC"]
    model_list = list(results["models"].keys())
    x = range(len(model_list))
    width = 0.35

    for ax, metric, title in zip(axes, metrics, titles):
        b_vals = [results["models"][m]["baseline"]["test_metrics"][metric] for m in model_list]
        o_vals = [results["models"][m]["test_metrics"][metric] for m in model_list]
        ax.bar([i - width / 2 for i in x], b_vals, width, label="Baseline", alpha=0.8)
        ax.bar([i + width / 2 for i in x], o_vals, width, label="Otimizado (AG)", alpha=0.8)
        ax.set_xticks(list(x))
        ax.set_xticklabels([m.replace(" ", "\n") for m in model_list], fontsize=8)
        ax.set_title(title)
        ax.set_ylim(0.7, 1.05)
        ax.legend(fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)

    plt.suptitle(f"Comparação Baseline vs. AG — {results['experiment']}")
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, "comparacao_grafico.png"), dpi=120)
    plt.close()


def _save_consolidated_report(all_results: dict):
    """Relatório consolidado dos 3 experimentos."""
    report_dir = _ensure_dir(RESULTS_DIR)
    rows = []

    for exp_name, exp_data in all_results.items():
        for model_name, model_data in exp_data["models"].items():
            rows.append({
                "Experimento": exp_name,
                "Modelo": model_name,
                "Pop": exp_data["config"]["population_size"],
                "Geracoes": exp_data["config"]["generations"],
                "Mutacao": exp_data["config"]["mutation_rate"],
                "Cruzamento": exp_data["config"]["crossover_rate"],
                "Baseline_F1": model_data["baseline"]["test_metrics"]["f1"],
                "Otimizado_F1": model_data["test_metrics"]["f1"],
                "Delta_F1": model_data["improvement_vs_baseline"]["f1"],
                "Baseline_Recall": model_data["baseline"]["test_metrics"]["recall"],
                "Otimizado_Recall": model_data["test_metrics"]["recall"],
                "Delta_Recall": model_data["improvement_vs_baseline"]["recall"],
                "Melhor_Fitness_CV": model_data["cv_metrics"]["fitness"],
            })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(report_dir, "relatorio_consolidado.csv"), index=False)

    with open(os.path.join(report_dir, "relatorio_consolidado.json"), "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)

    print(f"\nRelatorios salvos em: {report_dir}")
