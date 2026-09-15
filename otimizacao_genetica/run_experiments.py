#!/usr/bin/env python
"""
Entry point — executa os 3 experimentos de otimização genética.

Uso:
    python -m otimizacao_genetica.run_experiments
    python -m otimizacao_genetica.run_experiments --quick   # teste rápido
"""
import argparse

from otimizacao_genetica.config import GAConfig
from otimizacao_genetica.device import print_device_info
from otimizacao_genetica.experiments import run_all_experiments


def main():
    parser = argparse.ArgumentParser(
        description="Otimização de hiperparâmetros via Algoritmo Genético (Fase 2)"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Modo rápido para validação (população e gerações reduzidas)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Otimizar apenas um modelo (ex: 'Random Forest')",
    )
    args = parser.parse_args()

    if args.quick:
        experiments = [
            GAConfig("quick_exp1", 10, 5, 0.10, 0.70),
            GAConfig("quick_exp2", 15, 5, 0.20, 0.80),
            GAConfig("quick_exp3", 8, 8, 0.05, 0.60),
        ]
    else:
        experiments = None  # usa EXPERIMENT_CONFIGS padrão

    models = [args.model] if args.model else None

    print_device_info()
    run_all_experiments(experiments=experiments, models=models, verbose=True)


if __name__ == "__main__":
    main()
