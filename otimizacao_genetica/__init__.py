"""
Otimização de hiperparâmetros via Algoritmo Genético — Tech Challenge Fase 2, Etapa 1.
"""

from .config import GAConfig
from .genetic_algorithm import GeneticAlgorithm
from .experiments import run_all_experiments

__all__ = ["GeneticAlgorithm", "GAConfig", "run_all_experiments"]
