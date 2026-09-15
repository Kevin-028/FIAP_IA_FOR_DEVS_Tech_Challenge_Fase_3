"""
Algoritmo Genético para otimização de hiperparâmetros.
Avaliação em lote: GPU (sequencial na placa) ou CPU (paralelo joblib).
"""
from dataclasses import dataclass, field, replace

import numpy as np

from .config import (
    GAConfig,
    GPU_GENERATIONS,
    GPU_IMMIGRANTS,
    GPU_PATIENCE,
    STRONG_MUTATION_GENES,
)
from .device import detect_devices
from .encoding import get_gene_count, random_chromosome
from .fitness import evaluate_chromosome, evaluate_chromosomes_batch, uses_gpu


def _gpu_acelerada() -> bool:
    """True se há aceleração por GPU (XGBoost CUDA ou PyTorch CUDA)."""
    info = detect_devices()
    return bool(info.get("use_gpu") and (info.get("xgboost_gpu") or info.get("torch_cuda")))


def scale_config_for_hardware(config: GAConfig) -> GAConfig:
    """Com GPU: até 100 gerações e mais paciência para o AG explorar de verdade."""
    if not _gpu_acelerada():
        return config
    return replace(
        config,
        generations=max(config.generations, GPU_GENERATIONS),
        patience=max(config.patience, GPU_PATIENCE),
        immigrants=max(config.immigrants, GPU_IMMIGRANTS),
    )


@dataclass
class Individual:
    genes: np.ndarray
    fitness: float = 0.0
    metrics: dict = field(default_factory=dict)
    hyperparams: dict = field(default_factory=dict)


@dataclass
class GAResult:
    best_individual: Individual
    history: list[dict]
    config: GAConfig
    model_name: str


class GeneticAlgorithm:
    """Otimizador genético de hiperparâmetros para um modelo sklearn/GPU."""

    def __init__(
        self,
        model_name: str,
        config: GAConfig,
        X_train: np.ndarray,
        y_train: np.ndarray,
        cv: int = 5,
    ):
        self.model_name = model_name
        self.config = scale_config_for_hardware(config)
        self.X_train = X_train
        self.y_train = y_train
        self.cv = cv
        self.n_genes = get_gene_count(model_name)
        self.rng = np.random.default_rng(self.config.random_state)
        self._on_gpu = uses_gpu(model_name)
        self._gpu_hw = _gpu_acelerada()

    def _evaluate(self, genes: np.ndarray) -> Individual:
        metrics, hyperparams = evaluate_chromosome(
            self.model_name, genes, self.X_train, self.y_train, cv=self.cv
        )
        return Individual(
            genes=genes.copy(),
            fitness=metrics["fitness"],
            metrics=metrics,
            hyperparams=hyperparams,
        )

    def _evaluate_batch(self, genes_list: list[np.ndarray]) -> list[Individual]:
        """Avalia vários indivíduos — paralelo (CPU) ou GPU."""
        if not genes_list:
            return []
        results = evaluate_chromosomes_batch(
            self.model_name, genes_list, self.X_train, self.y_train, cv=self.cv
        )
        return [
            Individual(
                genes=genes.copy(),
                fitness=m["fitness"],
                metrics=m,
                hyperparams=hp,
            )
            for genes, (m, hp) in zip(genes_list, results)
        ]

    def _tournament_selection(self, population: list[Individual]) -> Individual:
        k = min(self.config.tournament_size, len(population))
        contenders = self.rng.choice(population, size=k, replace=False)
        return max(contenders, key=lambda ind: ind.fitness)

    def _uniform_crossover(
        self, parent_a: Individual, parent_b: Individual
    ) -> tuple[np.ndarray, np.ndarray]:
        mask = self.rng.random(self.n_genes) < 0.5
        child1 = np.where(mask, parent_a.genes, parent_b.genes)
        child2 = np.where(mask, parent_b.genes, parent_a.genes)
        return child1, child2

    def _mutate(self, genes: np.ndarray) -> np.ndarray:
        """
        Mutação gaussiana nos genes.
        Genes A (0) e D (3): mutação bruta — sigma alto ou reset total do gene.
        Demais genes: perturbações menores (exploração fina).
        """
        mutated = genes.copy()
        cfg = self.config
        for i in range(self.n_genes):
            if self.rng.random() >= cfg.mutation_rate:
                continue
            if i in STRONG_MUTATION_GENES:
                # Salto bruto: redesenha o gene ou aplica ruído forte
                if self.rng.random() < cfg.strong_reset_prob:
                    mutated[i] = self.rng.uniform(0.0, 1.0)
                else:
                    mutated[i] += self.rng.normal(0, cfg.strong_mutation_sigma)
            else:
                mutated[i] += self.rng.normal(0, cfg.mutation_sigma)
        return np.clip(mutated, 0.0, 1.0)

    def _initialize_population(self) -> list[Individual]:
        genes_list = [
            random_chromosome(self.model_name, self.rng)
            for _ in range(self.config.population_size)
        ]
        return self._evaluate_batch(genes_list)

    def run(self, verbose: bool = True) -> GAResult:
        if verbose:
            mode = "GPU" if self._on_gpu else "CPU paralelo"
            print(f"  Modo avaliacao: {mode}")
            if self._gpu_hw:
                print(
                    f"  GPU detectada -> {self.config.generations} geracoes "
                    f"(patience={self.config.patience}, "
                    f"mutacao bruta nos genes A/D)"
                )
            else:
                print(
                    f"  geracoes={self.config.generations} | "
                    f"patience={self.config.patience} | "
                    f"mutacao bruta nos genes A/D"
                )

        population = self._initialize_population()
        history: list[dict] = []
        best = max(population, key=lambda ind: ind.fitness)
        stagnant = 0

        for gen in range(self.config.generations):
            population.sort(key=lambda ind: ind.fitness, reverse=True)
            new_population = population[: self.config.elitism]

            # Imigrantes: indivíduos 100% aleatórios para manter diversidade
            n_immigrants = min(
                self.config.immigrants,
                max(0, self.config.population_size - len(new_population)),
            )
            n_children = (
                self.config.population_size - len(new_population) - n_immigrants
            )

            # Gera filhos e avalia em lote (mais rapido)
            pending_genes: list[np.ndarray] = []
            while len(pending_genes) < n_children:
                parent_a = self._tournament_selection(population)
                parent_b = self._tournament_selection(population)

                if self.rng.random() < self.config.crossover_rate:
                    child_a, child_b = self._uniform_crossover(parent_a, parent_b)
                else:
                    child_a, child_b = parent_a.genes.copy(), parent_b.genes.copy()

                pending_genes.append(self._mutate(child_a))
                if len(pending_genes) < n_children:
                    pending_genes.append(self._mutate(child_b))

            pending_genes.extend(
                random_chromosome(self.model_name, self.rng)
                for _ in range(n_immigrants)
            )

            new_population.extend(self._evaluate_batch(pending_genes))
            population = new_population

            gen_best = max(population, key=lambda ind: ind.fitness)
            if gen_best.fitness > best.fitness:
                best = gen_best
                stagnant = 0
            else:
                stagnant += 1

            record = {
                "generation": gen + 1,
                "best_fitness": gen_best.fitness,
                "mean_fitness": float(np.mean([ind.fitness for ind in population])),
                "best_recall": gen_best.metrics.get("recall", 0),
                "best_f1": gen_best.metrics.get("f1", 0),
            }
            history.append(record)

            if verbose:
                print(
                    f"  Gen {gen + 1:3d}/{self.config.generations} | "
                    f"fitness={gen_best.fitness:.4f} | "
                    f"recall={gen_best.metrics['recall']:.4f} | "
                    f"f1={gen_best.metrics['f1']:.4f}"
                )

            if self.config.patience and stagnant >= self.config.patience:
                if verbose:
                    print(
                        f"  Parada antecipada: sem melhora ha "
                        f"{stagnant} geracoes (gen {gen + 1})"
                    )
                break

        return GAResult(
            best_individual=best,
            history=history,
            config=self.config,
            model_name=self.model_name,
        )
