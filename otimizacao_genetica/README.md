# Etapa 1 — Otimização via Algoritmos Genéticos

Módulo da **Fase 2** do Tech Challenge FIAP. Otimiza hiperparâmetros dos 5 modelos de diagnóstico de câncer de mama desenvolvidos no Módulo 1.

## Requisitos atendidos

| Requisito | Implementação |
|---|---|
| Codificação de genes | `encoding.py` — vetor `[0,1]` por hiperparâmetro |
| Seleção | Torneio (`genetic_algorithm.py`) |
| Cruzamento | Uniforme entre pais |
| Mutação | Gaussiana com clipping em `[0, 1]` |
| Fitness | `0.35·Recall + 0.30·F1 + 0.20·Acc + 0.15·AUC` (CV 5-fold) |
| Comparação baseline | Hiperparâmetros originais do `analise_cancer.ipynb` |
| 3 experimentos | `exp1_pop30_mut01`, `exp2_pop50_mut02`, `exp3_pop20_mut005` |

## Modelos otimizados

1. Regressão Logística
2. Random Forest
3. Gradient Boosting
4. KNN
5. SVM

## Como executar

### Notebook (recomendado para apresentação)

Abra `Analise/otimizacao_genetica.ipynb` — mesmo estilo explicativo do `analise_cancer.ipynb`, com seção dedicada à **comparação modelo anterior vs. otimizado**.

```bash
jupyter notebook Analise/otimizacao_genetica.ipynb
```

### Script (execução em lote)

```bash
# Na raiz do projeto (com venv ativo)
python -m otimizacao_genetica.run_experiments
```

### Aceleração GPU (opcional)

```bash
python -m pip install -r requirements-gpu.txt
# Para PyTorch com CUDA (Regressão Logística na GPU):
# python -m pip install torch --index-url https://download.pytorch.org/whl/cu124
```

| Modelo | Aceleração |
|---|---|
| Gradient Boosting | XGBoost CUDA (automático se GPU detectada) |
| Regressão Logística | PyTorch CUDA (se instalado com suporte CUDA) |
| RF, KNN, SVM | CPU paralelo (joblib, todos os núcleos) |

Desativar GPU: `set OTIMIZACAO_USE_GPU=0` (Windows) antes de rodar.

# Teste rápido (validação)
python -m otimizacao_genetica.run_experiments --quick

# Apenas um modelo
python -m otimizacao_genetica.run_experiments --model "Random Forest"
```

## Estrutura

```
otimizacao_genetica/
├── config.py              # 3 configs de experimento + baseline
├── data.py                # Pipeline de dados (Módulo 1)
├── encoding.py            # Cromossomo ↔ hiperparâmetros
├── fitness.py             # Avaliação e métricas
├── genetic_algorithm.py   # AG: seleção, cruzamento, mutação
├── experiments.py         # Runner e relatórios
└── run_experiments.py     # CLI
```

## Saídas

Resultados em `resultados/otimizacao_genetica/`:

- `exp1_pop30_mut01/resultados.json` — métricas detalhadas
- `exp1_pop30_mut01/comparacao_baseline_vs_otimizado.csv`
- `exp1_pop30_mut01/evolucao_*.png` — curvas de fitness
- `relatorio_consolidado.csv` — resumo dos 3 experimentos

## Codificação genética (exemplo — Random Forest)

| Gene | Hiperparâmetro | Intervalo |
|---|---|---|
| g0 | n_estimators | 50 – 300 |
| g1 | max_depth | 2 – 30 (ou None) |
| g2 | min_samples_split | 2 – 20 |
| g3 | min_samples_leaf | 1 – 10 |
| g4 | max_features | sqrt, log2, 0.5, 0.8 |

## Configurações dos 3 experimentos

| Experimento | População | Gerações | Mutação | Cruzamento |
|---|---|---|---|---|
| exp1_pop30_mut01 | 30 | 25 | 0.10 | 0.70 |
| exp2_pop50_mut02 | 50 | 30 | 0.20 | 0.80 |
| exp3_pop20_mut005 | 20 | 40 | 0.05 | 0.60 |
