# Arquitetura — Tech Challenge Fase 2

## Visão geral

A Fase 2 estende a aplicação web Flask do Módulo 1 com:

1. **Módulo 2 (AG)** — predição com modelo KNN otimizado por Algoritmo Genético
2. **Comparação** — execução lado a lado dos dois modelos no mesmo input
3. **Assistente LLM** — interpretação em linguagem natural + chat (Ollama local)
4. **Monitoramento** — logging estruturado, métricas e auto-scaling simulado
5. **Identidade visual** — Fase 1 (vermelho) vs Fase 2 (roxo) para comparação clara

## Diagrama de componentes

```mermaid
flowchart TB
    Browser["Navegador (Jinja2 + Bootstrap + JS)"]

    subgraph Flask["Flask App — web/app.py"]
        MW["Middleware de métricas<br/>(before/after_request)"]
        subgraph Controllers["Controllers (Blueprints)"]
            C1["prediction_controller<br/>/fase1"]
            C2["fase2_controller<br/>/fase2"]
            C3["comparar_controller<br/>/comparar"]
            C4["llm_controller<br/>/llm, /api/llm"]
            C5["ops_controller<br/>/ops, /api"]
            C6["visao_controller<br/>/visao"]
        end
        subgraph Models["Models"]
            M1["cancer_model (Fase 1)"]
            M2["cancer_model_fase2 (Fase 2)"]
            ML["model_loader<br/>(lazy load + cache)"]
        end
        subgraph LLM["Pacote llm/"]
            L1["interpreter"]
            L2["client (Ollama | fallback)"]
            L3["prompts / context"]
            L4["feedback (auditoria)"]
        end
        subgraph Mon["monitoring/"]
            ML1["logger (JSON)"]
            ML2["metrics + auto-scale"]
        end
    end

    Ollama["Ollama local<br/>http://localhost:11434"]
    PKL[("modelo_otimizado_ag.pkl<br/>+ DATA/data.csv")]
    Logs[("web/logs/*.jsonl / app.log")]

    Browser -->|HTTP| MW --> Controllers
    C1 --> M1
    C2 --> M2
    C3 --> M2
    M1 --> ML
    M2 --> ML
    ML --> PKL
    C2 --> L1
    C4 --> L1
    L1 --> L2 --> Ollama
    L1 --> L3
    C4 --> L4 --> Logs
    Controllers --> Mon
    ML1 --> Logs
```

## Fluxo de predição + interpretação (Módulo 2)

```mermaid
sequenceDiagram
    participant U as Médico (navegador)
    participant F as Flask (/fase2/predict)
    participant M as cancer_model_fase2
    participant Loader as model_loader
    participant JS as llm_interpretation.js
    participant API as /api/llm/interpret
    participant I as interpreter
    participant Cli as OllamaClient / Fallback

    U->>F: POST formulário (features)
    F->>Loader: load_fase2() (cache)
    Loader-->>F: modelo + preprocessor
    F->>M: predict(form)
    M-->>F: diagnóstico + probabilidade + métricas
    F-->>U: página de resultado + assistente (drawer)
    U->>JS: abre o assistente
    JS->>API: POST {result, feature_highlights}
    API->>I: interpret(payload)
    I->>Cli: complete(system, user)
    alt Ollama disponível
        Cli-->>I: texto em linguagem natural
    else Ollama indisponível
        Cli-->>I: fallback por regras clínicas
    end
    I-->>JS: {text, provider, sections}
    JS-->>U: interpretação renderizada + 👍/👎
```

## Rotas

| Rota | Módulo | Descrição |
|------|--------|-----------|
| `/` | Hub | Página inicial com links F1, F2, Comparar |
| `/fase1`, `/fase1/predict` | Módulo 1 | Modelo baseline (hiperparâmetros fixos) |
| `/fase1/exemplos` | Módulo 1 | Casos reais rotulados |
| `/fase2`, `/fase2/predict` | Módulo 2 | Modelo otimizado por AG + interpretação LLM |
| `/comparar`, `/comparar/predict` | Comparação | Ambos os modelos no mesmo exame |
| `/llm/avaliacao` | LLM | Rubrica heurística de qualidade |
| `/llm/auditoria` | LLM | Curadoria dos feedbacks dos médicos |
| `/api/llm/interpret`, `/api/llm/chat`, `/api/llm/feedback`, `/api/llm/status` | LLM | Endpoints JSON |
| `/ops/monitoramento` | Ops | Dashboard de métricas |
| `/api/health`, `/api/metrics` | Ops | Health check e snapshot JSON |
| `/visao/*` | Visão | Gestos (inalterado) |

## Algoritmo Genético (Módulo 2)

```mermaid
flowchart LR
    Init["População inicial<br/>(genes aleatórios [0,1])"] --> Eval["Avaliação por CV<br/>fitness = 0.35·recall + 0.30·f1<br/>+ 0.20·acc + 0.15·roc_auc"]
    Eval --> Sel["Seleção por torneio"]
    Sel --> Cross["Crossover uniforme"]
    Cross --> Mut["Mutação gaussiana"]
    Mut --> Elit["Elitismo (mantém melhores)"]
    Elit --> Eval
    Elit --> Best["Melhor indivíduo<br/>→ save_web_model.py → .pkl"]
```

- **Codificação** (`encoding.py`): vetor de floats `[0,1]`, um gene por hiperparâmetro; decodificação por interpolação linear/logarítmica e escolha categórica.
- **Aptidão** (`fitness.py`): validação cruzada, priorizando **recall** (evitar falsos negativos em rastreamento).
- **Configurações** (`config.py`): 3 experimentos com população/mutação/gerações distintas.

## Decisões de implementação

### Separação Fase 1 vs Fase 2

- **Fase 1** mantém o fluxo original com prefixo `/fase1` e tema vermelho FIAP.
- **Fase 2** usa hero roxo, badge `MÓDULO 2 — ALGORITMO GENÉTICO` e a interpretação LLM.
- A página **Comparar** exibe resultados em duas colunas com bordas coloridas.

### Carregamento de modelos

`model_loader.py` usa cache thread-safe:

- Fase 1 treina **KNN baseline** com `BASELINE_HYPERPARAMS["KNN"]` (ou carrega `.pkl` se existir).
- Fase 2 carrega `resultados/otimizacao_genetica/notebook/modelo_otimizado_ag.pkl`.
- Regenerar Fase 2: `python -m otimizacao_genetica.save_web_model`

### Assistente LLM

- Roda **localmente via Ollama** (`FIAP_LLM_PROVIDER=ollama`); ver [`ARQUITETURA_LLM.md`](ARQUITETURA_LLM.md).
- Degrada com elegância para **interpretação por regras** se o Ollama estiver indisponível.
- Feedback dos médicos (👍/👎 + motivo + tags) alimenta a página de **auditoria/curadoria**.

### Monitoramento e logging

- **Logs**: JSON em `web/logs/app.log` via `monitoring/logger.py`.
- **Métricas**: contadores em memória (`request_count`, `prediction_count`, latência média/p95, taxa de erro).
- Middleware em `app.py` mede a latência de cada requisição.

### Auto-scaling (simulado)

`MetricsStore` ajusta `workers` conforme `active_requests`:

| Variável de ambiente | Padrão | Função |
|---------------------|--------|--------|
| `FIAP_MIN_WORKERS` | 2 | Mínimo de workers |
| `FIAP_MAX_WORKERS` | 8 | Máximo de workers |
| `FIAP_SCALE_UP_RPS` | 5 | Escala para cima se requisições ativas ≥ limiar |
| `FIAP_SCALE_DOWN_RPS` | 2 | Escala para baixo se requisições ativas ≤ limiar |

> O auto-scale é **demonstrativo** — o valor de workers aparece no dashboard `/ops/monitoramento`. Em produção real, usar Kubernetes HPA ou similar com métricas externas.

## Testes automatizados

Suíte pytest em `tests/` (ver README, seção *Testes*). Os testes de LLM rodam offline (provedor `fallback`), garantindo determinismo e independência de rede.

```mermaid
flowchart LR
    subgraph tests
        E["test_encoding<br/>(AG)"]
        Me["test_metrics<br/>(monitoramento)"]
        L["test_llm<br/>(prompts/interpreter)"]
        Fb["test_feedback<br/>(auditoria)"]
        Mo["test_models<br/>(predição F1/F2)"]
        R["test_routes<br/>(Flask API/páginas)"]
    end
    conftest["conftest.py<br/>sys.path + LLM=fallback"] --> tests
```

## Como executar localmente

```powershell
cd Fase_2
.venv\Scripts\activate   # Windows
python web/app.py
```

Acesse: http://localhost:5000

## Comparação de modelos

A tabela em `/comparar` lê `resultados/otimizacao_genetica/notebook/comparacao_anterior_vs_otimizado.csv` (gerada pelo notebook).

Na predição, `predict_compare()` executa Fase 1 e Fase 2 e retorna concordância/divergência de diagnóstico e diferença de probabilidade.
