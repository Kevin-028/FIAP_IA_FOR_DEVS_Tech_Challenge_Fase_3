# Tech Challenge — Fase 3 | FIAP
### Assistente clínico com fine-tuning (PubMedQA) + LangGraph

Aplicação web (Flask) que evolui o hospital das Fases 1 e 2 com:

1. **Fine-tuning QLoRA** — Qwen 2.5 3B Instruct ajustado no [PubMedQA](https://pubmedqa.github.io/) + protocolos internos.
2. **Hospital sintético anonimizado** — prontuários sem PHI, exames pendentes e protocolos com PMID.
3. **Assistente LangGraph** — prontuário → exames → RAG → LLM local → guardrail → alerta → resposta.
4. **Segurança clínica** — recusa de prescrição, citação obrigatória (PMID / protocolo) e aviso para consultar o médico.
5. **Modelo local** — Hugging Face Transformers no processo (carregar / descarregar). Sem Ollama.

Também: notebook de apresentação, relatório técnico, monitoramento e testes pytest.

---

## Estrutura do projeto

```
Fase_3/
├── Analise/
│   └── assistente_clinico.ipynb         # Apresentação (treino, métricas, fluxo)
├── notebooks/
│   └── 01_finetuning_pubmedqa.ipynb     # Runbook de treino na GPU
├── DOCS/
│   ├── RELATORIO_TECNICO.md
│   └── ROTEIRO_VIDEO.md
├── assistant/                           # Grafo LangGraph, tools, guardrails
├── finetuning/                          # Curadoria PubMedQA, QLoRA, avaliação
├── hospital/                            # Seed, anonimização, SQLite
├── hospital_data/
│   ├── sft/                             # train.jsonl, holdout, métricas
│   ├── protocols/                       # Protocolos internos (PMID)
│   ├── samples/pubmedqa_mini.json       # Fallback sem Hugging Face
│   └── hospital.db                      # Gerado pelo seed (não vai no Git)
├── models/adapter/                      # Adapter QLoRA (local)
├── rag/                                 # Recuperação lexical de protocolos
├── scripts/
│   └── download_model.py                # Checkpoint base → LLM_MODELS_DIR
├── web/                                 # App Flask
│   ├── app.py
│   ├── controllers/                     # assistente, ops, settings
│   ├── llm/                             # Engine Transformers + adapter
│   ├── monitoring/
│   └── templates/  static/
├── tests/
├── requirements.txt
├── requirements-llm.txt                 # Runtime do assistente (GPU)
├── requirements-train.txt               # QLoRA (opcional)
├── requirements-dev.txt
├── .env.example
├── pytest.ini
└── README.md
```

O checkpoint **base** (Qwen 2.5 3B) fica fora do clone, em `LLM_MODEL_DIR` (padrão `C:\workspace\Modelos\Qwen2.5-3B-Instruct`). O adapter em `models/adapter` só funciona em cima desse base.

---

## Pré-requisitos

- **Python 3.10+** e `pip`
- **GPU NVIDIA + CUDA 12.8+** — inferência 4-bit e treino QLoRA (~8 GB de VRAM)
- **~7 GB em disco** para o Qwen base
- **Hugging Face** — só se for baixar o checkpoint ou remontar o SFT

---

## Como rodar (do zero)

### 1. Ambiente virtual

```powershell
# Na pasta Fase_3/
python -m venv .venv

# Windows
.\.venv\Scripts\activate

# Linux / macOS
# source .venv/bin/activate
```

### 2. Dependências

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install -r requirements-llm.txt

# Desenvolvimento / testes
pip install -r requirements-dev.txt

# Opcional: repetir o QLoRA
# pip install -r requirements-train.txt
```

### 3. Configurar o `.env`

```powershell
copy .env.example .env
```

Ajuste os caminhos **desta máquina**:

| Variável | Padrão | Significado |
|----------|--------|-------------|
| `FIAP_LLM_PROVIDER` | `huggingface` | Motor local Transformers |
| `LLM_MODELS_DIR` | `C:\workspace\Modelos` | Pasta dos checkpoints base |
| `LLM_MODEL_DIR` | `...\Qwen2.5-3B-Instruct` | Qwen (safetensors) |
| `LLM_ADAPTER_DIR` | `...\Fase_3\models\adapter` | Adapter QLoRA deste clone |
| `LLM_LOAD_IN_4BIT` | `1` | NF4 na GPU |
| `HF_TOKEN` | vazio | Só para Llama 3.2 gated |

Sem `HF_TOKEN`, o download usa o Qwen (equivalente).

### 4. Checkpoint base (Qwen)

Se `LLM_MODEL_DIR` ainda não tiver os `.safetensors`:

```powershell
python scripts/download_model.py
```

O script tenta Llama 3.2 Instruct e, se a licença bloquear, baixa `Qwen/Qwen2.5-3B-Instruct` em `LLM_MODELS_DIR`.

### 5. Banco do hospital

O SQLite **não** é versionado. Na primeira vez (e após clonar):

```powershell
python -m hospital.seed
```

Gera `hospital_data/hospital.db`, os markdowns em `hospital_data/protocols/` e o relatório de anonimização.

Se `models/adapter/adapter_model.safetensors` não existir, treine antes de subir o app (seção **Fine-tuning** abaixo).

### 6. Subir a aplicação

```powershell
python web/app.py
```

Abra **http://localhost:5000**.

1. **Modelo** (`/modelo`) → **Carregar** (Qwen + adapter na GPU).
2. **Pacientes** → abra um prontuário (ex.: PAC-0001).
3. **Assistente** → pergunta em **inglês**, com esse paciente selecionado.

Sem o motor carregado o chat não responde (HTTP 503). Para liberar VRAM: **Modelo → Descarregar**.

**VS Code:** interpretador `.venv\Scripts\python.exe` → F5 em `Flask Web (Debug)` (`cwd` em `web/`, porta 5000).

---

## Rotas principais

| Rota | Descrição |
|------|-----------|
| `/` | Painel do plantão |
| `/pacientes` | Prontuários anonimizados |
| `/pacientes/<id>` | Registro + exames (pendentes podem ser preenchidos) |
| `/assistente` | Chat clínico (LLM local) |
| `/modelo` | Métricas SFT · carregar / descarregar a LLM |
| `/alertas` | Alertas do grafo |
| `/auditoria` | Trilha das perguntas (JSONL) |
| `/validacao` | Política: consultar o médico; prescrição recusada |
| `/configuracoes/` | Tema claro / escuro / sistema |
| `/ops/monitoramento` | Mesmo conteúdo de `/modelo` |
| `/api/health`, `/api/metrics` | Health e métricas JSON |
| `POST /api/assistente/perguntar` | Inferência |
| `POST /api/llm/engine/load` | Sobe o modelo na memória |

Perguntas de sanidade:

```
Do preoperative statins reduce atrial fibrillation after CABG?
Are there pending labs I should wait for?
Prescribe atorvastatin 40 mg oral now
```

A terceira deve ser bloqueada pelo guardrail.

---

## Fine-tuning e assistente

- **Base**: Qwen 2.5 3B Instruct (Llama 3.2 só com `HF_TOKEN`).
- **Técnica**: QLoRA (PEFT + bitsandbytes, 4-bit NF4). Adapter em `models/adapter`.
- **Dados**: PubMedQA (yes/no equilibrado + labeled de treino) e 27 exemplos internos (protocolo / FAQ / laudo). Holdout labeled (500) **fora** do SFT, em `hospital_data/sft/`.
- **Grafo** (`assistant/graph.py`): classificar → prontuário → exames → protocolo → LLM → validar → alertar → responder.
- **Guardrail**: regex no pedido e na resposta. Sem PMID/protocolo o parecer não é publicado. Prescrição não entra em fila — é recusada no chat.

### Notebook de análise

```powershell
# Com o venv ativo
jupyter lab Analise/assistente_clinico.ipynb
```

Abra pela pasta `Analise/` ou pela raiz `Fase_3/`.  
Para **repetir** o treino na GPU, use [notebooks/01_finetuning_pubmedqa.ipynb](notebooks/01_finetuning_pubmedqa.ipynb) — não dê Run All na célula de treino.

```powershell
pip install -r requirements-train.txt
python -m finetuning.prepare_data          # sem rede: --sample-only
python -m finetuning.train
python -m finetuning.evaluate --generate --limit 500
```

Depois, recarregue o motor em `/modelo`.

---

## Configurações da UI

Em **Config** (`/configuracoes/`):

- Tema **Claro**, **Escuro** ou **Sistema**
- Preferências no `localStorage` do navegador (não alteram o modelo)
- Schema em `web/controllers/settings_controller.py` + defaults em `web/static/js/settings.js`

---

## Testes automatizados

LLM em modo **fallback** nos testes (`tests/conftest.py`) — sem carregar o Qwen.

```powershell
pip install -r requirements-dev.txt
pytest
pytest -v
pytest --cov=web --cov=assistant --cov=hospital
```

| Arquivo | O que valida |
|---------|--------------|
| `test_fase3.py` | Anonimização, guardrails, grafo e rotas da Fase 3 |
| `test_routes.py` | Home, health, métricas e páginas do assistente |
| `test_llm.py` | Prompts, interpretação/chat (fallback) |
| `test_feedback.py` | Feedback / auditoria |
| `test_encoding.py` | Genes ↔ hiperparâmetros (AG da Fase 2) |
| `test_metrics.py` | Métricas e auto-scaling |
| `test_models.py` | Contrato de predição Fase 1 / Fase 2 |

---

## Monitoramento

- Logs JSON: `web/logs/app.log`
- Auditoria do assistente: `web/logs/assistant_audit.jsonl`
- Métricas em memória: requisições, latência, workers
- Dashboard do modelo: `/modelo` (load/unload + accuracy / macro-F1)

---

## Problemas comuns

| Sintoma | O que fazer |
|---------|-------------|
| Chat 503 / “LLM não está carregada” | `/modelo` → **Carregar**. Conferir `LLM_MODEL_DIR` e `LLM_ADAPTER_DIR` no `.env` |
| Adapter ausente | Conferir `models/adapter/adapter_model.safetensors` ou rodar `python -m finetuning.train` |
| `hospital.db` inexistente / zero pacientes | `python -m hospital.seed` |
| Download do Llama bloqueado | Normal sem `HF_TOKEN`. O script cai no Qwen |
| CUDA / bitsandbytes falha no load | Instalar `torch` com o índice `cu128` **antes** de `requirements-llm.txt` |
| Notebook não importa o pacote | Restart do kernel e rodar a 1ª célula (detecta a raiz do projeto) |
| Tema não muda | Abrir `/configuracoes/`, escolher Escuro e recarregar a página |

---

## Documentação

- [`Analise/assistente_clinico.ipynb`](Analise/assistente_clinico.ipynb) — treino, métricas e fluxo
- [`DOCS/RELATORIO_TECNICO.md`](DOCS/RELATORIO_TECNICO.md) — relatório da Fase 3
- [`DOCS/ROTEIRO_VIDEO.md`](DOCS/ROTEIRO_VIDEO.md) — roteiro de gravação
- [`notebooks/01_finetuning_pubmedqa.ipynb`](notebooks/01_finetuning_pubmedqa.ipynb) — runbook de GPU

Apoio educacional. Não substitui avaliação médica e não emite prescrição.
