# Integração LLM (somente Fase 2 / Módulo 2)

A interpretação LLM aparece **apenas** nos resultados do **Módulo 2** (KNN otimizado por AG).
O Módulo 1 (baseline) e a página Comparar não exibem interpretação IA.

A LLM roda **localmente via Ollama** — sem cota, sem custo e offline. Se o Ollama
estiver indisponível, o sistema cai automaticamente em uma **interpretação por regras**
(fallback), para o médico sempre ver algum conteúdo.

## Objetivo

Gerar explicações em linguagem natural dos diagnósticos do modelo **Fase 2 (KNN + AG)**,
transformando probabilidades e métricas em insights acionáveis para médicos.

## Arquitetura

```
Página de resultado Módulo 2 (/fase2/predict)
        │
        ▼ (async fetch)
POST /api/llm/interpret   |   POST /api/llm/chat
        │
        ▼
web/llm/interpreter.py
   ├── prompts.py      (system + few-shot + user estruturado)
   ├── context.py      (destaque de features morfológicas)
   └── client.py       (Ollama | fallback por regras)
        │
        ▼
web/logs/llm_interpretations.jsonl
```

## Provedores suportados

| Provedor | Variável | Uso |
|----------|----------|-----|
| `ollama` | `OLLAMA_BASE_URL`, `OLLAMA_MODEL` | **LLM local (padrão)** — gpt-oss, llama3.2, etc. |
| `fallback` | — | Interpretação por regras, offline (usado se o Ollama falhar) |

## Prompt engineering

1. **System prompt** — papel de assistente clínico, regras de segurança, formato markdown fixo
2. **Few-shot** — exemplos benigno e borderline no system prompt
3. **User prompt** — dados estruturados: modelo, probabilidade, métricas, achados morfológicos
4. **Temperatura baixa** (0.25) — respostas consistentes
5. **Seções obrigatórias** — Resumo, Interpretação, Insights, Confiabilidade, Limitações

## Avaliação de qualidade

Página: `/llm/avaliacao`

Rubrica heurística (0–1):
- **Clareza** — seções presentes
- **Relevância** — diagnóstico e probabilidade mencionados
- **Segurança** — disclaimers, sem linguagem definitiva
- **Acionabilidade** — recomendações clínicas

Logs em `web/logs/llm_interpretations.jsonl` para auditoria.

## Configurar Ollama (local)

### Passo a passo

1. Instale o **[Ollama](https://ollama.com)** e inicie um modelo:

```bash
ollama run gpt-oss
# alternativas: ollama run llama3.2   |   ollama run qwen2.5
```

2. Crie o arquivo `.env` na raiz do projeto (`Fase_2/`) — ou `copy .env.example .env`:

```env
FIAP_LLM_PROVIDER=ollama
OLLAMA_MODEL=gpt-oss
OLLAMA_BASE_URL=http://localhost:11434
FIAP_LLM_TIMEOUT=180
```

3. Reinicie o Flask: `python web/app.py`

4. Faça uma predição em `/fase2/` — o badge deve mostrar `ollama/gpt-oss`.
   Confira também `GET http://localhost:5000/api/llm/status`.

### Observações

- **Nome do modelo**: use exatamente o que aparece em `ollama list`
  (ex.: `gpt-oss:20b`). O nome curto (`gpt-oss`) normalmente resolve.
- **Primeira resposta é lenta**: o Ollama carrega o modelo na memória; por isso o
  `FIAP_LLM_TIMEOUT` alto evita cair no fallback por timeout.
- **Modelos de raciocínio** (ex.: gpt-oss): o Ollama devolve a resposta em
  `message.content` e o raciocínio em `message.thinking`; o cliente usa o `content`
  e, se vazio, o `thinking`.

## Como funciona no código

```
Resultado Fase 2 → POST /api/llm/interpret
    → prompts.py (system + few-shot + dados clínicos)
    → OllamaClient → http://localhost:11434/api/chat
    → Resposta markdown exibida na página
```

Teste da API de status:
```bash
curl http://localhost:5000/api/llm/status
```
