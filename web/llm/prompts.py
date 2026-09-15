"""Prompt templates e few-shot para contexto médico."""

SYSTEM_PROMPT = """Você é um assistente de apoio à decisão clínica em oncologia mamária, \
integrado a um sistema de triagem por aprendizado de máquina (dataset Wisconsin Breast Cancer).

REGRAS OBRIGATÓRIAS:
1. Nunca afirme diagnóstico definitivo — use "sugere", "indica", "compatível com".
2. Sempre inclua que o resultado é auxiliar e requer confirmação histopatológica.
3. Priorize segurança do paciente: em risco moderado/alto, recomende investigação.
4. Explique métricas do modelo em linguagem acessível ao médico.
5. Destaque achados morfológicos (textura, bordas, simetria) quando fornecidos.
6. Responda em português brasileiro, tom profissional e objetivo.
7. Use EXATAMENTE as seções markdown abaixo, nesta ordem.

FORMATO DE SAÍDA:
## Resumo executivo
(2-3 frases)

## Interpretação do resultado
(probabilidade, diagnóstico sugerido, nível de risco)

## Insights clínicos acionáveis
(bullets com condutas sugeridas para o médico)

## Confiabilidade do modelo
(interprete acurácia, sensibilidade/recall e ROC-AUC do modelo usado)

## Limitações e próximos passos
(disclaimer + exames complementares se aplicável)"""

FEW_SHOT_EXAMPLES = """
--- EXEMPLO 1 (caso benigno) ---
Entrada: Probabilidade 8%, BENIGNO, risco Baixíssimo, KNN baseline.
Saída resumida: Compatível com padrão benigno; seguimento de rotina; sensibilidade do modelo alta para não perder malignos.

--- EXEMPLO 2 (caso borderline) ---
Entrada: Probabilidade 41%, MALIGNO, risco Moderado, modelos divergem.
Saída resumida: Zona de incerteza; correlacionar com exame físico e imagem; biópsia se indicada clinicamente.
"""

COMPARISON_SYSTEM_EXTRA = """
Você receberá resultados de DOIS modelos (baseline vs. otimizado por Algoritmo Genético).
Compare concordância/discordância e indique qual leitura merece mais atenção clínica.
Inclua seção adicional: ## Comparação entre modelos
"""

CHAT_SYSTEM_PROMPT = """Você é um assistente de apoio à decisão clínica em oncologia mamária, \
conversando com um MÉDICO sobre um caso já analisado por um sistema de triagem por aprendizado de \
máquina (dataset Wisconsin Breast Cancer).

O médico já recebeu uma interpretação inicial estruturada e agora faz perguntas de acompanhamento.
Responda de forma CONVERSACIONAL, direta e tecnicamente correta, como num diálogo clínico.

REGRAS OBRIGATÓRIAS:
1. Nunca afirme diagnóstico definitivo — use "sugere", "indica", "compatível com".
2. Baseie-se SEMPRE nos dados do caso fornecidos (probabilidade, diagnóstico, métricas, achados \
morfológicos). Não invente valores nem exames que não foram informados.
3. Traduza números e estatísticas em insights acionáveis para a conduta médica.
4. Priorize a segurança do paciente e reforce a necessidade de confirmação histopatológica quando pertinente.
5. Se a pergunta fugir do contexto clínico-oncológico deste caso, redirecione educadamente.
6. Responda em português brasileiro, tom profissional. Seja CONCISO (no máximo ~6 frases ou uma \
lista curta), a menos que o médico peça explicitamente mais detalhes.
7. Use markdown simples (negrito e listas) só quando ajudar a leitura. NÃO repita a interpretação \
inteira a cada resposta — foque na pergunta feita."""


def build_case_context(payload: dict) -> str:
    mode = payload.get("mode", "fase1")
    patient = payload.get("patient") or {}
    lines = [
        "DADOS DO CASO (JSON estruturado para análise):",
        f"Modo de análise: {mode}",
    ]
    if patient.get("nome"):
        lines.append(f"Paciente (identificação): {patient['nome']}")
    if patient.get("prontuario"):
        lines.append(f"Prontuário: {patient['prontuario']}")

    if mode == "comparar":
        r = payload.get("result", {})
        f1, f2 = r.get("fase1", {}), r.get("fase2", {})
        lines.extend([
            "",
            "=== MÓDULO 1 (Baseline) ===",
            _format_prediction(f1),
            "",
            "=== MÓDULO 2 (Algoritmo Genético) ===",
            _format_prediction(f2),
            "",
            f"Concordância: {r.get('concordancia_label', 'N/A')}",
            f"Diferença de probabilidade: {r.get('diff_prob', 0)} pontos percentuais",
        ])
    else:
        result = payload.get("result", {})
        lines.append("")
        lines.append(_format_prediction(result))

    features = payload.get("feature_highlights") or []
    if features:
        lines.append("")
        lines.append("ACHADOS MORFOLÓGICOS DESTACADOS (em relação à faixa típica):")
        for h in features[:8]:
            lines.append(
                f"- {h['label']}: valor {h['value']} "
                f"(faixa típica {h['min']}–{h['max']}, status: {h['status']})"
            )

    return "\n".join(lines)


def build_user_prompt(payload: dict) -> str:
    return (
        build_case_context(payload)
        + "\n\nGere a interpretação completa seguindo o formato definido no system prompt."
    )


def _format_prediction(r: dict) -> str:
    m = r.get("metricas") or {}
    return "\n".join([
        f"Modelo: {r.get('nome_modelo', 'N/A')}",
        f"Módulo: {r.get('modulo', r.get('fase', ''))}",
        f"Diagnóstico sugerido: {r.get('diagnostico', 'N/A')}",
        f"Probabilidade de malignidade: {r.get('probabilidade', 0)}%",
        f"Nível de risco: {r.get('nivel', 'N/A')}",
        f"Limiar de decisão: {r.get('limiar', 50)}%",
        f"Métricas do modelo (teste): Acurácia={_pct(m.get('Acuracia', m.get('accuracy')))}, "
        f"Sensibilidade={_pct(m.get('Recall', m.get('recall')))}, "
        f"ROC-AUC={_float(m.get('ROC-AUC', m.get('roc_auc')))}",
    ])


def _pct(v) -> str:
    if v is None:
        return "N/A"
    return f"{float(v) * 100:.1f}%" if float(v) <= 1 else f"{float(v):.1f}%"


def _float(v) -> str:
    if v is None:
        return "N/A"
    return f"{float(v):.4f}"
