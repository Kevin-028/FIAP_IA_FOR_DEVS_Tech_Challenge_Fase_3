"""Cliente LLM local (Ollama) e fallback por regras clínicas."""
from __future__ import annotations

from abc import ABC, abstractmethod

import requests

from .config import (
    LLM_MAX_TOKENS,
    LLM_PROVIDER,
    LLM_TEMPERATURE,
    LLM_TIMEOUT,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
)


class BaseLLMClient(ABC):
    @abstractmethod
    def complete(self, system: str, user: str) -> str:
        pass

    def chat(self, system: str, messages: list[dict]) -> str:
        convo = []
        for m in messages:
            who = "MÉDICO" if m.get("role") == "user" else "ASSISTENTE"
            convo.append(f"{who}: {m.get('content', '')}")
        return self.complete(system, "\n\n".join(convo))

    @property
    @abstractmethod
    def provider_name(self) -> str:
        pass


class OllamaClient(BaseLLMClient):
    @property
    def provider_name(self) -> str:
        return f"ollama/{OLLAMA_MODEL}"

    @staticmethod
    def _extract(data: dict) -> str:
        msg = data.get("message", {}) or {}
        text = (msg.get("content") or "").strip()
        if not text:
            text = (msg.get("thinking") or "").strip()
        if not text:
            raise RuntimeError("Ollama retornou conteúdo vazio")
        return text

    def _post(self, messages: list[dict]) -> str:
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={
                "model": OLLAMA_MODEL,
                "messages": messages,
                "stream": False,
                "options": {"temperature": LLM_TEMPERATURE, "num_predict": LLM_MAX_TOKENS},
            },
            timeout=LLM_TIMEOUT,
        )
        resp.raise_for_status()
        return self._extract(resp.json())

    def complete(self, system: str, user: str) -> str:
        return self._post(
            [{"role": "system", "content": system}, {"role": "user", "content": user}]
        )

    def chat(self, system: str, messages: list[dict]) -> str:
        return self._post([{"role": "system", "content": system}, *messages])


class FallbackClient(BaseLLMClient):
    @property
    def provider_name(self) -> str:
        return "fallback/regras-clinicas"

    def complete(self, system: str, user: str) -> str:
        return generate_rule_based_interpretation(user)

    def chat(self, system: str, messages: list[dict]) -> str:
        return generate_rule_based_chat(messages)


class LocalLlamaClient(BaseLLMClient):
    @property
    def provider_name(self) -> str:
        return "huggingface/local"

    def complete(self, system: str, user: str) -> str:
        from .engine import engine

        return engine.complete(system, user)

    def chat(self, system: str, messages: list[dict]) -> str:
        return self.complete(system, self._flatten(messages))

    @staticmethod
    def _flatten(messages: list[dict]) -> str:
        lines = []
        for message in messages:
            who = "PHYSICIAN" if message.get("role") == "user" else "ASSISTANT"
            lines.append(f"{who}: {message.get('content', '')}")
        return "\n\n".join(lines)


def get_client() -> BaseLLMClient:
    if LLM_PROVIDER == "fallback":
        return FallbackClient()
    if LLM_PROVIDER == "ollama":
        return OllamaClient()
    return LocalLlamaClient()


def _case_facts(text: str) -> dict:
    prob = _extract_float(text, "Probabilidade de malignidade:", "%")
    diag = "MALIGNO" if "Diagnóstico sugerido: MALIGNO" in text else "BENIGNO"
    if "Diagnóstico sugerido: BENIGNO" in text:
        diag = "BENIGNO"
    modelo = _extract_line(text, "Modelo:")
    nivel = _extract_line(text, "Nível de risco:")
    limiar = _extract_line(text, "Limiar de decisão:")
    metricas = _extract_line(text, "Métricas do modelo (teste):")
    concordam = "Concordância: Concordam" in text
    divergem = "Concordância: Divergem" in text

    highlights = []
    for line in text.splitlines():
        if line.strip().startswith("- ") and "faixa típica" in line:
            highlights.append(line.strip()[2:])

    if prob is None:
        prob = 50.0

    if prob < 20:
        risco = "Baixíssimo"
        conduta = [
            "Manter seguimento de rotina conforme diretrizes de rastreamento.",
            "Correlacionar com exame físico e mamografia/ultrassom se disponíveis.",
        ]
    elif prob < 50:
        risco = "Moderado"
        conduta = [
            "Revisar achados morfológicos destacados e histórico clínico.",
            "Considerar repetição da citologia ou investigação por imagem.",
            "Encaminhar para especialista se houver fatores de risco adicionais.",
        ]
    elif prob < 75:
        risco = "Alto"
        conduta = [
            "Priorizar avaliação especializada em mama/oncologia.",
            "Considerar biópsia ou procedimento diagnóstico conforme protocolo institucional.",
            "Não postergar investigação em presença de sintomas ou achados de imagem suspeitos.",
        ]
    else:
        risco = "Muito Alto"
        conduta = [
            "Tratar como suspeita significativa até confirmação histopatológica.",
            "Encaminhamento urgente para biópsia e estadiamento conforme diretrizes.",
            "Registrar discordância clínica se o exame físico não corroborar o achado.",
        ]

    return {
        "prob": prob,
        "diag": diag,
        "modelo": modelo,
        "nivel": nivel or risco,
        "risco": risco,
        "limiar": limiar,
        "metricas": metricas,
        "concordam": concordam,
        "divergem": divergem,
        "highlights": highlights,
        "conduta": conduta,
    }


def generate_rule_based_interpretation(user_prompt: str) -> str:
    facts = _case_facts(user_prompt)
    prob = facts["prob"]
    diag = facts["diag"]
    modelo = facts["modelo"]
    risco = facts["risco"]
    highlights = facts["highlights"]
    conduta = facts["conduta"]
    concordam = facts["concordam"]
    divergem = facts["divergem"]

    bullets = "\n".join(f"- {c}" for c in conduta)
    morph = ""
    if highlights:
        morph = "\n".join(f"- {h}" for h in highlights[:5])
        morph = f"\n\nAchados morfológicos relevantes:\n{morph}"

    comp_section = ""
    if divergem:
        comp_section = """
## Comparação entre modelos
Os modelos **baseline** e **otimizado** **divergem** neste caso.
Recomenda-se cautela redobrada: priorize a leitura com maior sensibilidade (menor risco de falso negativo)
e confirme com histopatologia. A divergência pode ocorrer em casos borderline ou com atipia leve.
"""
    elif concordam:
        comp_section = """
## Comparação entre modelos
Ambos os modelos **concordam** no diagnóstico sugerido, aumentando a confiança na predição automática.
Ainda assim, a decisão terapêutica depende de confirmação anatomopatológica.
"""

    return f"""## Resumo executivo
O modelo **{modelo or 'de ML'}** sugere padrão **{diag}** com probabilidade de malignidade de **{prob:.1f}%**
(classificação de risco: **{risco}**). Este resultado é um auxílio à triagem citológica e não substitui o diagnóstico do patologista.

## Interpretação do resultado
A probabilidade estimada indica {"baixa" if prob < 30 else "moderada" if prob < 60 else "elevada"} suspeita de malignidade
com base nas features morfológicas do núcleo celular (PAAF). O limiar de decisão do sistema foi calibrado para
**priorizar sensibilidade**, reduzindo falsos negativos em contexto de rastreamento.{morph}

## Insights clínicos acionáveis
{bullets}

## Confiabilidade do modelo
Consulte as métricas exibidas no painel (acurácia, sensibilidade/recall e ROC-AUC). Em triagem oncológica,
**sensibilidade elevada** é prioritária para não deixar casos malignos sem investigação. Valores de ROC-AUC
próximos de 1 indicam boa separação entre classes no conjunto de teste.

## Limitações e próximos passos
⚠️ **Aviso:** decisão clínica definitiva exige correlação com exame físico, imagem e laudo histopatológico.
O sistema não substitui o julgamento médico. Em caso de dúvida, indicar investigação adicional.
{comp_section}"""


def generate_rule_based_chat(messages: list[dict]) -> str:
    context = ""
    question = ""
    for m in messages:
        role = m.get("role", "")
        content = str(m.get("content", "")).strip()
        if not content:
            continue
        if role == "user":
            if "CONTEXTO DO CASO" in content or "Diagnóstico sugerido:" in content:
                context = content
            else:
                question = content
    if not question and messages:
        question = str(messages[-1].get("content", "")).strip()

    facts = _case_facts(context or question)
    q = question.lower()
    disclaimer = (
        "\n\n⚠️ Resposta em **modo local (regras)** — apoio educacional. "
        "Não substitui avaliação médica nem laudo histopatológico."
    )

    if any(k in q for k in ("influenci", "achado", "morfol", "feature", "variáve", "destaque")):
        if facts["highlights"]:
            bullets = "\n".join(f"- {h}" for h in facts["highlights"][:6])
            body = (
                f"Neste caso (**{facts['diag']}**, {facts['prob']:.1f}%), "
                f"os achados que mais pesam no parecer são:\n\n{bullets}\n\n"
                "Valores fora da faixa típica (especialmente **acima**) tendem a elevar a "
                "suspeita de malignidade na citologia de agulha fina."
            )
        else:
            body = (
                f"O parecer atual aponta **{facts['diag']}** com {facts['prob']:.1f}% de "
                "probabilidade de malignidade. Não há lista detalhada de features fora da faixa "
                "neste contexto; use o painel de métricas e as medidas do formulário para revisão."
            )
        return body + disclaimer

    if any(k in q for k in ("sensibil", "recall", "métric", "metric", "acurác", "roc", "confiabil")):
        met = facts["metricas"] or "consulte o painel do laudo"
        body = (
            f"**Sensibilidade (recall)** mede a fração de casos malignos realmente detectados. "
            f"Em triagem oncológica, priorizamos sensibilidade alta para reduzir falso negativo.\n\n"
            f"Métricas deste modelo: {met}.\n\n"
            f"O limiar atual ({facts['limiar'] or 'padrão'}) foi calibrado para esse equilíbrio; "
            f"a probabilidade estimada neste exame é **{facts['prob']:.1f}%**."
        )
        return body + disclaimer

    if any(k in q for k in ("conduta", "recomenda", "próximo", "proximo", "o que fazer", "condutas")):
        bullets = "\n".join(f"- {c}" for c in facts["conduta"])
        body = (
            f"Para este perfil (**{facts['diag']}**, risco **{facts['risco']}**, "
            f"{facts['prob']:.1f}%):\n\n{bullets}\n\n"
            "A decisão final depende de correlação clínica, imagem e, quando indicado, histopatologia."
        )
        return body + disclaimer

    if any(k in q for k in ("compar", "diverg", "concord", "baseline", "otimiz")):
        if facts["divergem"]:
            body = (
                "Os modelos **divergem** neste exame. Trate com cautela: priorize a leitura "
                "mais sensível (menor risco de falso negativo) e confirme com investigação complementar."
            )
        elif facts["concordam"]:
            body = (
                "Os modelos **concordam** no diagnóstico sugerido, o que aumenta a confiança "
                "na predição automática — sem eliminar a necessidade de confirmação clínica."
            )
        else:
            body = (
                "Neste laudo não há comparação explícita entre baseline e otimizado. "
                "Use a tela **Comparar laudos** com as mesmas medidas para ver concordância."
            )
        return body + disclaimer

    if any(k in q for k in ("risco", "probabil", "diagnóst", "diagnost", "resultado", "laudo", "resumo")):
        body = (
            f"**Resumo do caso:** modelo **{facts['modelo'] or 'de ML'}** sugere "
            f"**{facts['diag']}** com probabilidade **{facts['prob']:.1f}%** "
            f"(nível **{facts['nivel']}**).\n\n"
            "Posso detalhar achados morfológicos, sensibilidade do modelo ou condutas sugeridas. "
            "Escreva a dúvida no chat."
        )
        return body + disclaimer

    body = (
        f"Com base no caso (**{facts['diag']}**, {facts['prob']:.1f}%):\n\n"
        f"- Risco estimado: **{facts['risco']}**\n"
        f"- Modelo: **{facts['modelo'] or 'regressão logística'}**\n\n"
        "Escreva a próxima dúvida no chat."
    )
    return body + disclaimer


def _extract_float(text: str, key: str, suffix: str = "") -> float | None:
    for line in text.splitlines():
        if key in line:
            part = line.split(key, 1)[1].replace(suffix, "").strip()
            try:
                return float(part.split()[0])
            except ValueError:
                return None
    return None


def _extract_line(text: str, key: str) -> str:
    for line in text.splitlines():
        if line.startswith(key):
            return line.split(key, 1)[1].strip()
    return ""
