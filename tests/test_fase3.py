"""Fase 3: anonimização, guardrails, grafo e rotas."""
from __future__ import annotations

import pytest
from langgraph.checkpoint.memory import MemorySaver
from assistant.graph import build_graph, is_interrupted, thread_config
from assistant.guardrails import postfilter, prefilter
from assistant.service import configure
from hospital.anonymize import anonymize_patient, residual_phi
from hospital.catalog import RAW_PATIENTS
from hospital.db import connect, pending_exams, record_exam_result
from hospital.seed import build
from finetuning.metrics import accuracy, macro_f1
from finetuning.prepare_data import llama32_text, prepare
from hospital.artifacts import snapshot


@pytest.fixture()
def hospital_db(tmp_path, monkeypatch):
    path = str(tmp_path / "hospital.db")
    summary = build(path)
    assert summary["residual_phi"] == []
    monkeypatch.setattr("assistant.tools.DB_PATH", path)
    monkeypatch.setattr("hospital.db.DB_PATH", path)
    configure(path)
    return path


def test_anonimizacao_remove_phi():
    clean, _preview = anonymize_patient(RAW_PATIENTS[0], 1)
    blob = " ".join([clean["display_name"], clean["notes"], clean["diagnosis"], clean["chart_hash"]])
    assert residual_phi(blob) == []
    assert clean["display_name"] == "PAC-0001"
    assert "390" not in blob
    assert "@" not in blob


def test_prepare_separa_teste_e_usa_template_llama(tmp_path, monkeypatch):
    monkeypatch.setattr("finetuning.prepare_data.SFT_DIR", str(tmp_path))
    monkeypatch.setattr("finetuning.prepare_data.PUBMEDQA_RAW", str(tmp_path / "raw"))
    stats = prepare(sample_cap=10, prefer_sample=True)
    assert stats["test_labeled"] >= 1
    text = llama32_text("Q?", "yes. PMID 1")
    assert "<|start_header_id|>system<|end_header_id|>" in text
    assert text.strip().endswith("<|eot_id|>")


def test_web_consome_metricas_do_sft():
    report = snapshot()
    assert report["ft_n"] == 500
    assert report["ft_accuracy"] == 0.628
    assert report["base_accuracy"] == 0.266
    assert report["adapter_ok"] is True


def test_metricas_penalizam_sempre_yes():
    gold = ["yes", "no", "maybe"]
    assert accuracy(gold, ["yes", "yes", "yes"]) < accuracy(gold, gold)
    assert macro_f1(gold, gold) == 1.0


def test_guardrail_prescricao_e_fora_de_escopo():
    assert prefilter("Prescribe 500 mg oral")["prescription_request"]
    assert prefilter("What is the weather today?")["out_of_scope"]
    assert postfilter("Start 40 mg IV now")
    assert not prefilter("Do statins reduce atrial fibrillation after CABG?")["prescription_request"]


def _stub_llm(question: str, docs: list, record: dict) -> str:
    pending = [p.get("name") for p in (record.get("pending") or []) if p.get("name")]
    top = docs[0] if docs else None
    pmid = top["pmid"] if top else "none"
    q = (question or "").lower()
    if any(w in q for w in ("pending", "lab", "labs", "wait for")):
        names = ", ".join(pending) if pending else "none"
        return f"Pending exams for this chart: {names}. PMID {pmid}."
    if top:
        return f"Evidence supports a lower AF rate with perioperative statin. PMID {pmid}."
    return "Insufficient protocol evidence for this question."


def test_grafo_cita_fonte_e_bloqueia_prescricao(hospital_db):
    graph = build_graph(checkpointer=MemorySaver(), complete=_stub_llm)
    ok = graph.invoke(
        {
            "thread_id": "t-ok",
            "patient_id": "PAC-0001",
            "question": "Do preoperative statins reduce atrial fibrillation after CABG?",
        },
        thread_config("t-ok"),
    )
    assert not is_interrupted(ok)
    assert "PMID" in ok["final_answer"]
    assert "consult" in ok["final_answer"].lower() or "physician" in ok["final_answer"].lower()
    assert ok["sources"]

    blocked = graph.invoke(
        {
            "thread_id": "t-rx",
            "patient_id": "PAC-0001",
            "question": "Prescribe atorvastatin 40 mg oral now",
        },
        thread_config("t-rx"),
    )
    assert not is_interrupted(blocked)
    text = (blocked.get("final_answer") or "").lower()
    assert "does not prescribe" in text or "consult" in text
    assert "40 mg" not in text
    assert "prescription_blocked" in (blocked.get("safety_flags") or [])


def test_pending_labs_nao_inventa_exame_de_outro_paciente(hospital_db):
    graph = build_graph(checkpointer=MemorySaver(), complete=_stub_llm)
    out = graph.invoke(
        {
            "thread_id": "t-labs",
            "patient_id": "PAC-0001",
            "question": "Are there pending labs I should wait for before a statin discussion?",
        },
        thread_config("t-labs"),
    )
    text = (out["final_answer"] or "").lower()
    assert "lipid" in text
    assert "troponin" not in text
    pmids = {str(s.get("pmid")) for s in out.get("sources") or []}
    assert "17625060" in pmids
    assert "25173350" not in pmids


def test_grafo_exige_complete():
    with pytest.raises(ValueError, match="obrigatório"):
        build_graph(checkpointer=MemorySaver())


def test_contexto_reler_exame(hospital_db):
    conn = connect(hospital_db)
    before = pending_exams(conn, "PAC-0002")
    assert before
    record_exam_result(conn, before[0]["id"], "troponin negative")
    after = pending_exams(conn, "PAC-0002")
    conn.close()
    assert len(after) == len(before) - 1


def test_rotas_fase3(client, tmp_path, monkeypatch):
    path = str(tmp_path / "hospital.db")
    build(path)
    monkeypatch.setattr("hospital.paths.DB_PATH", path)
    monkeypatch.setattr("hospital.db.DB_PATH", path)
    monkeypatch.setattr("assistant.tools.DB_PATH", path)
    home = client.get("/")
    assert home.status_code == 200
    assert b"Fase 3" in home.data
    assert client.get("/pacientes").status_code == 200
    assert client.get("/assistente").status_code == 200
    assert client.get("/validacao").status_code == 200
    assert client.get("/alertas").status_code == 200
    assert client.get("/ops/monitoramento").status_code == 200
    assert client.get("/modelo").status_code == 200
    assert client.get("/auditoria").status_code == 200
