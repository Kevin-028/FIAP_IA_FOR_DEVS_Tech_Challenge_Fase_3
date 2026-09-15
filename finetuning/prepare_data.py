"""Curadoria PubMedQA → JSONL SFT."""
from __future__ import annotations

import argparse
import json
import os
import random
import re
from collections import Counter

from hospital.paths import PUBMEDQA_RAW, SFT_DIR
from finetuning.config import SEED, SYSTEM_PROMPT, resolve_base_model

SAMPLE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "hospital_data", "samples", "pubmedqa_mini.json",
)

PHI_SCAN = re.compile(
    r"([A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}|\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b)",
    re.I,
)


def qwen_text(user: str, assistant: str, system: str = SYSTEM_PROMPT) -> str:
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        f"<|im_start|>assistant\n{assistant}<|im_end|>\n"
    )


def format_text(user: str, assistant: str) -> str:
    if "qwen" in resolve_base_model().lower():
        return qwen_text(user, assistant)
    return llama32_text(user, assistant)


def llama32_text(user: str, assistant: str, system: str = SYSTEM_PROMPT) -> str:
    return (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"{system}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        f"{user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        f"{assistant}<|eot_id|>"
    )


def _join_context(raw) -> str:
    if isinstance(raw, list):
        return " ".join(str(x) for x in raw if x)
    if isinstance(raw, dict):
        ctx = raw.get("contexts") or raw.get("context") or []
        if isinstance(ctx, list):
            return " ".join(str(x) for x in ctx)
        return str(ctx)
    return str(raw or "")


def normalize_row(row: dict, source: str, split: str | None = None) -> dict | None:
    question = (row.get("question") or "").strip()
    context = _join_context(row.get("context") or row.get("contexts"))
    decision = (row.get("final_decision") or "").strip().lower()
    long_answer = (row.get("long_answer") or "").strip()
    pubid = str(row.get("pubid") or row.get("pmid") or "")
    if not question or not context or decision not in {"yes", "no", "maybe"}:
        return None
    if PHI_SCAN.search(question + context + long_answer):
        return None
    return {
        "pubid": pubid,
        "question": question,
        "context": context[:4000],
        "long_answer": long_answer,
        "final_decision": decision,
        "source": source,
        "split": split or row.get("split") or "train",
    }


def _from_hf_item(item: dict, source: str, split: str | None = None) -> dict | None:
    ctx = item.get("context") or {}
    return normalize_row(
        {
            "pubid": item.get("pubid"),
            "question": item.get("question"),
            "contexts": ctx.get("contexts") if isinstance(ctx, dict) else ctx,
            "long_answer": item.get("long_answer"),
            "final_decision": item.get("final_decision"),
        },
        source,
        split,
    )


def load_huggingface(artificial_cap: int = 800) -> list[dict]:
    from datasets import load_dataset

    rows = []
    labeled = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
    labeled_rows = []
    for item in labeled:
        row = _from_hf_item(item, "pqa_labeled")
        if row:
            labeled_rows.append(row)
    labeled_rows.sort(key=lambda r: int(r["pubid"]) if str(r["pubid"]).isdigit() else 0)
    cut = max(1, len(labeled_rows) // 2)
    for i, row in enumerate(labeled_rows):
        row["split"] = "train" if i < cut else "test"
        rows.append(row)

    need_each = max(artificial_cap // 2, 1)
    yes, no = [], []
    artificial = load_dataset("qiaojin/PubMedQA", "pqa_artificial", split="train", streaming=True)
    for item in artificial:
        row = _from_hf_item(item, "pqa_artificial", "train")
        if not row:
            continue
        label = row["final_decision"]
        if label == "yes" and len(yes) < need_each:
            yes.append(row)
        elif label == "no" and len(no) < need_each:
            no.append(row)
        if len(yes) >= need_each and len(no) >= need_each:
            break
    rows.extend(yes + no)
    return rows


def load_sample() -> list[dict]:
    with open(SAMPLE_PATH, encoding="utf-8") as f:
        payload = json.load(f)
    rows = []
    for item in payload["labeled"]:
        row = normalize_row(item, "pqa_labeled", item.get("split"))
        if row:
            rows.append(row)
    for item in payload["artificial"]:
        row = normalize_row(item, "pqa_artificial", "train")
        if row:
            rows.append(row)
    return rows


def balance_artificial(rows: list[dict], rng: random.Random, cap: int) -> list[dict]:
    yes = [r for r in rows if r["final_decision"] == "yes"]
    no = [r for r in rows if r["final_decision"] == "no"]
    rng.shuffle(yes)
    rng.shuffle(no)
    keep_no = no
    keep_yes = yes[: max(len(keep_no), 1)]
    picked = keep_yes + keep_no
    rng.shuffle(picked)
    return picked[:cap]


def qa_user(row: dict) -> str:
    return (
        f"Clinical question: {row['question']}\n\n"
        f"Abstract:\n{row['context']}\n\n"
        "Answer with yes, no, or maybe, then a short explanation. Cite the PMID."
    )


def qa_assistant(row: dict) -> str:
    explanation = row["long_answer"] or "See the abstract."
    return f"{row['final_decision']}. {explanation} PMID {row['pubid']}."


def protocol_pair(row: dict) -> tuple[str, str]:
    user = f"What does the evidence say about: {row['question']}"
    assistant = (
        f"Protocol note based on PMID {row['pubid']}: {row['long_answer'] or row['context'][:400]} "
        f"REQUIRES PHYSICIAN VALIDATION."
    )
    return user, assistant


def report_pair(row: dict) -> tuple[str, str]:
    user = f"Draft an internal report for this question: {row['question']}"
    assistant = (
        f"Report draft\nQuestion: {row['question']}\n"
        f"Decision: {row['final_decision']}\n"
        f"Findings: {row['long_answer'] or row['context'][:300]}\n"
        f"Source: PMID {row['pubid']}\n"
        "Suggested prescription: none — this assistant does not prescribe.\n"
        "REQUIRES PHYSICIAN VALIDATION."
    )
    return user, assistant


def to_sft(rows: list[dict]) -> list[dict]:
    out = []
    for i, row in enumerate(rows):
        pairs = [(qa_user(row), qa_assistant(row))]
        if i % 3 == 0:
            pairs.append(protocol_pair(row))
        if i % 5 == 0:
            pairs.append(report_pair(row))
        for user, assistant in pairs:
            out.append({
                "text": format_text(user, assistant),
                "pubid": row["pubid"],
                "task": "qa" if user.startswith("Clinical") else "protocol_or_report",
                "final_decision": row["final_decision"],
                "origin": "pubmedqa",
            })
    return out


def hospital_sft() -> list[dict]:
    from hospital.catalog import PROTOCOLS

    rows = []
    for proto in PROTOCOLS:
        pid, pmid = proto["id"], proto["pmid"]
        rows.append({
            "text": format_text(
                f"What does hospital protocol {pid} say?",
                f"{proto['body']} PMID {pmid}.",
            ),
            "pubid": pmid,
            "task": "hospital_protocol",
            "final_decision": "",
            "origin": "hospital",
        })
        rows.append({
            "text": format_text(
                f"Physician FAQ: a colleague asks about {proto['title']}. What should I answer?",
                f"Use protocol {pid} (PMID {pmid}). {proto['body']}",
            ),
            "pubid": pmid,
            "task": "physician_faq",
            "final_decision": "",
            "origin": "hospital",
        })
        rows.append({
            "text": format_text(
                f"Draft an internal report, a procedure note and a prescription draft for protocol {pid}.",
                (
                    f"Report draft\nProtocol: {pid}\nPMID: {pmid}\n"
                    f"Findings: {proto['body'][:280]}\n"
                    "Procedure: follow the protocol pathway and record pending exams in the chart.\n"
                    "Suggested prescription: none. This assistant does not prescribe a drug, dose or route.\n"
                    "REQUIRES PHYSICIAN VALIDATION."
                ),
            ),
            "pubid": pmid,
            "task": "report_procedure_prescription",
            "final_decision": "",
            "origin": "hospital",
        })
    return rows


def prepare(sample_cap: int | None = None, prefer_sample: bool = False) -> dict:
    if sample_cap is None:
        from finetuning.config import detect_profile

        sample_cap = detect_profile()["sample"]
    os.makedirs(PUBMEDQA_RAW, exist_ok=True)
    os.makedirs(SFT_DIR, exist_ok=True)
    source = "sample"
    rows = []
    if not prefer_sample:
        try:
            rows = load_huggingface(sample_cap)
            source = "huggingface"
        except Exception as exc:
            print(f"Hugging Face indisponível ({exc}). Usando amostra local.")
    if not rows:
        rows = load_sample()
        source = "sample"

    rng = random.Random(SEED)
    train_labeled = [r for r in rows if r["source"] == "pqa_labeled" and r["split"] == "train"]
    test_labeled = [r for r in rows if r["source"] == "pqa_labeled" and r["split"] == "test"]
    artificial = [r for r in rows if r["source"] == "pqa_artificial"]
    artificial = balance_artificial(artificial, rng, sample_cap)

    train_rows = artificial + train_labeled
    sft = hospital_sft() + to_sft(train_rows)
    rng.shuffle(sft)

    def dump(name, payload):
        path = os.path.join(SFT_DIR, name)
        with open(path, "w", encoding="utf-8") as f:
            if name.endswith(".jsonl"):
                for item in payload:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            else:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        return path

    train_path = dump("train.jsonl", sft)
    sample_path = dump("train_sample_200.jsonl", sft[:200])
    test_path = dump("test_labeled.json", test_labeled)
    stats = {
        "source": source,
        "train_examples": len(sft),
        "hospital_examples": sum(1 for row in sft if row.get("origin") == "hospital"),
        "template": "qwen" if "qwen" in resolve_base_model().lower() else "llama32",
        "test_labeled": len(test_labeled),
        "train_label_counts": dict(Counter(r["final_decision"] for r in train_rows)),
        "artificial_before_balance_yes_share": None,
        "paths": {"train": train_path, "sample": sample_path, "test": test_path},
    }
    dump("stats.json", stats)
    return stats


def main() -> None:
    from finetuning.config import detect_profile

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cap",
        type=int,
        default=None,
        help="Teto do pqa_artificial. Padrão: sample do perfil de VRAM.",
    )
    parser.add_argument("--sample-only", action="store_true")
    args = parser.parse_args()
    cap = args.cap if args.cap is not None else detect_profile()["sample"]
    print(json.dumps(prepare(cap, args.sample_only), indent=2))


if __name__ == "__main__":
    main()
