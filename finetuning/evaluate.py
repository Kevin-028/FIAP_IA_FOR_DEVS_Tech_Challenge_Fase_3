"""Compara previsões base vs fine-tuned."""
from __future__ import annotations

import argparse
import json
import os

from finetuning.metrics import accuracy, confusion, macro_f1, normalize_label
from hospital.paths import SFT_DIR


def score_file(path: str) -> dict:
    gold, pred = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            gold.append(row["gold"])
            pred.append(normalize_label(row.get("prediction", "")) or "maybe")
    return {
        "n": len(gold),
        "accuracy": accuracy(gold, pred),
        "macro_f1": macro_f1(gold, pred),
        "confusion": confusion(gold, pred),
    }


def predict_split(rows: list[dict], adapter: str | None, limit: int, out_path: str) -> str:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    from finetuning.config import resolve_base_model
    from finetuning.metrics import normalize_label
    from finetuning.prepare_data import qa_user

    base = resolve_base_model()
    tokenizer = AutoTokenizer.from_pretrained(base)
    model = AutoModelForCausalLM.from_pretrained(
        base,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        ),
        device_map="auto",
    )
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    model.eval()

    picked = rows[:limit]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        for row in picked:
            messages = [
                {
                    "role": "system",
                    "content": "Answer with yes, no, or maybe, then a short explanation. Cite the PMID.",
                },
                {"role": "user", "content": qa_user(row)},
            ]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=480)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=24,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )
            text = tokenizer.decode(generated[0, inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
            label = normalize_label(text) or "maybe"
            handle.write(
                json.dumps(
                    {"pubid": row.get("pubid"), "gold": row["final_decision"], "prediction": label, "raw": text[:180]},
                    ensure_ascii=False,
                )
                + "\n"
            )
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="")
    parser.add_argument("--tuned", default="")
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument(
        "--adapter",
        default=os.environ.get(
            "LLM_ADAPTER_DIR",
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "adapter"),
        ),
    )
    parser.add_argument("--out", default=os.path.join(SFT_DIR, "eval_model.json"))
    args = parser.parse_args()
    if args.generate:
        with open(os.path.join(SFT_DIR, "test_labeled.json"), encoding="utf-8") as f:
            rows = json.load(f)
        base_path = os.path.join(SFT_DIR, "pred_base.jsonl")
        tuned_path = os.path.join(SFT_DIR, "pred_tuned.jsonl")
        print("base", predict_split(rows, None, args.limit, base_path))
        print("tuned", predict_split(rows, args.adapter, args.limit, tuned_path))
        args.base, args.tuned = base_path, tuned_path
    if not args.base or not args.tuned:
        raise SystemExit("Passe --base e --tuned, ou --generate.")
    report = {"base": score_file(args.base), "fine_tuned": score_file(args.tuned)}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
