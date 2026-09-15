"""QLoRA com peft + bitsandbytes."""
from __future__ import annotations

import argparse
import json
import os

from finetuning.config import detect_profile
from hospital.paths import SFT_DIR

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUT = os.environ.get(
    "LLM_ADAPTER_DIR",
    os.path.join(_ROOT, "models", "adapter"),
)


def train(output_dir: str = DEFAULT_OUT) -> str:
    profile = detect_profile()
    train_file = os.path.join(SFT_DIR, "train.jsonl")
    if not os.path.isfile(train_file):
        raise SystemExit("Rode finetuning/prepare_data.py antes do treino.")

    import torch
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    tokenizer = AutoTokenizer.from_pretrained(profile["base_model"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        profile["base_model"],
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        ),
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(
        model,
        LoraConfig(
            r=profile["lora_r"],
            lora_alpha=profile["lora_alpha"],
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )

    dataset = load_dataset("json", data_files=train_file, split="train")
    hospital = dataset.filter(lambda row: row.get("origin") == "hospital")
    pubmed = dataset.filter(lambda row: row.get("origin") != "hospital")
    hospital_unique = len(hospital)
    hospital = upsample_hospital(hospital, pubmed)
    dataset = concatenate(hospital, pubmed)

    os.makedirs(output_dir, exist_ok=True)
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=SFTConfig(
            output_dir=output_dir,
            dataset_text_field="text",
            max_length=profile["max_seq"],
            per_device_train_batch_size=profile["batch"],
            gradient_accumulation_steps=profile["grad_accum"],
            num_train_epochs=profile["epochs"],
            learning_rate=profile["lr"],
            logging_steps=25,
            warmup_ratio=0.03,
            lr_scheduler_type="cosine",
            save_strategy="no",
            seed=profile["seed"],
            report_to="none",
            gradient_checkpointing=profile.get("grad_checkpoint", True),
            fp16=True,
            optim="adamw_torch",
            max_grad_norm=0.3,
        ),
    )
    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    history = [
        {"step": row["step"], "loss": row["loss"]}
        for row in trainer.state.log_history
        if "loss" in row
    ]
    metrics_path = os.path.join(SFT_DIR, "train_metrics.json")
    os.makedirs(SFT_DIR, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "n": len(dataset),
                "hospital": len(hospital),
                "hospital_unique": hospital_unique,
                "pubmed": len(pubmed),
                "epochs": profile["epochs"],
                "lora_r": profile["lora_r"],
                "max_seq": profile["max_seq"],
                "history": history,
            },
            handle,
            indent=2,
        )
    del trainer
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return output_dir


def upsample_hospital(hospital, pubmed, target_share: float = 0.05):
    if len(hospital) == 0 or len(pubmed) == 0:
        return hospital
    need = max(len(hospital), int(len(pubmed) * target_share / max(1.0 - target_share, 0.01)))
    repeats = max(1, (need + len(hospital) - 1) // len(hospital))
    if repeats == 1:
        return hospital
    from datasets import concatenate_datasets

    return concatenate_datasets([hospital] * repeats)


def concatenate(left, right):
    from datasets import concatenate_datasets

    if len(left) == 0:
        return right
    if len(right) == 0:
        return left
    return concatenate_datasets([left, right])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=DEFAULT_OUT)
    args = parser.parse_args()
    print(detect_profile())
    print(train(args.output))


if __name__ == "__main__":
    main()
