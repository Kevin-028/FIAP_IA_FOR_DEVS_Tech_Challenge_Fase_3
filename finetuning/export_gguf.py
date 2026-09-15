"""Merge do adapter e conversão GGUF."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys


def merge(adapter_dir: str, merged_dir: str) -> str:
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base = AutoTokenizer.from_pretrained(adapter_dir).name_or_path
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    model = AutoModelForCausalLM.from_pretrained(base, device_map="cpu")
    model = PeftModel.from_pretrained(model, adapter_dir)
    model = model.merge_and_unload()
    os.makedirs(merged_dir, exist_ok=True)
    model.save_pretrained(merged_dir)
    tokenizer.save_pretrained(merged_dir)
    return merged_dir


def to_gguf(merged_dir: str, outfile: str, llama_cpp: str) -> None:
    script = os.path.join(llama_cpp, "convert_hf_to_gguf.py")
    if not os.path.isfile(script):
        raise SystemExit(f"Não achei {script}. Clone llama.cpp e passe --llama-cpp.")
    f16 = outfile.replace(".gguf", ".f16.gguf")
    subprocess.check_call([sys.executable, script, merged_dir, "--outfile", f16, "--outtype", "f16"])
    quant = os.path.join(llama_cpp, "build", "bin", "llama-quantize")
    if os.path.isfile(quant):
        subprocess.check_call([quant, f16, outfile, "Q4_K_M"])
    else:
        print("llama-quantize não encontrado. Use o f16 ou instale o binário.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default="models/adapter")
    parser.add_argument("--merged", default="models/merged")
    parser.add_argument("--outfile", default="models/fiap-fase3-medico.Q4_K_M.gguf")
    parser.add_argument("--llama-cpp", default="")
    parser.add_argument("--skip-gguf", action="store_true")
    args = parser.parse_args()
    merge(args.adapter, args.merged)
    if not args.skip_gguf:
        to_gguf(args.merged, args.outfile, args.llama_cpp)


if __name__ == "__main__":
    main()
