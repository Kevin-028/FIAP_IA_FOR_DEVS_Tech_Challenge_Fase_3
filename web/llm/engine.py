"""Motor local (Hugging Face Transformers)."""
from __future__ import annotations

import gc
import os
import threading

from .config import LLM_ADAPTER_DIR, LLM_LOAD_IN_4BIT, LLM_MAX_TOKENS, LLM_MODEL_DIR, LLM_TEMPERATURE


class LocalEngine:
    def __init__(self):
        self._lock = threading.Lock()
        self._model = None
        self._tokenizer = None
        self._error = None
        self._backend = "huggingface"

    def status(self) -> dict:
        device = None
        if self._model is not None:
            try:
                device = str(next(self._model.parameters()).device)
            except StopIteration:
                device = None
        return {
            "loaded": self._model is not None,
            "path": LLM_MODEL_DIR,
            "adapter": LLM_ADAPTER_DIR if _adapter_ready() else None,
            "backend": self._backend,
            "load_in_4bit": LLM_LOAD_IN_4BIT,
            "device": device,
            "error": self._error,
        }

    def load(self) -> dict:
        with self._lock:
            if self._model is not None:
                return self.status()
            try:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer

                tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_DIR)
                if tokenizer.pad_token_id is None:
                    tokenizer.pad_token = tokenizer.eos_token

                model = self._load_model(torch, AutoModelForCausalLM)
                model = _maybe_adapter(model)
                self._tokenizer = tokenizer
                self._model = model
                self._error = None
            except Exception as exc:
                self._model = None
                self._tokenizer = None
                self._error = str(exc)
                raise
            return self.status()

    def _load_model(self, torch, auto_cls):
        if LLM_LOAD_IN_4BIT and torch.cuda.is_available():
            try:
                from transformers import BitsAndBytesConfig

                return auto_cls.from_pretrained(
                    LLM_MODEL_DIR,
                    quantization_config=BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_quant_type="nf4",
                    ),
                    device_map="auto",
                )
            except Exception:
                pass
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        kwargs = {"dtype": dtype}
        if torch.cuda.is_available():
            kwargs["device_map"] = "auto"
        return auto_cls.from_pretrained(LLM_MODEL_DIR, **kwargs)

    def unload(self) -> dict:
        with self._lock:
            self._model = None
            self._tokenizer = None
            self._error = None
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            return self.status()

    def complete(self, system: str, user: str, max_new_tokens: int | None = None) -> str:
        if self._model is None:
            self.load()
        import torch

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        prompt = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self._tokenizer(prompt, return_tensors="pt")
        device = next(self._model.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items()}
        tokens = max_new_tokens if max_new_tokens is not None else LLM_MAX_TOKENS
        with torch.inference_mode():
            generated = self._model.generate(
                **inputs,
                max_new_tokens=tokens,
                do_sample=LLM_TEMPERATURE > 0,
                temperature=LLM_TEMPERATURE if LLM_TEMPERATURE > 0 else None,
                pad_token_id=self._tokenizer.pad_token_id,
            )
        new_tokens = generated[0, inputs["input_ids"].shape[-1] :]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def _adapter_ready() -> bool:
    return os.path.isfile(os.path.join(LLM_ADAPTER_DIR, "adapter_config.json"))


def _maybe_adapter(model):
    if not _adapter_ready():
        return model
    from peft import PeftModel

    return PeftModel.from_pretrained(model, LLM_ADAPTER_DIR)


engine = LocalEngine()
