"""Configuração do LLM local (Hugging Face Transformers).

  FIAP_LLM_PROVIDER   — huggingface | fallback | ollama (compat) | llama_cpp (alias)
  LLM_MODEL_DIR       — pasta do checkpoint (safetensors)
  LLM_LOAD_IN_4BIT    — 1 = 4-bit; 0 = fp16 na GPU
"""
import os


def _load_dotenv():
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    path = os.path.join(root, ".env")
    if not os.path.isfile(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key, val = key.strip(), val.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = val


_load_dotenv()


def _resolve_provider() -> str:
    p = os.environ.get("FIAP_LLM_PROVIDER", "").lower().strip()
    if p == "llama_cpp":
        return "huggingface"
    return p if p in ("huggingface", "fallback", "ollama") else "huggingface"


LLM_PROVIDER = _resolve_provider()
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gpt-oss")
LLM_TIMEOUT = int(os.environ.get("FIAP_LLM_TIMEOUT", "180"))
LLM_TEMPERATURE = float(os.environ.get("FIAP_LLM_TEMPERATURE", "0.25"))
LLM_MAX_TOKENS = int(os.environ.get("FIAP_LLM_MAX_TOKENS", "512"))
MODELS_DIR = os.environ.get("LLM_MODELS_DIR", r"C:\workspace\Modelos")
LLM_MODEL_DIR = os.environ.get(
    "LLM_MODEL_DIR",
    os.path.join(MODELS_DIR, "Qwen2.5-3B-Instruct"),
)
LLM_LOAD_IN_4BIT = os.environ.get("LLM_LOAD_IN_4BIT", "1").strip() not in ("0", "false", "False")
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LLM_ADAPTER_DIR = os.environ.get(
    "LLM_ADAPTER_DIR",
    os.path.join(_ROOT, "models", "adapter"),
)
LLM_GGUF_PATH = LLM_MODEL_DIR

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
INTERPRETATION_LOG = os.path.join(LOG_DIR, "llm_interpretations.jsonl")
