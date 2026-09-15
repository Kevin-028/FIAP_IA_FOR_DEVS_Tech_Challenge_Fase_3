"""Baixa o LLM para C:\\workspace\\Modelos."""
from __future__ import annotations

import os
import sys

from huggingface_hub import snapshot_download
from huggingface_hub.errors import GatedRepoError, HfHubHTTPError

def _load_project_env() -> None:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(root, ".env")
    if not os.path.isfile(path):
        return
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key, val = key.strip(), val.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = val


_load_project_env()
DEST_ROOT = os.environ.get("LLM_MODELS_DIR", r"C:\workspace\Modelos")
CANDIDATES = (
    ("meta-llama/Llama-3.2-3B-Instruct", "Llama-3.2-3B-Instruct"),
    ("Qwen/Qwen2.5-3B-Instruct", "Qwen2.5-3B-Instruct"),
)


def _token() -> str | None:
    token = (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or "").strip()
    return token or None


def download(repo_id: str, folder: str) -> str:
    dest = os.path.join(DEST_ROOT, folder)
    os.makedirs(dest, exist_ok=True)
    path = snapshot_download(
        repo_id=repo_id,
        local_dir=dest,
        token=_token(),
        ignore_patterns=["*.msgpack", "flax_model*", "tf_model*", "rust_model*", "*.gguf"],
    )
    print(path)
    return path


def main() -> str:
    os.makedirs(DEST_ROOT, exist_ok=True)
    errors: list[str] = []
    for repo_id, folder in CANDIDATES:
        try:
            return download(repo_id, folder)
        except (GatedRepoError, HfHubHTTPError, OSError) as exc:
            errors.append(f"{repo_id}: {exc}")
            print(f"aviso: {repo_id} indisponível ({exc})", file=sys.stderr)
    raise SystemExit("Nenhum modelo baixado.\n" + "\n".join(errors))


if __name__ == "__main__":
    main()
