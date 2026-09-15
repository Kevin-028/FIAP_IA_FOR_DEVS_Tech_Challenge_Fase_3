"""Pacote `llm` na raiz — o código vive em `web/llm`."""
from __future__ import annotations

import os

_WEB_LLM = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "web", "llm"))
__path__ = [_WEB_LLM]
