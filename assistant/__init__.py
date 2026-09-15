"""Assistente clínico: LangGraph, tools e guardrails."""
from __future__ import annotations

import os
import sys

_WEB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web")
if _WEB not in sys.path:
    sys.path.insert(0, _WEB)
