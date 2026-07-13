"""Optional local-LLM polish for analyst briefs via Ollama — free, opt-in, zero new deps.

Uses only ``urllib`` against a locally running Ollama server (https://ollama.com). The LLM
is a *rephraser*, not an author: it receives the finished template brief plus the dossier
facts and is instructed to rewrite fluently without adding numbers or claims. Any failure
(server down, timeout, bad response) returns ``None`` and the caller shows the template
brief unchanged — the app never depends on Ollama being installed.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Optional

DEFAULT_HOST = "http://localhost:11434"
DEFAULT_MODEL = "llama3.2:3b"

_POLISH_PROMPT = """You are a financial writing editor. Rewrite the trading brief below into fluent,
professional analyst prose. STRICT RULES:
- Keep every number, price level, and ticker exactly as given. Do not invent, add, or change any
  figure, claim, or recommendation.
- Keep the same section structure and roughly the same length.
- Keep the final disclaimer sentence verbatim.
Facts (JSON, for reference only — do not add facts that are not in the brief):
{facts}

Brief to rewrite:
{brief}
"""


class OllamaClient:
    def __init__(self, host: str = DEFAULT_HOST, model: str = DEFAULT_MODEL,
                 timeout: float = 60.0) -> None:
        self.host = (host or DEFAULT_HOST).rstrip("/")
        self.model = model or DEFAULT_MODEL
        self.timeout = timeout

    def available(self) -> bool:
        """Whether an Ollama server is answering at ``host`` (fast check, never raises)."""
        try:
            req = urllib.request.Request(f"{self.host}/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=3.0) as resp:
                return resp.status == 200
        except Exception:
            return False

    def polish(self, brief_markdown: str, facts_json: str = "{}") -> Optional[str]:
        """Rewrite ``brief_markdown`` fluently. Returns ``None`` on any failure so the
        caller falls back to the template text."""
        payload = json.dumps({
            "model": self.model,
            "prompt": _POLISH_PROMPT.format(facts=facts_json, brief=brief_markdown),
            "stream": False,
            "options": {"temperature": 0.3},
        }).encode("utf-8")
        try:
            req = urllib.request.Request(
                f"{self.host}/api/generate", data=payload,
                headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            text = (body.get("response") or "").strip()
            # A degenerate rewrite (empty or suspiciously short) is worse than the template.
            if len(text) < max(80, len(brief_markdown) // 4):
                return None
            return text
        except Exception:
            return None
