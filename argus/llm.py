"""LLM backends — the models behind synthesis, the judge, and the optimizer.

Argus optimizes ONLY via an LLM. There is no deterministic fallback. A backend exposes
`complete_text(system, user, *, model)` and `complete_json(...)`; everything else is
backend-agnostic, so the same code runs against:

  * "anthropic" — the Anthropic API (needs ANTHROPIC_API_KEY + the `anthropic` SDK)
  * "local"     — a local model server speaking the OpenAI chat API (e.g. Ollama)
  * "agent"     — handoff: an external agent supplies the output

Three roles, three models (each overridable from `.env`):
  synthesis  — the answer writer inside the system under test (cheap; default Haiku)
  judge      — grades the answer against the key            (strong; default Sonnet)
  optimizer  — the brain that tunes the knobs               (strongest; default Opus)
"""

from __future__ import annotations

import json
import os
import re
import urllib.request
from pathlib import Path

DEFAULT_MODELS = {
    "synthesis": "claude-haiku-4-5-20251001",
    "judge": "claude-sonnet-4-6",
    "optimizer": "claude-opus-4-8",
}
_ROLE_ENV = {"synthesis": "SYNTHESIS_MODEL", "judge": "JUDGE_MODEL", "optimizer": "OPTIMIZER_MODEL"}


class NoBrainConfigured(RuntimeError):
    """No LLM backend is available. There is no rule-based fallback by design."""


def load_dotenv() -> None:
    """Populate os.environ from a `.env` file (cwd, package dir, or its parent).

    Dependency-free. Existing environment variables win, so a real shell export overrides
    the file. Lines are `KEY=value`; `#` comments and blank lines are ignored.
    """
    seen = set()
    for base in (Path.cwd(), Path(__file__).resolve().parent, Path(__file__).resolve().parent.parent):
        env_path = base / ".env"
        if env_path in seen or not env_path.is_file():
            continue
        seen.add(env_path)
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def model_for(role: str) -> str:
    """Resolve the model id for a role, honouring a `.env`/env override."""
    load_dotenv()
    return os.environ.get(_ROLE_ENV[role], DEFAULT_MODELS[role])


# --- Anthropic API ----------------------------------------------------------------

def anthropic_available() -> bool:
    load_dotenv()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return False
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return False
    return True


class AnthropicBackend:
    def __init__(self) -> None:
        import anthropic

        load_dotenv()
        self.client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment

    def complete_text(self, system: str, user: str, *, model: str,
                      max_tokens: int = 600, temperature: float = 0.0) -> str:
        # `temperature` is intentionally not forwarded — newer models (e.g. Opus 4.8) reject it.
        # Reproducibility comes from the response cache instead.
        msg = self.client.messages.create(
            model=model, max_tokens=max_tokens,
            system=system, messages=[{"role": "user", "content": user}],
        )
        return "".join(b.text for b in msg.content if b.type == "text")

    def complete_json(self, system: str, user: str, *, model: str,
                      max_tokens: int = 1200, temperature: float = 0.0) -> dict:
        return _extract_json(self.complete_text(system, user, model=model,
                                                max_tokens=max_tokens, temperature=temperature))


# --- Local model server (OpenAI-compatible, e.g. Ollama) --------------------------

class LocalBackend:
    """Talks to a local OpenAI-compatible chat endpoint (Ollama serves one at /v1).

    Env: OLLAMA_HOST (default http://localhost:11434). No API key, no per-call cost.
    """

    def __init__(self, host: str | None = None) -> None:
        load_dotenv()
        self.host = (host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")

    def complete_text(self, system: str, user: str, *, model: str,
                      max_tokens: int = 600, temperature: float = 0.0) -> str:
        body = json.dumps({
            "model": model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "stream": False,
            "options": {"num_predict": max_tokens, "temperature": temperature},
        }).encode()
        req = urllib.request.Request(
            f"{self.host}/v1/chat/completions", data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=180) as resp:  # noqa: S310 — local, trusted host
            data = json.loads(resp.read())
        return data["choices"][0]["message"]["content"]

    def complete_json(self, system: str, user: str, *, model: str,
                      max_tokens: int = 1200, temperature: float = 0.0) -> dict:
        return _extract_json(self.complete_text(system, user, model=model,
                                                max_tokens=max_tokens, temperature=temperature))


def make_backend(brain: str):
    """Return the requested LLM backend, or raise NoBrainConfigured with guidance."""
    if brain == "anthropic":
        if not anthropic_available():
            raise NoBrainConfigured(
                "Anthropic brain needs ANTHROPIC_API_KEY (loaded from .env) and `pip install anthropic`.")
        return AnthropicBackend()
    if brain == "local":
        return LocalBackend()  # connectivity is checked on first call
    if brain == "agent":
        raise NoBrainConfigured(
            "The 'agent' brain is driven by an external agent loop, not the in-process optimizer.")
    raise ValueError(f"unknown brain: {brain!r} (expected anthropic | local | agent)")


def _extract_json(text: str) -> dict:
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else text
    start, end = candidate.find("{"), candidate.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON object in model reply: {text[:200]!r}")
    return json.loads(candidate[start : end + 1])
