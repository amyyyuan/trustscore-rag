"""Generator access through any OpenAI-compatible chat endpoint (Groq, OpenAI, Ollama, vLLM).

Every call is cached on disk keyed by (model, prompt, temperature, max_tokens), so re-running a
script never repeats a request. A token-bucket rate limiter keeps free-tier limits.
"""

from __future__ import annotations

import json
import os
import random
import re
import time
from pathlib import Path

from .utils import clean_str, sha1


class LLMClient:
    def __init__(self, cfg: dict, cache_dir: Path, model: str | None = None):
        self.cfg = cfg
        self.backend = cfg.get("backend", "openai_compatible")
        self.model = model or cfg["model"]
        self.temperature = float(cfg.get("temperature", 0.0))
        self.max_tokens = int(cfg.get("max_tokens", 64))
        self.cache_path = cache_dir / f"{re.sub(r'[^A-Za-z0-9_.-]', '_', self.model)}.jsonl"
        self.cache: dict[str, str] = {}
        if self.cache_path.exists():
            with open(self.cache_path) as f:
                for line in f:
                    if line.strip():
                        row = json.loads(line)
                        self.cache[row["key"]] = row["response"]
        self.rpm = float(cfg.get("requests_per_minute", 25))
        self.tpm = float(cfg.get("tokens_per_minute", 5500))
        self._req_times: list[float] = []
        self._tok_log: list[tuple[float, int]] = []
        self.max_retries = int(cfg.get("max_retries", 8))
        self._params = {"len_key": "max_tokens", "use_temperature": True}
        self.n_calls = 0
        self.n_cached = 0
        self._client = None
        if self.backend == "openai_compatible":
            from openai import OpenAI  # local import: not needed for the fake backend

            api_key = os.environ.get(cfg.get("api_key_env", "OPENAI_API_KEY"), "")
            if not api_key and "localhost" not in cfg.get("base_url", ""):
                raise RuntimeError(f"set the {cfg.get('api_key_env')} environment variable")
            self._client = OpenAI(base_url=cfg.get("base_url"), api_key=api_key or "none")
        elif self.backend == "fake":
            self.fake_rng = random.Random(0)
        else:
            raise ValueError(self.backend)

    # ------------------------------------------------------------------------------------------
    def _key(self, prompt: str, max_tokens: int | None) -> str:
        return sha1(json.dumps([self.model, prompt, self.temperature, max_tokens or self.max_tokens]))

    def _wait_for_budget(self, est_tokens: int) -> None:
        now = time.time()
        self._req_times = [t for t in self._req_times if now - t < 60]
        self._tok_log = [(t, n) for t, n in self._tok_log if now - t < 60]
        while len(self._req_times) >= self.rpm or sum(n for _, n in self._tok_log) + est_tokens > self.tpm:
            time.sleep(1.0)
            now = time.time()
            self._req_times = [t for t in self._req_times if now - t < 60]
            self._tok_log = [(t, n) for t, n in self._tok_log if now - t < 60]

    def _call(self, prompt: str, max_tokens: int) -> str:
        est = len(prompt) // 4 + max_tokens          # about four characters per token for English prose
        if est > self.tpm:
            raise ValueError(f"a single prompt ({est} estimated tokens) exceeds llm.tokens_per_minute ({self.tpm:.0f}); "
                             "raise the limit or shorten the passages")
        params = dict(self._params)
        for attempt in range(self.max_retries):
            self._wait_for_budget(est)
            try:
                self._req_times.append(time.time())
                self._tok_log.append((time.time(), est))
                kwargs = {params["len_key"]: max_tokens}
                if params["use_temperature"]:
                    kwargs["temperature"] = self.temperature
                resp = self._client.chat.completions.create(
                    model=self.model, messages=[{"role": "user", "content": prompt}], **kwargs)
                self._params = params
                return (resp.choices[0].message.content or "").strip()
            except Exception as exc:  # noqa: BLE001 - adapt parameters or retry on transport/rate errors
                msg = str(exc)
                if "max_completion_tokens" in msg and params["len_key"] == "max_tokens":
                    params["len_key"] = "max_completion_tokens"      # newer OpenAI models
                    continue
                if "temperature" in msg and params["use_temperature"]:
                    params["use_temperature"] = False                # models that only accept the default
                    continue
                wait = min(60, 2 ** attempt) + random.random()
                print(f"  [llm] {type(exc).__name__}: {msg[:120]}; retry in {wait:.0f}s", flush=True)
                time.sleep(wait)
        raise RuntimeError("LLM call failed after retries")

    def complete(self, prompt: str, max_tokens: int | None = None, fake_hint: dict | None = None) -> str:
        key = self._key(prompt, max_tokens)
        if key in self.cache:
            self.n_cached += 1
            return self.cache[key]
        if self.backend == "fake":
            text = self._fake(prompt, fake_hint or {})
        else:
            text = self._call(prompt, max_tokens or self.max_tokens)
        self.cache[key] = text
        self.n_calls += 1
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "a") as f:
            f.write(json.dumps({"key": key, "response": text, "model": self.model}) + "\n")
        return text

    def _fake(self, prompt: str, hint: dict) -> str:
        """Deterministic stand-in used by the smoke test: follows a visible target answer most of
        the time, otherwise returns the correct answer or abstains."""
        if "secret" in hint:                       # known-answer probe: echo the key unless hijacked
            return "hijacked" if hint.get("hijack") else hint["secret"]
        if hint.get("paraphrase"):
            return hint["paraphrase"]
        target, correct = hint.get("target"), hint.get("correct")
        r = self.fake_rng.random()
        if target and clean_str(target) in clean_str(prompt) and r < 0.85:
            return str(target)
        if correct and r < 0.9:
            return str(correct)
        return "I don't know"
