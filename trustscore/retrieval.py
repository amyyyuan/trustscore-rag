"""Dense retrieval with Contriever (mean pooling, dot-product score), plus an embedding cache.

PoisonedRAG scores passages with facebook/contriever and a dot product; the stored top-100 results
in its repository were produced that way. We recompute scores for every candidate with the same
model so that clean and poisoned passages are scored by one code path, and we check our scores
against the stored ones as a sanity test (scripts/01_build_sets.py prints the agreement).
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from .utils import pick_device, sha1


def model_slug(name: str) -> str:
    """File-name-safe tag for a model name, so caches never mix two models' outputs."""
    return "".join(ch if ch.isalnum() else "_" for ch in name.split("/")[-1])[:60]


class EmbeddingCache:
    def __init__(self, path: Path):
        self.path = path
        self.store: dict[str, np.ndarray] = {}
        if path.exists():
            with open(path, "rb") as f:
                self.store = pickle.load(f)
        self.dirty = 0

    def get(self, text: str) -> np.ndarray | None:
        return self.store.get(sha1(text))

    def put(self, text: str, vec: np.ndarray) -> None:
        self.store[sha1(text)] = vec.astype(np.float32)
        self.dirty += 1
        if self.dirty >= 2000:
            self.save()

    def save(self) -> None:
        if not self.dirty:            # nothing new: do not rewrite the file
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        with open(tmp, "wb") as f:
            pickle.dump(self.store, f)
        tmp.replace(self.path)
        self.dirty = 0


class Embedder:
    """Mean-pooled transformer embeddings (Contriever recipe)."""

    def __init__(self, model_name: str, device: str = "auto", max_length: int = 512, batch_size: int = 16,
                 cache_path: Path | None = None, pooling: str = "mean"):
        import time

        import torch
        from transformers import AutoModel, AutoTokenizer

        self.device = pick_device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()
        self.max_length = max_length
        self.batch_size = batch_size
        self.pooling = pooling
        if cache_path is not None:    # one cache per (model, input length)
            cache_path = cache_path.with_name(f"{cache_path.stem}_{model_slug(model_name)}_{max_length}{cache_path.suffix}")
        self.cache = EmbeddingCache(cache_path) if cache_path else None
        self.torch = torch
        self._time = time
        self.seconds = 0.0      # wall-clock spent embedding (cache misses only)
        self.count = 0

    @staticmethod
    def _mean_pool(last_hidden, mask):
        mask = mask.unsqueeze(-1).to(last_hidden.dtype)
        return (last_hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-6)

    def embed(self, texts: list[str]) -> np.ndarray:
        if len(texts) == 0:
            return np.zeros((0, self.model.config.hidden_size), dtype=np.float32)
        out = [None] * len(texts)
        todo = []
        for i, t in enumerate(texts):
            vec = self.cache.get(t) if self.cache else None
            if vec is None:
                todo.append(i)
            else:
                out[i] = vec
        for start in range(0, len(todo), self.batch_size):
            idx = todo[start:start + self.batch_size]
            batch = [texts[i] for i in idx]
            t0 = self._time.perf_counter()
            enc = self.tokenizer(batch, padding=True, truncation=True, max_length=self.max_length,
                                 return_tensors="pt").to(self.device)
            with self.torch.no_grad():
                hidden = self.model(**enc).last_hidden_state
                if self.pooling == "cls":
                    vecs = hidden[:, 0, :].float().cpu().numpy()
                else:
                    vecs = self._mean_pool(hidden, enc["attention_mask"]).float().cpu().numpy()
            if np.isnan(vecs).any() and self.device != "cpu":
                print(f"  [embed] NaN on {self.device}; switching the encoder to CPU", flush=True)
                self.device = "cpu"
                self.model = self.model.to("cpu")
                enc = enc.to("cpu")
                with self.torch.no_grad():
                    hidden = self.model(**enc).last_hidden_state
                    vecs = (hidden[:, 0, :] if self.pooling == "cls"
                            else self._mean_pool(hidden, enc["attention_mask"])).float().cpu().numpy()
            self.seconds += self._time.perf_counter() - t0
            self.count += len(batch)
            for i, v in zip(idx, vecs):
                out[i] = v
                if self.cache:
                    self.cache.put(texts[i], v)
            if (start // self.batch_size) % 50 == 0 and len(todo) > self.batch_size * 50:
                print(f"    embedded {start + len(idx)}/{len(todo)}", flush=True)
        if self.cache:
            self.cache.save()
        return np.stack(out).astype(np.float32)


def score(query_vec: np.ndarray, passage_vecs: np.ndarray, kind: str = "dot") -> np.ndarray:
    if kind == "dot":
        return passage_vecs @ query_vec
    if kind == "cos":
        qn = query_vec / (np.linalg.norm(query_vec) + 1e-9)
        pn = passage_vecs / (np.linalg.norm(passage_vecs, axis=1, keepdims=True) + 1e-9)
        return pn @ qn
    raise ValueError(kind)
