"""Small shared helpers: seeding, JSON I/O, timing, text normalisation."""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:  # torch is optional for the pure-numpy steps
        pass


def read_json(path: str | os.PathLike) -> Any:
    with open(path) as f:
        return json.load(f)


def write_json(obj: Any, path: str | os.PathLike, indent: int | None = 1) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=indent, ensure_ascii=False)


def read_jsonl(path: str | os.PathLike) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def iter_jsonl(path: str | os.PathLike) -> Iterator[dict]:
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(rows: Iterable[dict], path: str | os.PathLike) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


@contextmanager
def timer(label: str, log: dict | None = None):
    """Context manager that records wall-clock seconds under `label`."""
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    if log is not None:
        log[label] = log.get(label, 0.0) + elapsed
    print(f"[time] {label}: {elapsed:.1f}s", flush=True)


def clean_str(s: Any) -> str:
    """PoisonedRAG's answer normalisation: strip, drop a trailing full stop, lower-case."""
    s = str(s).strip()
    if len(s) > 1 and s[-1] == ".":
        s = s[:-1]
    return s.lower()


_TOKEN_RE = re.compile(r"[a-z0-9]+")

STOPWORDS = set(
    """a an the of in on at to for from by with and or is are was were be been being am do does did
    what when where who whom which why how many much does this that these those it its as into
    about than then there their they them he she his her we you your our i me my us not no
    can could would should will shall may might has have had having over under up down out off
    if but so very also just only own same such too s t""".split()
)


def tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def content_tokens(text: str) -> list[str]:
    return [t for t in tokens(text) if t not in STOPWORDS]


def pick_device(requested: str = "auto") -> str:
    if requested != "auto":
        return requested
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"
