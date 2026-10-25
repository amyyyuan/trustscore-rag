#!/usr/bin/env python3
"""Step 2: compute every trust signal for every passage of every set, plus the TrustRAG stage-1
baseline flags. All model outputs are cached, so re-running after adding conditions only costs
the new passages.

Usage: python scripts/02_compute_signals.py --config configs/default.yaml
Output: results/<setting>/signals.csv, results/<setting>/latency.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trustscore.baselines import trustrag_stage1  # noqa: E402
from trustscore.config import load_config, resolve  # noqa: E402
from trustscore.retrieval import Embedder  # noqa: E402
from trustscore.signals import SignalComputer  # noqa: E402
from trustscore.utils import read_jsonl, set_seed, write_json  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--settings", nargs="*", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    set_seed(cfg["seed"])
    results = resolve(cfg, "results")
    cache = results / "cache"
    names = args.settings or [n for n, d in cfg["datasets"].items() if d["enabled"]]
    computer = SignalComputer(cfg, cache)
    tr_cfg = cfg.get("baselines", {}).get("trustrag", {})
    tr_embedder = Embedder(tr_cfg.get("embedding_model", "BAAI/bge-base-en-v1.5"), cfg["signals"].get("device", "auto"),
                           512, cfg["signals"].get("batch_size", 16), cache / "emb_trustrag.pkl",
                           pooling=tr_cfg.get("pooling", "cls"))
    for name in names:
        sets = read_jsonl(results / name / "sets.jsonl")
        print(f"[{name}] computing signals for {len(sets)} sets")
        rows = computer.compute(sets)
        # TrustRAG stage 1 flags (separate encoder, as in the paper); all texts embedded in one batch
        uniq = sorted({p["text"] for s in sets for p in s["passages"]})
        print(f"  TrustRAG baseline: embedding {len(uniq)} unique passages with {tr_cfg.get('embedding_model')}")
        vec_of = dict(zip(uniq, tr_embedder.embed(uniq)))
        flag_of = {}
        for i, s in enumerate(sets):
            texts = [p["text"] for p in s["passages"]]
            vecs = np.stack([vec_of[t] for t in texts])
            flags = trustrag_stage1(texts, vecs, tr_cfg.get("cos_threshold", 0.88), tr_cfg.get("rouge_threshold", 0.25))
            for p, f in zip(s["passages"], flags):
                flag_of[(s["set_id"], p["doc_id"])] = bool(f)
            if (i + 1) % 500 == 0:
                print(f"    TrustRAG flags {i + 1}/{len(sets)} sets", flush=True)
        for r in rows:
            r["trustrag_flag"] = flag_of[(r["set_id"], r["doc_id"])]
        df = pd.DataFrame(rows)
        out = results / name / "signals.csv"
        df.to_csv(out, index=False)
        write_json(computer.latency_report(len(sets)), results / name / "latency.json")
        n_pos = int(df["is_poison"].sum())
        print(f"[{name}] wrote {len(df)} passage rows ({n_pos} poisoned) to {out}")
        print(df.groupby("condition")[["log_ppl", "inj_prob", "sim_z", "q_lcs", "nn_dist", "nli_contra_max"]]
              .mean().round(3).to_string())


if __name__ == "__main__":
    main()
