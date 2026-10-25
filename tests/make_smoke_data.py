#!/usr/bin/env python3
"""Fabricate the offline inputs for the smoke test: a fake BEIR zip whose passages are made-up text
for exactly the document ids the pipeline will ask for, next to the real PoisonedRAG and RobustRAG
files (which must already be cloned under third_party/).

Usage: python tests/make_smoke_data.py --third-party third_party --out data_smoke --config configs/smoke.yaml
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trustscore.config import load_config  # noqa: E402
from trustscore.data import select_beir_queries  # noqa: E402

WORDS = ("season episode series film album river mountain city county president election battle treaty "
         "university company founded released published population capital language island party "
         "championship record player team coach stadium novel author painter composer symphony").split()


def fake_text(rng: random.Random, seed_words: list[str]) -> str:
    n = rng.randint(40, 90)
    words = [rng.choice(WORDS + seed_words) for _ in range(n)]
    sents, i = [], 0
    while i < n:
        j = min(n, i + rng.randint(6, 14))
        sents.append(" ".join(words[i:j]).capitalize() + ".")
        i = j
    return " ".join(sents)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--third-party", default="third_party")
    ap.add_argument("--out", default="data_smoke")
    ap.add_argument("--config", default="configs/smoke.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    tp = Path(args.third_party)
    out = Path(args.out)
    adv = tp / "PoisonedRAG" / "results" / "adv_targeted_results" / "nq.json"
    beir = tp / "PoisonedRAG" / "results" / "beir_results" / "nq-contriever.json"
    ds = cfg["datasets"]["nq_beir"]
    targets, target_ids, extra_ids, results = select_beir_queries(adv, beir, ds["n_target"], ds["n_extra_clean"], cfg["seed"])
    needed = set()
    for qid in target_ids + extra_ids:
        needed.update(list(results[qid].keys())[: ds["candidate_pool"]])
    rng = random.Random(1)
    zip_path = out / "downloads" / "nq.zip"
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    question_of = {t["id"]: t["question"] for t in targets}
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        lines = []
        for did in sorted(needed):
            lines.append(json.dumps({"_id": did, "title": f"Article {did}", "text": fake_text(rng, [])}))
        for i in range(300):   # filler passages so the random sample has something to draw from
            lines.append(json.dumps({"_id": f"filler{i}", "title": "", "text": fake_text(rng, [])}))
        zf.writestr("nq/corpus.jsonl", "\n".join(lines) + "\n")
        qlines = []
        for qid in sorted(results.keys()):
            text = question_of.get(qid, f"what is the {rng.choice(WORDS)} of the {rng.choice(WORDS)} {qid}")
            qlines.append(json.dumps({"_id": qid, "text": text}))
        zf.writestr("nq/queries.jsonl", "\n".join(qlines) + "\n")
    print(f"fake BEIR zip with {len(needed)} needed passages written to {zip_path}")
    for sub in ("nq_beir", "nq_google"):
        shutil.rmtree(out / sub, ignore_errors=True)


if __name__ == "__main__":
    main()
