"""Dataset preparation and loading.

Two retrieval settings are supported.

* ``nq_beir``   PoisonedRAG's setting: Natural Questions queries, the BEIR NQ Wikipedia
                corpus, Contriever retrieval. Poison texts and the stored top-100 retrieval
                results come from the PoisonedRAG repository.
* ``nq_google`` RobustRAG's setting: Natural Questions queries with Google-search snippets as
                the candidate passages. Poison texts come from the RobustRAG repository.

``prepare_*`` functions write a uniform layout under ``data/<setting>/``:

    queries.json        {qid: {"question", "is_target", "correct_answer", "incorrect_answer",
                               "adv_texts"}}
    corpus.jsonl        {"_id", "title", "text"} for every candidate passage
    candidates.json     {qid: [[doc_id, stored_score_or_null], ...]} ordered as stored upstream
    random.jsonl        (nq_beir only) random corpus passages for the irrelevant-but-benign check
"""

from __future__ import annotations

import io
import json
import random
import sys
import zipfile
from pathlib import Path

from .utils import iter_jsonl, read_json, write_json, write_jsonl

BEIR_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nq.zip"


# ----------------------------------------------------------------------------------------------
# Upstream files
# ----------------------------------------------------------------------------------------------

def poisonedrag_files(third_party: Path) -> tuple[Path, Path]:
    root = third_party / "PoisonedRAG"
    adv = root / "results" / "adv_targeted_results" / "nq.json"
    beir = root / "results" / "beir_results" / "nq-contriever.json"
    for p in (adv, beir):
        if not p.exists():
            raise FileNotFoundError(f"missing upstream file {p}; run setup first")
    return adv, beir


def robustrag_file(third_party: Path) -> Path:
    p = third_party / "RobustRAG" / "data" / "open_nq.json"
    if not p.exists():
        raise FileNotFoundError(f"missing upstream file {p}; run setup first")
    return p


# ----------------------------------------------------------------------------------------------
# nq_beir
# ----------------------------------------------------------------------------------------------

def select_beir_queries(adv_path: Path, beir_path: Path, n_target: int, n_extra: int, seed: int):
    adv = read_json(adv_path)
    results = read_json(beir_path)
    targets = list(adv.values())[:n_target]
    target_ids = [t["id"] for t in targets]
    pool = sorted(set(results.keys()) - set(target_ids))
    rng = random.Random(seed)
    extra_ids = rng.sample(pool, n_extra)
    return targets, target_ids, extra_ids, results


def _download(url: str, dest: Path) -> None:
    """Download to dest.part and rename on completion, so an interrupted transfer never leaves a
    truncated file at the final path."""
    import requests  # local import so the module loads without requests

    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        done = 0
        with open(part, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r  downloading {done / 1e6:.0f} / {total / 1e6:.0f} MB", end="", flush=True)
    print()
    if total and done < total:
        part.unlink(missing_ok=True)
        raise RuntimeError(f"download incomplete ({done} of {total} bytes)")
    part.replace(dest)


def _iter_zip_member(zip_path: Path, member_suffix: str):
    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if n.endswith(member_suffix)]
        if not names:
            raise FileNotFoundError(f"{member_suffix} not found inside {zip_path}")
        with zf.open(names[0]) as raw:
            for line in io.TextIOWrapper(raw, encoding="utf-8"):
                line = line.strip()
                if line:
                    yield json.loads(line)


def extract_beir_subset(zip_path: Path, needed_ids: set[str], extra_query_ids: set[str],
                        n_random: int, seed: int, out_dir: Path) -> dict[str, str]:
    """Stream the BEIR NQ corpus once, keeping needed passages and a random sample.

    Returns the text of the extra queries (id -> question).
    """
    rng = random.Random(seed)
    kept: dict[str, dict] = {}
    reservoir: list[dict] = []
    seen = 0
    print("  scanning corpus.jsonl (2.7M passages, a few minutes)...", flush=True)
    for doc in _iter_zip_member(zip_path, "corpus.jsonl"):
        did = doc["_id"]
        if did in needed_ids:
            kept[did] = {"_id": did, "title": doc.get("title", ""), "text": doc.get("text", "")}
        # reservoir sampling for the random benign set
        if len(reservoir) < n_random:
            reservoir.append({"_id": did, "title": doc.get("title", ""), "text": doc.get("text", "")})
        else:
            j = rng.randrange(seen + 1)
            if j < n_random:
                reservoir[j] = {"_id": did, "title": doc.get("title", ""), "text": doc.get("text", "")}
        seen += 1
        if seen % 500000 == 0:
            print(f"    {seen} passages scanned, {len(kept)} kept", flush=True)
    missing = needed_ids - set(kept)
    if missing:
        print(f"  warning: {len(missing)} needed passages not found in corpus", file=sys.stderr)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(kept.values(), out_dir / "corpus.jsonl")
    # drop random passages that are also candidates, to keep the sets disjoint
    write_jsonl([d for d in reservoir if d["_id"] not in needed_ids], out_dir / "random.jsonl")
    queries = {}
    for q in _iter_zip_member(zip_path, "queries.jsonl"):
        if q["_id"] in extra_query_ids:
            queries[q["_id"]] = q["text"]
    return queries


def _extract_beir_subset_hf(needed_ids: set[str], extra_query_ids: set[str], n_random: int,
                            seed: int, out_dir: Path) -> dict[str, str]:
    """Fallback when the UKP host is unreachable: stream BeIR/nq from the Hugging Face hub."""
    from datasets import load_dataset  # optional dependency

    rng = random.Random(seed)
    kept, reservoir, seen = {}, [], 0
    corpus = load_dataset("BeIR/nq", "corpus", split="corpus", streaming=True)
    for doc in corpus:
        did = str(doc["_id"])
        row = {"_id": did, "title": doc.get("title", ""), "text": doc.get("text", "")}
        if did in needed_ids:
            kept[did] = row
        if len(reservoir) < n_random:
            reservoir.append(row)
        else:
            j = rng.randrange(seen + 1)
            if j < n_random:
                reservoir[j] = row
        seen += 1
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(kept.values(), out_dir / "corpus.jsonl")
    write_jsonl([d for d in reservoir if d["_id"] not in needed_ids], out_dir / "random.jsonl")
    queries = {}
    for q in load_dataset("BeIR/nq", "queries", split="queries", streaming=True):
        if str(q["_id"]) in extra_query_ids:
            queries[str(q["_id"])] = q["text"]
    return queries


def prepare_nq_beir(cfg: dict, third_party: Path, data_dir: Path, zip_path: Path | None = None) -> None:
    ds = cfg["datasets"]["nq_beir"]
    out_dir = data_dir / "nq_beir"
    if (out_dir / "candidates.json").exists():
        print("nq_beir already prepared, skipping")
        return
    adv_path, beir_path = poisonedrag_files(third_party)
    targets, target_ids, extra_ids, results = select_beir_queries(
        adv_path, beir_path, ds["n_target"], ds["n_extra_clean"], cfg["seed"])
    pool = ds["candidate_pool"]
    candidates = {}
    needed: set[str] = set()
    for qid in target_ids + extra_ids:
        ranked = list(results[qid].items())[:pool]
        candidates[qid] = [[did, float(score)] for did, score in ranked]
        needed.update(did for did, _ in ranked)
    print(f"nq_beir: {len(target_ids)} target + {len(extra_ids)} extra queries, {len(needed)} passages needed")

    zip_path = zip_path or (data_dir / "downloads" / "nq.zip")
    extra_questions: dict[str, str]
    if zip_path.exists() and not zipfile.is_zipfile(zip_path):
        print(f"  {zip_path} is not a valid zip (interrupted download?); deleting it and downloading again")
        zip_path.unlink()
    if not zip_path.exists():
        try:
            print(f"downloading {BEIR_URL} (this is the large step)")
            _download(BEIR_URL, zip_path)
        except Exception as exc:  # noqa: BLE001
            print(f"  download failed ({exc}); trying the Hugging Face mirror BeIR/nq")
            zip_path = None
    if zip_path is not None:
        extra_questions = extract_beir_subset(zip_path, needed, set(extra_ids), ds["n_random_passages"],
                                              cfg["seed"], out_dir)
    else:
        extra_questions = _extract_beir_subset_hf(needed, set(extra_ids), ds["n_random_passages"],
                                                  cfg["seed"], out_dir)

    queries = {}
    for t in targets:
        queries[t["id"]] = {
            "question": t["question"],
            "is_target": True,
            "correct_answer": t["correct answer"],
            "incorrect_answer": t["incorrect answer"],
            "adv_texts": t["adv_texts"],
        }
    for qid in extra_ids:
        if qid not in extra_questions:
            print(f"  warning: no text for extra query {qid}", file=sys.stderr)
            continue
        queries[qid] = {"question": extra_questions[qid], "is_target": False}
    write_json(queries, out_dir / "queries.json")
    write_json({qid: candidates[qid] for qid in queries}, out_dir / "candidates.json")
    print(f"nq_beir prepared under {out_dir}")


# ----------------------------------------------------------------------------------------------
# nq_google
# ----------------------------------------------------------------------------------------------

NQ_GOOGLE_PREP_VERSION = 2   # bump when the preparation rules change, so stale data is rebuilt


def prepare_nq_google(cfg: dict, third_party: Path, data_dir: Path) -> None:
    """RobustRAG's released NQ items with Google-search snippets.

    Two preparation rules (version 2): exact duplicate snippet texts inside one item are dropped
    (the released file has them for about a third of the items; duplicates would give identical
    retriever scores and distort every set-level signal), and only items with at least k distinct
    snippets are used, so that every clean set has k passages (six of the first 400 items have
    fewer than ten). Items are taken in file order.
    """
    ds = cfg["datasets"]["nq_google"]
    out_dir = data_dir / "nq_google"
    version_file = out_dir / "prep_version.json"
    if (out_dir / "candidates.json").exists() and version_file.exists() and \
            read_json(version_file).get("version") == NQ_GOOGLE_PREP_VERSION:
        print("nq_google already prepared, skipping")
        return
    items = read_json(robustrag_file(third_party))
    n_target, n_extra = ds["n_target"], ds["n_extra_clean"]
    k = cfg["retrieval"]["k"]
    usable = []
    n_dup, n_short = 0, 0
    for item in items:
        seen, ctx = set(), []
        for c in item["context"]:
            if "text" not in c:
                continue
            key = c["text"].strip()
            if key in seen:
                n_dup += 1
                continue
            seen.add(key)
            ctx.append(c)
        if len(ctx) < k:
            n_short += 1
            continue
        usable.append({**item, "context": ctx})
        if len(usable) == n_target + n_extra:
            break
    if len(usable) < n_target + n_extra:
        raise ValueError(f"open_nq.json has only {len(usable)} items with at least {k} distinct snippets; need {n_target + n_extra}")
    print(f"nq_google: dropped {n_dup} duplicate snippets, skipped {n_short} items with fewer than {k} distinct snippets")
    queries, candidates, corpus = {}, {}, []
    for idx, item in enumerate(usable):
        qid = f"g{item['id']}"
        is_target = idx < n_target
        entry = {"question": item["question"], "is_target": is_target}
        if is_target:
            answers = list(item["correct answer"]) + list(item.get("expanded answer", []))
            entry.update({
                "correct_answer": answers,            # list: any match counts, as in RobustRAG
                "incorrect_answer": item["incorrect answer"],
                "adv_texts": item["incorrect_context"],
            })
        queries[qid] = entry
        cands = []
        for j, c in enumerate(item["context"]):
            if "text" not in c:
                continue
            did = f"{qid}_c{j}"
            corpus.append({"_id": did, "title": c.get("title", ""), "text": c["text"]})
            cands.append([did, None])
        candidates[qid] = cands
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(queries, out_dir / "queries.json")
    write_json(candidates, out_dir / "candidates.json")
    write_jsonl(corpus, out_dir / "corpus.jsonl")
    write_json({"version": NQ_GOOGLE_PREP_VERSION, "duplicates_dropped": n_dup, "items_skipped_short": n_short},
               version_file)
    print(f"nq_google prepared under {out_dir}: {len(queries)} queries, {len(corpus)} snippets")


# ----------------------------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------------------------

class Setting:
    """A prepared retrieval setting: queries, candidate passages and their text.

    Two strings exist for every passage. ``retrieval_text`` (title, space, text) is what the
    retriever embeds, as in PoisonedRAG's released retrieval run and BEIR's convention.
    ``passage_text`` is what the generator and the detector see: the passage text alone for
    ``nq_beir`` (PoisonedRAG hands the LLM ``corpus[id]["text"]``) and title + newline + snippet for
    ``nq_google`` (RobustRAG's ``include_title``).
    """

    def __init__(self, name: str, data_dir: Path, include_title_in_passage: bool | None = None):
        self.name = name
        self.dir = data_dir / name
        self.queries: dict[str, dict] = read_json(self.dir / "queries.json")
        self.candidates: dict[str, list] = read_json(self.dir / "candidates.json")
        self.corpus: dict[str, dict] = {d["_id"]: d for d in iter_jsonl(self.dir / "corpus.jsonl")}
        rp = self.dir / "random.jsonl"
        self.random_passages: list[dict] = list(iter_jsonl(rp)) if rp.exists() else []
        if include_title_in_passage is None:
            include_title_in_passage = name == "nq_google"
        self.include_title = include_title_in_passage

    @staticmethod
    def _join(title: str, text: str, sep: str) -> str:
        title, text = (title or "").strip(), (text or "").strip()
        return f"{title}{sep}{text}" if title else text

    def passage_text(self, doc_id: str) -> str:
        d = self.corpus[doc_id]
        return self._join(d.get("title", "") if self.include_title else "", d.get("text", ""), "\n")

    def retrieval_text(self, doc_id: str) -> str:
        d = self.corpus[doc_id]
        return self._join(d.get("title", ""), d.get("text", ""), " ")

    def random_texts(self, d: dict) -> tuple[str, str]:
        """(passage_text, retrieval_text) for a random passage record."""
        return (self._join(d.get("title", "") if self.include_title else "", d.get("text", ""), "\n"),
                self._join(d.get("title", ""), d.get("text", ""), " "))

    @property
    def target_ids(self) -> list[str]:
        return [q for q, v in self.queries.items() if v["is_target"]]

    @property
    def extra_ids(self) -> list[str]:
        return [q for q, v in self.queries.items() if not v["is_target"]]
