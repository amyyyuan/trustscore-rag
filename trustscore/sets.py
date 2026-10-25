"""Build the retrieved sets that the detector and the generator see.

A *set* is the ordered list of k passages a RAG system would hand to the generator for one query
under one condition. Clean sets exist for every query; attacked conditions exist for the target
queries only. Poisoned passages compete for a slot on their retriever score ("retrieved"
placement), so each attacked set records how many of its m poison passages actually made it into
the top-k. The "forced" placement is kept as an ablation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .attacks import Passage, carrier_passages, condition_catalogue, fact_passages, standalone_passages
from .data import Setting
from .retrieval import Embedder, score
from .utils import read_json, write_jsonl


def _rank(passages: list[Passage]) -> list[Passage]:
    ordered = sorted(passages, key=lambda p: -float(p.score))
    for r, p in enumerate(ordered, start=1):
        p.rank = r
    return ordered


def build_sets(setting: Setting, cfg: dict, embedder: Embedder, out_path: Path) -> dict:
    k = cfg["retrieval"]["k"]
    score_kind = cfg["retrieval"]["score"]
    placement = cfg["attacks"]["placement"]
    inj_cfg = cfg["attacks"]["injection"]
    para_path = setting.dir / "paraphrases.json"
    paraphrases = read_json(para_path) if para_path.exists() else None
    conditions = condition_catalogue(cfg["attacks"]["m_values"], inj_cfg["repeat_variant"],
                                     with_paraphrase=paraphrases is not None)

    qids = list(setting.queries.keys())
    questions = [setting.queries[q]["question"] for q in qids]
    print(f"[{setting.name}] embedding {len(qids)} queries")
    q_vecs = embedder.embed(questions)

    # clean candidate scores
    clean_pool: dict[str, list[Passage]] = {}
    stored, ours = [], []
    all_doc_ids = sorted({did for cands in setting.candidates.values() for did, _ in cands})
    print(f"[{setting.name}] embedding {len(all_doc_ids)} candidate passages")
    doc_vecs = embedder.embed([setting.retrieval_text(d) for d in all_doc_ids])
    vec_of = {d: v for d, v in zip(all_doc_ids, doc_vecs)}
    for qi, qid in enumerate(qids):
        pool = []
        for did, stored_score in setting.candidates[qid]:
            s = float(score(q_vecs[qi], vec_of[did][None, :], score_kind)[0])
            pool.append(Passage(doc_id=did, text=setting.passage_text(did), score=s))
            if stored_score is not None:
                stored.append(stored_score)
                ours.append(s)
        clean_pool[qid] = pool
    agreement = None
    if stored:
        stored_a, ours_a = np.array(stored), np.array(ours)
        agreement = {"n": len(stored), "pearson": float(np.corrcoef(stored_a, ours_a)[0, 1]),
                     "mean_abs_diff": float(np.mean(np.abs(stored_a - ours_a)))}
        print(f"[{setting.name}] stored vs recomputed scores: {agreement}")

    rows = []
    planted_log: list[dict] = []
    target_ids = setting.target_ids
    retrieval_stats: dict[str, list[int]] = {}

    def emit(qid: str, condition: str, family: str | None, m: int, passages: list[Passage],
             n_poison_total: int) -> None:
        q = setting.queries[qid]
        rows.append({
            "set_id": f"{setting.name}|{condition}|{qid}", "setting": setting.name, "condition": condition,
            "qid": qid, "question": q["question"], "is_target": q["is_target"], "family": family, "m": m,
            "k": k, "correct_answer": q.get("correct_answer"), "incorrect_answer": q.get("incorrect_answer"),
            "n_poison_in_topk": sum(p.is_poison for p in passages), "n_poison_total": n_poison_total,
            "passages": [p.to_dict() for p in passages],
        })

    # clean sets for every query
    for qid in qids:
        top = _rank([Passage(**{**p.__dict__}) for p in clean_pool[qid]])[:k]
        emit(qid, "clean", None, 0, top, 0)

    # attacked sets for target queries
    for cond in conditions:
        name, m = cond["name"], cond["m"]
        for qi, qid in enumerate(target_ids):
            q = setting.queries[qid]
            question, target = q["question"], q["incorrect_answer"]
            pool = [Passage(**{**p.__dict__}) for p in clean_pool[qid]]
            if cond["kind"] == "fact":
                poison = fact_passages(qid, question, q["adv_texts"], m, cond["prefix"], name,
                                       paraphrases if cond.get("paraphrase") else None)
            elif cond["kind"] == "carrier":
                ranked_clean = _rank(pool)
                carriers = [(p.doc_id, p.text) for p in ranked_clean[:m]]
                poison = carrier_passages(qid, qi, question, target, carriers, m, name,
                                          position=inj_cfg["carrier_position"],
                                          paraphrases=paraphrases if cond.get("paraphrase") else None)
                edited = {p.carrier_id for p in poison}
                pool = [p for p in pool if p.doc_id not in edited]      # the edited page replaces the original
                for p in poison:                                        # retriever sees the edited page as the
                    rec = setting.corpus[p.carrier_id]                  # corpus does: title, space, text, instruction
                    instr = p.extra["instruction"]
                    base = setting.retrieval_text(p.carrier_id)
                    p.extra["retrieval_text"] = f"{base} {instr}" if inj_cfg["carrier_position"] == "end" else f"{instr} {base}"
                    p.extra["title"] = rec.get("title", "")
            elif cond["kind"] == "standalone":
                poison = standalone_passages(qid, qi, question, target, m, name, repeat=cond.get("repeat", 1))
            else:
                raise ValueError(cond)
            if cond["kind"] in ("fact", "standalone") and setting.include_title:
                # In settings where every passage carries a page title (nq_google), a planted page
                # gets one too, otherwise the missing title line alone would give it away. The attacker
                # reuses the title of the highest-ranked clean page for the question.
                top_title = (setting.corpus[_rank(list(pool))[0].doc_id].get("title") or "").strip()
                for p in poison:
                    p.extra["retrieval_text"] = f"{top_title} {p.text}" if top_title else p.text
                    p.extra["title"] = top_title
                    if top_title:
                        p.text = f"{top_title}\n{p.text}"
            p_vecs = embedder.embed([p.extra.get("retrieval_text", p.text) for p in poison])
            q_vec = q_vecs[qids.index(qid)]
            for p, v in zip(poison, p_vecs):
                p.score = float(score(q_vec, v[None, :], score_kind)[0])
            if placement == "retrieved":
                top = _rank(pool + poison)[:k]
            elif placement == "forced":
                top = _rank(_rank(pool)[: k - m] + poison)
            else:
                raise ValueError(placement)
            emit(qid, name, cond["family"], m, top, len(poison))
            retrieval_stats.setdefault(name, []).append(sum(p.is_poison for p in top))
            in_top = {p.doc_id for p in top}
            for p in poison:
                planted_log.append({"setting": setting.name, "condition": name, "qid": qid, "doc_id": p.doc_id,
                                    "template_id": p.template_id, "score": p.score, "rank": p.rank,
                                    "retrieved": p.doc_id in in_top})

    write_jsonl(rows, out_path)
    write_jsonl(planted_log, out_path.with_name("planted_log.jsonl"))
    summary = {name: {"mean_poison_in_topk": float(np.mean(v)), "frac_sets_with_poison": float(np.mean([x > 0 for x in v]))}
               for name, v in retrieval_stats.items()}
    for name, s in summary.items():
        print(f"[{setting.name}] {name}: mean poison in top-{k} = {s['mean_poison_in_topk']:.2f}, "
              f"sets with any poison = {s['frac_sets_with_poison']:.2f}")
    return {"n_sets": len(rows), "conditions": [c["name"] for c in conditions], "retrieval": summary,
            "score_agreement": agreement}
