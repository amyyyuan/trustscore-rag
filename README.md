# Per-passage trust scores for retrieval-augmented generation

Code for the COMPSCI 764 (University of Auckland, S2 2026) research project on topic 5.1,
*Detecting adversarially retrieved contexts in RAG systems*.

The question: can a black-box, per-passage trust score built from lightweight signals separate
adversarially retrieved passages from clean ones, across both **fact poisoning** (PoisonedRAG-style
passages asserting a wrong answer) and **instruction injection** (BIPIA/RobustRAG-style
instructions hidden in passages), at a false-positive rate low enough to use?

## What the pipeline does

1. **Build retrieved sets** (`scripts/01_build_sets.py`). For every query, Contriever ranks the
   candidate passages; for the 100 target questions, poisoned passages are added to the candidate
   pool and compete on retriever score, so each attacked set records how many poison passages
   actually reached the top-k. Conditions: `clean`, `fact_prefix_m{1,5}`, `fact_noprefix_m{1,5}`,
   `inj_carrier_m{1,5}`, `inj_standalone_m{1,5}`, `inj_standalone_rep10_m1`, `random`
   (irrelevant-but-benign passages), and, after the paraphrase step, `fact_para_m*` and `inj_para_m*`.
2. **Compute signals** (`scripts/02_compute_signals.py`). Per passage: GPT-2 perplexity (mean and
   windowed), prompt-injection classifier probability, instruction cue words; per set: query
   similarity z-score, query overlap and query-copy length, nearest and mean embedding distance,
   2-means cluster tightness, NLI contradiction and entailment against the other passages. Also the
   TrustRAG stage-1 baseline flags. Everything is cached.
3. **Evaluate** (`scripts/03_evaluate.py`). Grouped cross-validation by question; one-feature and
   fused logistic-regression scorers (planted passages weighted so every attack condition counts
   equally); thresholds set on never-attacked clean passages; AUROC (pooled and per fold), TPR at 1%
   and 5% FPR with bootstrap CIs by question; feature ablations; train-on-one-family transfer;
   leave-one-recipe-out; per-template retrieval and detection rates; a length-only reference;
   false positives on random passages; TrustRAG stage-1 baseline (re-implemented from the
   released code, `defend_module.py`).
4. **Paraphrase** (`scripts/02b_paraphrase.py`, needs an LLM) rewrites the attacks for the adaptive
   conditions, then steps 1 to 3 are re-run to include them.
5. **End task** (`scripts/04_llm_endtask.py`, needs an LLM). Attack success, accuracy and abstention
   of the generator with no defence, the fused trust filter, the best single-signal filter, RobustRAG
   keyword aggregation and known-answer detection.
6. **Figures** (`scripts/05_figures.py`).

Two retrieval settings: `nq_beir` (PoisonedRAG's: BEIR Natural Questions corpus, Contriever) and
`nq_google` (RobustRAG's: NQ questions with Google-search snippets; duplicate snippets dropped,
items with fewer than k distinct snippets skipped, planted pages given the title of the top clean
page). `configs/default.yaml` holds every number that shapes the experiment.

Updating a results folder produced before 19 Oct 2026: `bash rerun_after_update.sh` (renames the
model caches, rebuilds `nq_google`, re-runs the detector experiments reusing cached model outputs).

## Reproduce

```bash
bash setup_mac.sh          # clones upstream repos, creates .venv, downloads models and data (once)
bash run_signals.sh        # detector experiments, no LLM needed (about an hour on an M1 Air)
export GROQ_API_KEY=...    # or any OpenAI-compatible endpoint, see configs/default.yaml
bash run_llm.sh            # paraphrases, adaptive conditions, end-task evaluation
```

Outputs land in `results/<setting>/tables` (CSV and Markdown) and `results/<setting>/figures`.
`tests/run_smoke.sh` runs the whole pipeline offline with tiny stand-in models to check the code.

Useful switches: `--settings nq_beir`, `--set signals.device=cpu`,
`--set signals.nli_model=cross-encoder/nli-deberta-v3-small` (faster NLI),
`python scripts/04_llm_endtask.py --max-questions 20 --defenses none filter_fused` (cheap first pass).

## Upstream work used

* PoisonedRAG (Zou, Geng, Wang, Jia; USENIX Security 2025): poison texts, stored Contriever
  retrieval results and the black-box passage recipe. https://github.com/sleeepeer/PoisonedRAG (MIT)
* RobustRAG (Xiang et al.; 2024): the NQ/Google-snippet data with poison passages, the prompt
  injection template and the keyword-aggregation defence re-implemented here.
  https://github.com/inspire-group/RobustRAG
* TrustRAG (Zhou et al.; 2025): stage-1 filtering re-implemented from the released code
  (https://github.com/HuichiZhou/TrustRAG, `defend_module.py`), which differs from the paper's summary.
* Known-answer detection (Liu et al.; USENIX Security 2024) and the windowed perplexity filter
  (Jain et al.; 2023) re-implemented from the papers.
* Models: facebook/contriever, openai-community/gpt2, cross-encoder/nli-deberta-v3-base,
  protectai/deberta-v3-base-prompt-injection-v2, BAAI/bge-base-en-v1.5 (Hugging Face).

Upstream repositories are cloned into `third_party/` at pinned commits by `setup_mac.sh`; nothing
from them is redistributed here.
