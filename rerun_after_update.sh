#!/usr/bin/env bash
# One-off: apply the 19 Oct code update to a results folder produced by the earlier code, then
# re-run the detector experiments. Model outputs already computed are reused (the cache files are
# renamed to the new per-model names), so the re-run only scores passages that changed.
set -euo pipefail
cd "$(dirname "$0")"
# shellcheck disable=SC1091
source .venv/bin/activate
export HF_HUB_DISABLE_XET=1 TOKENIZERS_PARALLELISM=false PYTORCH_ENABLE_MPS_FALLBACK=1

echo "== 1/3 carrying over caches"
if [ -d results/cache ]; then
  cd results/cache
  mv_if() { if [ -f "$1" ] && [ ! -f "$2" ]; then mv "$1" "$2"; echo "   $1 -> $2"; fi; }
  mv_if emb_retrieval.pkl emb_retrieval_contriever_128.pkl
  mv_if emb_signals.pkl   emb_signals_contriever_512.pkl
  mv_if emb_trustrag.pkl  emb_trustrag_bge_base_en_v1_5_512.pkl
  mv_if ppl.pkl           ppl_gpt2.pkl
  mv_if inj.pkl           inj_deberta_v3_base_prompt_injection_v2.pkl
  mv_if nli.pkl           nli_nli_deberta_v3_base.pkl
  cd ../..
fi
rm -f results/*/tables/leave_one_condition_out.* results/*/tables/loco_* results/*/scores_loco_*.csv 2>/dev/null || true

echo "== 1b/3 NLI model (small cross-encoder; downloaded once)"
python - <<'PYSNIP'
from huggingface_hub import list_repo_files, snapshot_download
name = "cross-encoder/nli-MiniLM2-L6-H768"
top = [f for f in list_repo_files(name) if "/" not in f]
weights = ["*.safetensors"] if any(f.endswith(".safetensors") for f in top) else ["*.bin"]
snapshot_download(name, allow_patterns=["*.json", "*.txt", "*.model", "*.spm"] + weights, ignore_patterns=["*/*", "*.msgpack", "*.h5", "*.ot", "*.onnx"])
print("   ok")
PYSNIP

echo "== 2/3 data (nq_google is rebuilt without duplicate snippets; nq_beir is kept)"
python scripts/00_prepare_data.py --config configs/default.yaml

echo "== 3/3 detector experiments"
bash run_signals.sh
