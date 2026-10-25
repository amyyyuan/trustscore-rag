#!/usr/bin/env bash
# Experiments that need a generator LLM (any OpenAI-compatible endpoint):
#   a. paraphrase the attacks (adaptive variants), then rebuild sets/signals/tables to include them
#   b. end-task evaluation: attack success and accuracy with each defence
# Set the API key first, e.g.   export GROQ_API_KEY=...   (see configs/default.yaml, section llm)
# To use another endpoint:  bash run_llm.sh --set llm.base_url=https://api.openai.com/v1 llm.api_key_env=OPENAI_API_KEY llm.model=<model>
set -euo pipefail
cd "$(dirname "$0")"
export HF_HUB_DISABLE_XET=1          # plain HTTPS downloads from the Hub (see setup_mac.sh)
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ENABLE_MPS_FALLBACK=1
# shellcheck disable=SC1091
source .venv/bin/activate
CONFIG=${CONFIG:-configs/default.yaml}
# keep the detector tables from before the paraphrased conditions were added (training changes once they exist)
for d in results/*/tables; do
  [ -d "$d" ] && [ ! -d "${d}_before_paraphrase" ] && cp -r "$d" "${d}_before_paraphrase"
done
python scripts/02b_paraphrase.py --config "$CONFIG" "$@"
python scripts/01_build_sets.py --config "$CONFIG"
python scripts/02_compute_signals.py --config "$CONFIG"
python scripts/03_evaluate.py --config "$CONFIG"
python scripts/04_llm_endtask.py --config "$CONFIG" "$@"
python scripts/05_figures.py --config "$CONFIG"
echo "Done. End-task tables: results/*/tables/endtask_*.csv"
