#!/usr/bin/env bash
# Detector experiments that need no LLM: build retrieved sets, compute signals, evaluate, draw figures.
# Takes roughly an hour on an M1 MacBook Air (most of it NLI pairs); progress is printed.
# Re-running is cheap because every model output is cached under results/cache.
set -euo pipefail
cd "$(dirname "$0")"
export HF_HUB_DISABLE_XET=1          # plain HTTPS downloads from the Hub (see setup_mac.sh)
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ENABLE_MPS_FALLBACK=1   # an op missing on the Apple GPU falls back to the CPU instead of aborting
# shellcheck disable=SC1091
source .venv/bin/activate
CONFIG=${CONFIG:-configs/default.yaml}
python scripts/01_build_sets.py --config "$CONFIG" "$@"
python scripts/02_compute_signals.py --config "$CONFIG" "$@"
python scripts/03_evaluate.py --config "$CONFIG" "$@"
python scripts/05_figures.py --config "$CONFIG" "$@"
echo "Done. Tables: results/*/tables, figures: results/*/figures"
