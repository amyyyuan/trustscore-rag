#!/usr/bin/env bash
# Offline end-to-end check with tiny stand-in models and fabricated data (no downloads, no API).
# Needs third_party/PoisonedRAG and third_party/RobustRAG (setup_mac.sh step 1) and the Python deps.
set -euo pipefail
cd "$(dirname "$0")/.."
python tests/make_standin_models.py data_smoke/standin
python tests/make_smoke_data.py --third-party third_party --out data_smoke --config configs/smoke.yaml
python scripts/00_prepare_data.py --config configs/smoke.yaml --keep-zip
python scripts/01_build_sets.py --config configs/smoke.yaml
python scripts/02_compute_signals.py --config configs/smoke.yaml
python scripts/03_evaluate.py --config configs/smoke.yaml
python scripts/02b_paraphrase.py --config configs/smoke.yaml
python scripts/01_build_sets.py --config configs/smoke.yaml
python scripts/02_compute_signals.py --config configs/smoke.yaml
python scripts/03_evaluate.py --config configs/smoke.yaml
python scripts/04_llm_endtask.py --config configs/smoke.yaml
python scripts/05_figures.py --config configs/smoke.yaml
echo "smoke test passed"
