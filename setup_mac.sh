#!/usr/bin/env bash
# One-time setup on a Mac (Apple silicon or Intel) or Linux box with internet access.
#   1. clones the two upstream repositories at pinned commits (attack texts, retrieval results, data)
#   2. creates a virtual environment and installs the Python dependencies
#   3. downloads the small models used by the signals (about 2.5 GB in total)
#   4. prepares both retrieval settings (downloads the BEIR NQ corpus once, keeps only what is needed)
# Re-running is safe: every step skips work that is already done.
set -euo pipefail
cd "$(dirname "$0")"

# Use plain HTTPS downloads from the Hugging Face Hub. The newer "Xet" transfer backend returned
# 404 errors and crashed the interpreter on a Mac during the first setup run (Oct 2026).
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false

PY=${PYTHON:-python3}
if ! $PY -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)'; then
  echo "Python 3.10 or newer is required (found: $($PY --version)). On a Mac: brew install python@3.12 and re-run with PYTHON=python3.12"
  exit 1
fi

echo "== 1/4 upstream repositories"
mkdir -p third_party
if [ ! -d third_party/PoisonedRAG/.git ]; then
  git clone --quiet https://github.com/sleeepeer/PoisonedRAG third_party/PoisonedRAG
fi
git -C third_party/PoisonedRAG checkout --quiet f660d72174f06b13fae5163ce656e7b235db858f
if [ ! -d third_party/RobustRAG/.git ]; then
  git clone --quiet https://github.com/inspire-group/RobustRAG third_party/RobustRAG
fi
git -C third_party/RobustRAG checkout --quiet 9bc35b2fa5fa7d1088383b2789ec19a512d316b1

echo "== 2/4 Python environment"
if [ ! -d .venv ]; then
  $PY -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt
pip install --quiet "spacy>=3.7" https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl || \
  echo "   (spaCy install failed; the RobustRAG baseline will fall back to a simple keyword extractor)"
pip install --quiet "datasets>=2.19" || echo "   (datasets install failed; only needed if the BEIR download host is down)"

echo "== 3/4 models (cached under ~/.cache/huggingface)"
python - <<'EOF'
import time
from huggingface_hub import list_repo_files, snapshot_download

MODELS = ["facebook/contriever", "openai-community/gpt2", "cross-encoder/nli-MiniLM2-L6-H768",
          "protectai/deberta-v3-base-prompt-injection-v2", "BAAI/bge-base-en-v1.5"]
SMALL = ["*.json", "*.txt", "*.model", "*.spm"]
# Only top-level files are wanted; subfolders such as onnx/ hold alternative exports we never load.
IGNORE = ["*/*", "*.msgpack", "*.h5", "*.ot", "*.onnx"]

for name in MODELS:
    for attempt in range(1, 4):
        try:
            top = [f for f in list_repo_files(name) if "/" not in f]
            # one copy of the weights: safetensors when the repo has it, otherwise the PyTorch .bin
            weights = ["*.safetensors"] if any(f.endswith(".safetensors") for f in top) else ["*.bin"]
            print("  ", name, "(%s)" % weights[0][2:], flush=True)
            snapshot_download(name, allow_patterns=SMALL + weights, ignore_patterns=IGNORE)
            break
        except Exception as exc:  # noqa: BLE001 - transient Hub errors: wait and retry
            if attempt == 3:
                raise
            print("   download problem (%s: %s); retrying in %ds" % (type(exc).__name__, str(exc)[:100], 15 * attempt), flush=True)
            time.sleep(15 * attempt)
EOF

echo "== 4/4 data"
python scripts/00_prepare_data.py --config configs/default.yaml

echo
echo "Setup complete. Next: bash run_signals.sh"
