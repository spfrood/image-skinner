#!/usr/bin/env bash
# Runs once on pod boot.  Installs LivePortrait + downloads pretrained weights,
# then blocks until worker.py is uploaded and hands off to it.
set -euo pipefail

WORKSPACE=/workspace
LP_ROOT=$WORKSPACE/LivePortrait
LOG=$WORKSPACE/setup.log

exec > >(tee -a "$LOG") 2>&1
echo "[$(date -u)] Pod setup started."

# ── 1. Base Python deps ────────────────────────────────────────────────────────
pip install -q huggingface_hub loguru soundfile

# ── 2. Clone LivePortrait ──────────────────────────────────────────────────────
if [ ! -d "$LP_ROOT/.git" ]; then
    echo "[$(date -u)] Cloning LivePortrait…"
    git clone --depth 1 https://github.com/KwaiVision/LivePortrait.git "$LP_ROOT"
fi

# ── 3. Install LivePortrait Python requirements ────────────────────────────────
echo "[$(date -u)] Installing LivePortrait requirements…"
pip install -q -r "$LP_ROOT/requirements.txt"

# ── 4. Download pretrained weights from HuggingFace ───────────────────────────
echo "[$(date -u)] Downloading pretrained weights (KwaiVision/LivePortrait)…"
python3 - <<'PYEOF'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="KwaiVision/LivePortrait",
    local_dir="/workspace/LivePortrait/pretrained_weights",
    ignore_patterns=["*.md", "*.gitattributes"],
)
PYEOF

echo "[$(date -u)] Setup complete."
touch "$WORKSPACE/.lp_ready"

# ── 5. Wait for worker.py to be uploaded, then start it ───────────────────────
echo "[$(date -u)] Waiting for worker.py upload…"
while [ ! -f "$WORKSPACE/worker.py" ]; do sleep 3; done
echo "[$(date -u)] Starting worker."
python3 "$WORKSPACE/worker.py"
