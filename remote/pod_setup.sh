#!/usr/bin/env bash
# Runs once on pod boot.
# Installs LivePortrait + AnimateDiff/ControlNet models, then hands off to worker.py.
set -euo pipefail

WORKSPACE=/workspace
LP_ROOT=$WORKSPACE/LivePortrait
LOG=$WORKSPACE/setup.log

exec > >(tee -a "$LOG") 2>&1
echo "[$(date -u)] Pod setup started."

# ── 1. Base Python deps ────────────────────────────────────────────────────────
pip install -q \
    huggingface_hub \
    loguru \
    soundfile \
    "diffusers>=0.28.0" \
    accelerate \
    xformers \
    transformers \
    imageio \
    imageio-ffmpeg \
    mediapipe \
    opencv-python-headless

# ── 2. Clone + install LivePortrait ───────────────────────────────────────────
if [ ! -d "$LP_ROOT/.git" ]; then
    echo "[$(date -u)] Cloning LivePortrait…"
    git clone --depth 1 https://github.com/KwaiVision/LivePortrait.git "$LP_ROOT"
fi
echo "[$(date -u)] Installing LivePortrait requirements…"
pip install -q -r "$LP_ROOT/requirements.txt"

# ── 3. Download all pretrained weights in one Python session ──────────────────
echo "[$(date -u)] Downloading model weights…"
python3 - <<'PYEOF'
from huggingface_hub import snapshot_download, hf_hub_download
import os

HF_CACHE = "/workspace/hf_cache"

# LivePortrait weights
print("  → KwaiVision/LivePortrait")
snapshot_download(
    repo_id="KwaiVision/LivePortrait",
    local_dir="/workspace/LivePortrait/pretrained_weights",
    ignore_patterns=["*.md", "*.gitattributes"],
    cache_dir=HF_CACHE,
)

# AnimateDiff motion adapter v1.5-2
print("  → guoyww/animatediff-motion-adapter-v1-5-2")
snapshot_download(
    repo_id="guoyww/animatediff-motion-adapter-v1-5-2",
    local_dir=f"{HF_CACHE}/animatediff-motion-adapter",
    cache_dir=HF_CACHE,
)

# ControlNet OpenPose (SD1.5)
print("  → lllyasviel/sd-controlnet-openpose")
snapshot_download(
    repo_id="lllyasviel/sd-controlnet-openpose",
    local_dir=f"{HF_CACHE}/sd-controlnet-openpose",
    cache_dir=HF_CACHE,
)

# VAE (higher-quality SD1.5 VAE)
print("  → stabilityai/sd-vae-ft-mse")
snapshot_download(
    repo_id="stabilityai/sd-vae-ft-mse",
    local_dir=f"{HF_CACHE}/sd-vae-ft-mse",
    cache_dir=HF_CACHE,
)

# Base SD1.5 checkpoint (anime-friendly)
print("  → SG161222/Realistic_Vision_V5.1_noVAE")
snapshot_download(
    repo_id="SG161222/Realistic_Vision_V5.1_noVAE",
    local_dir=f"{HF_CACHE}/Realistic_Vision_V5.1_noVAE",
    cache_dir=HF_CACHE,
    ignore_patterns=["*.ckpt", "*.safetensors.bak"],
)

# IP-Adapter SD1.5 weights
print("  → h94/IP-Adapter (sd15)")
hf_hub_download(
    repo_id="h94/IP-Adapter",
    filename="models/ip-adapter_sd15.bin",
    local_dir=f"{HF_CACHE}/IP-Adapter",
    cache_dir=HF_CACHE,
)

print("All weights downloaded.")
PYEOF

echo "[$(date -u)] Setup complete."
touch "$WORKSPACE/.lp_ready"

# ── 4. Wait for worker.py upload, then start it ───────────────────────────────
echo "[$(date -u)] Waiting for worker.py…"
while [ ! -f "$WORKSPACE/worker.py" ]; do sleep 3; done
echo "[$(date -u)] Starting worker."
python3 "$WORKSPACE/worker.py"
