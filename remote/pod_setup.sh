#!/usr/bin/env bash
# Runs once on pod boot.
# Installs LivePortrait + RVC + AnimateDiff/ControlNet models, then hands off to worker.py.
set -euo pipefail

WORKSPACE=/workspace
LP_ROOT=$WORKSPACE/LivePortrait
RVC_ROOT=$WORKSPACE/Retrieval-based-Voice-Conversion-WebUI
HF_CACHE=$WORKSPACE/hf_cache
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
    opencv-python-headless \
    praat-parselmouth \
    pyworld

# ── 2. Clone + install LivePortrait ───────────────────────────────────────────
if [ ! -d "$LP_ROOT/.git" ]; then
    echo "[$(date -u)] Cloning LivePortrait…"
    git clone --depth 1 https://github.com/KwaiVision/LivePortrait.git "$LP_ROOT"
fi
echo "[$(date -u)] Installing LivePortrait requirements…"
pip install -q -r "$LP_ROOT/requirements.txt"

# ── 3. Clone + install RVC WebUI ──────────────────────────────────────────────
if [ ! -d "$RVC_ROOT/.git" ]; then
    echo "[$(date -u)] Cloning RVC WebUI…"
    git clone --depth 1 \
        https://github.com/RVC-Boss/Retrieval-based-Voice-Conversion-WebUI.git \
        "$RVC_ROOT"
fi
echo "[$(date -u)] Installing RVC dependencies…"
# faiss: prefer GPU build; fall back to CPU
pip install -q faiss-gpu 2>/dev/null || pip install -q faiss-cpu
# fairseq has known build quirks — disable build isolation and try two versions
pip install -q "fairseq==0.12.2" --no-build-isolation 2>/dev/null || \
    pip install -q "fairseq" --no-build-isolation 2>/dev/null || \
    echo "WARNING: fairseq install failed — voice conversion unavailable"
# Remaining RVC deps, skipping torch/torchaudio (already present on base image)
pip install -q \
    "scipy>=1.7" \
    librosa \
    "numba>=0.57" \
    tensorboard 2>/dev/null || true

# ── 4. Download pretrained weights ────────────────────────────────────────────
# Use cache_dir only (no local_dir) for diffusion models so that
# from_pretrained(repo_id) + HF_HOME=$HF_CACHE resolves them automatically.
# LivePortrait is the exception: its code hard-codes pretrained_weights/.
echo "[$(date -u)] Downloading model weights…"
python3 - <<PYEOF
from huggingface_hub import snapshot_download, hf_hub_download
import os

HF_CACHE = "$HF_CACHE"
os.makedirs(HF_CACHE, exist_ok=True)

repos = [
    ("guoyww/animatediff-motion-adapter-v1-5-2", None),
    ("lllyasviel/sd-controlnet-openpose",         None),
    ("stabilityai/sd-vae-ft-mse",                 None),
    ("SG161222/Realistic_Vision_V5.1_noVAE",      ["*.ckpt", "*.safetensors.bak"]),
]
for repo_id, ignore in repos:
    print(f"  → {repo_id}")
    snapshot_download(
        repo_id=repo_id,
        cache_dir=HF_CACHE,
        ignore_patterns=ignore,
    )

# IP-Adapter: only one file needed
print("  → h94/IP-Adapter (sd15)")
hf_hub_download(
    repo_id="h94/IP-Adapter",
    filename="models/ip-adapter_sd15.bin",
    cache_dir=HF_CACHE,
)

# LivePortrait: code expects weights at pretrained_weights/ inside the repo
print("  → KwaiVision/LivePortrait")
snapshot_download(
    repo_id="KwaiVision/LivePortrait",
    local_dir="$LP_ROOT/pretrained_weights",
    cache_dir=HF_CACHE,
    ignore_patterns=["*.md", "*.gitattributes"],
)

print("All weights downloaded.")
PYEOF

echo "[$(date -u)] Setup complete."
touch "$WORKSPACE/.lp_ready"

# ── 5. Wait for worker.py upload, then start it ───────────────────────────────
echo "[$(date -u)] Waiting for worker.py…"
while [ ! -f "$WORKSPACE/worker.py" ]; do sleep 3; done
echo "[$(date -u)] Starting worker."
python3 "$WORKSPACE/worker.py"
