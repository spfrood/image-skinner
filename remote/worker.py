"""
Runs ON the RunPod GPU pod.
Watches /workspace/trigger.json, then executes:
  1. RVC voice conversion  (recording audio → character voice)
  2. Motion / pose transfer (recording video → animated sketch)
  3. A/V mux               (combine converted audio + animated video)
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

from loguru import logger

from rvc_inference import run_rvc
from motion_transfer import run_motion_transfer, wait_for_liveportrait

WORKSPACE = Path("/workspace")
TRIGGER_FILE = WORKSPACE / "trigger.json"
OUTPUT_DIR = WORKSPACE / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def process(config: dict) -> None:
    recording = WORKSPACE / config["recording"]
    sketch = WORKSPACE / config["sketch"]
    voice_model = WORKSPACE / config["voice_model"]
    voice_index = WORKSPACE / config["voice_index"]
    output_path = Path(config["output"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tmp_audio_in = WORKSPACE / "tmp_audio_in.wav"
    tmp_audio_out = WORKSPACE / "tmp_audio_out.wav"
    tmp_video = WORKSPACE / "tmp_video.mp4"

    # ── Step 1: Extract audio from recording ──────────────────────────────────
    logger.info("Extracting audio from recording…")
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(recording), "-vn", "-ar", "44100",
         "-ac", "2", str(tmp_audio_in)],
        check=True, capture_output=True,
    )

    # ── Step 2: RVC voice conversion ─────────────────────────────────────────
    logger.info("Running RVC voice conversion…")
    run_rvc(
        audio_in=str(tmp_audio_in),
        audio_out=str(tmp_audio_out),
        model_path=str(voice_model),
        index_path=str(voice_index),
        f0_method="rmvpe",          # best pitch accuracy
        f0_up_key=0,                # no pitch shift — keep creator's expression
        filter_radius=3,
        rms_mix_rate=0.25,
        protect=0.33,
    )

    # ── Step 3: Motion / pose transfer ────────────────────────────────────────
    logger.info("Running motion transfer…")
    run_motion_transfer(
        driver_video=str(recording),
        reference_image=str(sketch),
        output_video=str(tmp_video),
        character_prompt=config.get(
            "character_prompt",
            "a stylized animated character, full body, anime art style",
        ),
    )

    # ── Step 4: Mux converted audio onto animated video ───────────────────────
    logger.info("Muxing audio + video…")
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", str(tmp_video),
            "-i", str(tmp_audio_out),
            "-c:v", "copy",
            "-c:a", "aac",
            "-shortest",
            str(output_path),
        ],
        check=True, capture_output=True,
    )

    logger.success(f"Output written to {output_path}")


def watch_loop() -> None:
    # Block here until pod_setup.sh finishes installing LivePortrait + models.
    # This happens in the background while job assets are being uploaded.
    wait_for_liveportrait()
    logger.info("Worker ready. Watching for trigger.json…")
    while True:
        if TRIGGER_FILE.exists():
            try:
                config = json.loads(TRIGGER_FILE.read_text())
                TRIGGER_FILE.unlink()
                process(config)
            except Exception as exc:
                logger.exception(f"Processing failed: {exc}")
                (WORKSPACE / "error.txt").write_text(str(exc))
        time.sleep(2)


if __name__ == "__main__":
    watch_loop()
