"""
Motion / Pose Transfer — LivePortrait backend.

Architecture:
  Face animation  : LivePortrait drives the character sketch with the creator's
                    facial expressions and head pose (the primary visual output).
  Body tracking   : MediaPipe Holistic extracts a full-body skeleton from the
                    driver video and stores it alongside the output.  Full-body
                    warping via AnimateDiff + ControlNet DWPose is the natural
                    next stage but is out-of-scope for this phase.

LivePortrait works as a video pipeline, not frame-by-frame, so we hand it the
full driver video and reference image in one call and let it produce the output.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from loguru import logger


LIVEPORTRAIT_ROOT = Path("/workspace/LivePortrait")
LIVEPORTRAIT_READY_FLAG = Path("/workspace/.lp_ready")


# ── LivePortrait readiness ─────────────────────────────────────────────────────

def wait_for_liveportrait(timeout_s: int = 1200, poll_s: int = 10) -> None:
    """Block until the pod startup script finishes installing LivePortrait."""
    if LIVEPORTRAIT_READY_FLAG.exists():
        return
    logger.info(
        f"Waiting for LivePortrait setup to complete "
        f"(up to {timeout_s // 60} min)…"
    )
    elapsed = 0
    while elapsed < timeout_s:
        if LIVEPORTRAIT_READY_FLAG.exists():
            logger.success("LivePortrait ready.")
            return
        time.sleep(poll_s)
        elapsed += poll_s
    raise TimeoutError(
        f"LivePortrait was not ready after {timeout_s}s. "
        "Check /workspace/setup.log on the pod."
    )


# ── Face animation via LivePortrait ───────────────────────────────────────────

def run_liveportrait(
    driver_video: str,
    reference_image: str,
    output_dir: str,
    output_stem: str,
    flag_relative_motion: bool = True,
    flag_pasteback: bool = True,
    flag_stitching: bool = True,
) -> Path:
    """
    Invoke LivePortrait's inference.py to animate the reference sketch with
    the driver's facial expressions and head pose.

    flag_relative_motion  – transfer expression *deltas* rather than absolute
                            pose; preserves the sketch's native look.
    flag_pasteback        – composite the animated face region back onto the
                            full source image (essential for non-portrait crops).
    flag_stitching        – smooth the boundary between animated and static
                            regions to avoid hard seams.
    """
    wait_for_liveportrait()

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(LIVEPORTRAIT_ROOT / "inference.py"),
        "-s", reference_image,
        "-d", driver_video,
        "--output-dir", str(out_path),
    ]
    if flag_relative_motion:
        cmd.append("--flag-relative-motion")
    if flag_pasteback:
        cmd.append("--flag-pasteback")
    if flag_stitching:
        cmd.append("--flag-stitching")

    logger.info(f"LivePortrait: {Path(reference_image).name} ← {Path(driver_video).name}")
    result = subprocess.run(
        cmd,
        cwd=str(LIVEPORTRAIT_ROOT),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        logger.error(result.stderr[-2000:])
        raise RuntimeError(f"LivePortrait failed (exit {result.returncode})")

    # LivePortrait writes: <output_dir>/<driving_stem>--<source_stem>.mp4
    # Locate the produced file and rename to our canonical stem.
    candidates = sorted(out_path.glob("*.mp4"))
    if not candidates:
        raise FileNotFoundError(
            f"LivePortrait produced no .mp4 in {out_path}. "
            f"stderr: {result.stderr[-1000:]}"
        )
    produced = candidates[-1]
    final = out_path / f"{output_stem}.mp4"
    produced.rename(final)
    logger.success(f"LivePortrait output: {final}")
    return final


# ── Body skeleton extraction (MediaPipe) ──────────────────────────────────────
# Body motion data is extracted here and saved as a companion JSON alongside
# the output video. Full-body synthesis (AnimateDiff + ControlNet DWPose) would
# consume this data in a future phase.

def extract_body_skeleton(driver_video: str, output_json: str) -> int:
    """
    Extract body + hand landmarks for every frame and write to a compact JSON.
    Returns the frame count.
    """
    import json
    import mediapipe as mp

    mp_holistic = mp.solutions.holistic
    frames_data: list[dict] = []

    cap = cv2.VideoCapture(driver_video)
    with mp_holistic.Holistic(
        min_detection_confidence=0.5, min_tracking_confidence=0.5,
        model_complexity=1,
    ) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = holistic.process(rgb)
            frame_entry: dict = {}
            if res.pose_landmarks:
                frame_entry["pose"] = [
                    {"x": lm.x, "y": lm.y, "z": lm.z, "v": lm.visibility}
                    for lm in res.pose_landmarks.landmark
                ]
            if res.left_hand_landmarks:
                frame_entry["left_hand"] = [
                    {"x": lm.x, "y": lm.y, "z": lm.z}
                    for lm in res.left_hand_landmarks.landmark
                ]
            if res.right_hand_landmarks:
                frame_entry["right_hand"] = [
                    {"x": lm.x, "y": lm.y, "z": lm.z}
                    for lm in res.right_hand_landmarks.landmark
                ]
            frames_data.append(frame_entry)
    cap.release()

    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(output_json).write_text(json.dumps(frames_data), encoding="utf-8")
    logger.info(f"Body skeleton: {len(frames_data)} frames → {output_json}")
    return len(frames_data)


# ── Main entry point ───────────────────────────────────────────────────────────

def run_motion_transfer(
    driver_video: str,
    reference_image: str,
    output_video: str,
) -> None:
    """
    Primary interface called by worker.py.

    Produces:
      <output_video>          – LivePortrait-animated character video (no audio)
      <output_video>.body.json – per-frame body skeleton for future body-warp phase
    """
    output_dir = str(Path(output_video).parent)
    output_stem = Path(output_video).stem

    # Face animation — the main deliverable
    run_liveportrait(
        driver_video=driver_video,
        reference_image=reference_image,
        output_dir=output_dir,
        output_stem=output_stem,
    )

    # Body skeleton — saved for future AnimateDiff/DWPose phase
    try:
        extract_body_skeleton(
            driver_video=driver_video,
            output_json=f"{output_video}.body.json",
        )
    except Exception as exc:
        # Non-fatal: body skeleton is supplementary data only
        logger.warning(f"Body skeleton extraction failed (non-fatal): {exc}")
