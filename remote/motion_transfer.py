"""
Motion / Pose Transfer — full pipeline.

Stage 1 – Body skeleton   : MediaPipe Holistic extracts per-frame body + hand
                             landmarks and writes them to a companion JSON.
Stage 2 – DWPose render   : Landmarks are converted to DWPose stick-figure images
                             that serve as ControlNet conditioning.
Stage 3 – Body animation  : AnimateDiff + ControlNet OpenPose + IP-Adapter
                             generates full-body character animation from the
                             DWPose frames and the character sketch.
Stage 4 – Face animation  : LivePortrait overlays high-fidelity facial
                             expressions from the driver onto the body output.
Stage 5 – Compositing     : The LivePortrait face region is blended onto each
                             AnimateDiff body frame using a soft mask derived
                             from the shoulder-width / nose-position heuristic.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from loguru import logger
from PIL import Image

from dwpose_renderer import (
    skeleton_json_to_dwpose_images,
    get_face_boxes_from_skeleton,
)
from body_animator import run_body_animation


LIVEPORTRAIT_ROOT      = Path("/workspace/LivePortrait")
LIVEPORTRAIT_READY_FLAG = Path("/workspace/.lp_ready")


# ── LivePortrait readiness ─────────────────────────────────────────────────────

def wait_for_liveportrait(timeout_s: int = 1200, poll_s: int = 10) -> None:
    """Block until pod_setup.sh finishes installing LivePortrait + models."""
    if LIVEPORTRAIT_READY_FLAG.exists():
        return
    logger.info(
        f"Waiting for LivePortrait setup (up to {timeout_s // 60} min)…"
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


# ── Stage 1: Body skeleton extraction ─────────────────────────────────────────

def extract_body_skeleton(driver_video: str, output_json: str) -> int:
    """
    Run MediaPipe Holistic on every frame of driver_video and write a compact
    per-frame JSON with pose, left_hand, and right_hand landmarks.
    Returns the frame count.
    """
    import json
    import mediapipe as mp

    mp_holistic = mp.solutions.holistic
    frames_data: list[dict] = []

    cap = cv2.VideoCapture(driver_video)
    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1,
    ) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = holistic.process(rgb)
            entry: dict = {}
            if res.pose_landmarks:
                entry["pose"] = [
                    {"x": lm.x, "y": lm.y, "z": lm.z, "v": lm.visibility}
                    for lm in res.pose_landmarks.landmark
                ]
            if res.left_hand_landmarks:
                entry["left_hand"] = [
                    {"x": lm.x, "y": lm.y, "z": lm.z}
                    for lm in res.left_hand_landmarks.landmark
                ]
            if res.right_hand_landmarks:
                entry["right_hand"] = [
                    {"x": lm.x, "y": lm.y, "z": lm.z}
                    for lm in res.right_hand_landmarks.landmark
                ]
            frames_data.append(entry)
    cap.release()

    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(output_json).write_text(json.dumps(frames_data), encoding="utf-8")
    logger.info(f"Body skeleton: {len(frames_data)} frames → {output_json}")
    return len(frames_data)


# ── Stage 3: LivePortrait face animation ──────────────────────────────────────

def run_liveportrait(
    driver_video: str,
    reference_image: str,
    output_dir: str,
    output_stem: str,
) -> Path:
    """
    Animate the reference sketch's face using the driver's facial motion.
    Returns the path to the produced video file.
    """
    wait_for_liveportrait()

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    result = subprocess.run(
        [
            sys.executable, str(LIVEPORTRAIT_ROOT / "inference.py"),
            "-s", reference_image,
            "-d", driver_video,
            "--output-dir", str(out_path),
            "--flag-relative-motion",
            "--flag-pasteback",
            "--flag-stitching",
        ],
        cwd=str(LIVEPORTRAIT_ROOT),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        logger.error(result.stderr[-2000:])
        raise RuntimeError(f"LivePortrait failed (exit {result.returncode})")

    candidates = sorted(out_path.glob("*.mp4"))
    if not candidates:
        raise FileNotFoundError(
            f"LivePortrait produced no .mp4 in {out_path}. "
            f"stderr: {result.stderr[-1000:]}"
        )
    final = out_path / f"{output_stem}_lp.mp4"
    candidates[-1].rename(final)
    logger.success(f"LivePortrait output: {final}")
    return final


# ── Stage 4: Face → Body compositing ─────────────────────────────────────────

def _build_face_mask(
    frame_h: int,
    frame_w: int,
    cx: int,
    cy: int,
    radius: int,
) -> np.ndarray:
    """Soft circular mask (float32, [0,1]) centred on the face region."""
    mask = np.zeros((frame_h, frame_w), dtype=np.float32)
    cv2.circle(mask, (cx, cy), radius, 1.0, -1)
    blur_k = max(radius // 3 * 2 + 1, 11)  # must be odd
    if blur_k % 2 == 0:
        blur_k += 1
    mask = cv2.GaussianBlur(mask, (blur_k, blur_k), radius // 4)
    return mask[:, :, np.newaxis]  # (H, W, 1) for broadcasting over channels


def composite_face_onto_body(
    lp_video_path: str,
    ad_video_path: str,
    output_path: str,
    face_boxes: list[Optional[tuple[int, int, int]]],
    fps: float,
) -> None:
    """
    For every frame:
      - source_face  : LivePortrait frame (high-quality face animation)
      - source_body  : AnimateDiff frame  (full-body pose)
      - result       : body with face region replaced by LP face
    The face_boxes list provides (cx, cy, radius) per frame from skeleton data.
    """
    import imageio

    lp_cap  = cv2.VideoCapture(lp_video_path)
    ad_cap  = cv2.VideoCapture(ad_video_path)
    ad_w    = int(ad_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    ad_h    = int(ad_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(output_path, fps=fps, codec="libx264", quality=8)

    frame_idx = 0
    while True:
        ret_lp, lp_frame = lp_cap.read()
        ret_ad, ad_frame = ad_cap.read()
        if not ret_lp or not ret_ad:
            break

        # Resize LP frame to match AnimateDiff output dimensions
        lp_resized = cv2.resize(
            cv2.cvtColor(lp_frame, cv2.COLOR_BGR2RGB),
            (ad_w, ad_h),
        )
        ad_rgb = cv2.cvtColor(ad_frame, cv2.COLOR_BGR2RGB)

        box = face_boxes[frame_idx] if frame_idx < len(face_boxes) else None
        if box:
            cx, cy, radius = box
            # Scale face box from original driver resolution to ad_h × ad_w
            orig_h = int(lp_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            orig_w = int(lp_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            sx = ad_w / orig_w
            sy = ad_h / orig_h
            cx_s  = int(cx * sx)
            cy_s  = int(cy * sy)
            rad_s = int(radius * min(sx, sy))
            mask  = _build_face_mask(ad_h, ad_w, cx_s, cy_s, rad_s)
            composited = (
                lp_resized.astype(np.float32) * mask
                + ad_rgb.astype(np.float32) * (1.0 - mask)
            ).astype(np.uint8)
        else:
            composited = ad_rgb

        writer.append_data(composited)
        frame_idx += 1

    lp_cap.release()
    ad_cap.release()
    writer.close()
    logger.success(f"Composited {frame_idx} frames → {output_path}")


# ── Public entry point ─────────────────────────────────────────────────────────

def run_motion_transfer(
    driver_video: str,
    reference_image: str,
    output_video: str,
    character_prompt: str = "a stylized animated character, full body, anime art style",
) -> None:
    """
    Orchestrates all five stages and writes the final silent video to output_video.

    Intermediate files (body.json, LP video, AD body video) are kept in the
    same directory as output_video and can be inspected for debugging.
    """
    out_dir  = Path(output_video).parent
    stem     = Path(output_video).stem

    skeleton_json = str(out_dir / f"{stem}.body.json")
    lp_video      = str(out_dir / f"{stem}_lp.mp4")
    ad_video      = str(out_dir / f"{stem}_ad.mp4")

    # Probe video metadata
    cap = cv2.VideoCapture(driver_video)
    fps   = cap.get(cv2.CAP_PROP_FPS) or 24.0
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    # ── Stage 1: extract skeleton ──────────────────────────────────────────────
    logger.info("Stage 1/4 — body skeleton extraction")
    extract_body_skeleton(driver_video, skeleton_json)

    # ── Stage 2: render DWPose conditioning images ─────────────────────────────
    logger.info("Stage 2/4 — rendering DWPose conditioning frames")
    conditioning_frames = skeleton_json_to_dwpose_images(skeleton_json, vid_w, vid_h)
    face_boxes = get_face_boxes_from_skeleton(skeleton_json, vid_w, vid_h)
    logger.info(f"  {len(conditioning_frames)} conditioning frames prepared")

    # ── Stage 3a: AnimateDiff body animation ───────────────────────────────────
    logger.info("Stage 3/4 — AnimateDiff body animation")
    reference_pil = Image.open(reference_image).convert("RGB")
    run_body_animation(
        conditioning_frames=conditioning_frames,
        reference_image=reference_pil,
        output_video_path=ad_video,
        character_prompt=character_prompt,
        fps=fps,
    )

    # ── Stage 3b: LivePortrait face animation ──────────────────────────────────
    logger.info("Stage 3b/4 — LivePortrait face animation")
    run_liveportrait(
        driver_video=driver_video,
        reference_image=reference_image,
        output_dir=str(out_dir),
        output_stem=stem,
    )

    # ── Stage 4: composite face onto body ─────────────────────────────────────
    logger.info("Stage 4/4 — compositing face onto body")
    composite_face_onto_body(
        lp_video_path=lp_video,
        ad_video_path=ad_video,
        output_path=output_video,
        face_boxes=face_boxes,
        fps=fps,
    )
