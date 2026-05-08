"""
Motion / Pose Transfer — maps the driver's body + face movements onto the
character sketch using a two-stage approach:

Stage 1 – Pose extraction  : MediaPipe Holistic extracts a landmark skeleton
                              (face, hands, body) from every frame of the
                              driver video.
Stage 2 – Frame synthesis  : Each skeleton is used to warp / drive the
                              reference sketch via a ControlNet-style pipeline
                              (DWPose + AnimateDiff or LivePortrait for face).

For production, swap the stub synthesiser below with your chosen model:
  • Viggle AI  – commercial API (send base image + skeleton frames)
  • LivePortrait – open-source, face-only, runs locally on the 4090
  • AnimateDiff + ControlNet (DWPose) – full body, diffusion-based
"""

from __future__ import annotations

from pathlib import Path
from typing import Generator

import cv2
import imageio
import mediapipe as mp
import numpy as np
from loguru import logger
from PIL import Image


mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils


# ── Pose extraction ────────────────────────────────────────────────────────────

def extract_pose_frames(driver_video: str) -> Generator[np.ndarray, None, None]:
    """Yields a pose-skeleton overlay frame for each video frame."""
    cap = cv2.VideoCapture(driver_video)
    with mp_holistic.Holistic(
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    ) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb)

            canvas = np.zeros_like(frame)
            _draw_landmarks(canvas, results)
            yield canvas
    cap.release()


def _draw_landmarks(canvas: np.ndarray, results) -> None:
    spec = mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2)
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(canvas, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, spec, spec)
    if results.face_landmarks:
        mp_drawing.draw_landmarks(canvas, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, spec, spec)
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(canvas, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, spec, spec)
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(canvas, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, spec, spec)


# ── Frame synthesis ────────────────────────────────────────────────────────────

def synthesise_frame(
    pose_frame: np.ndarray,
    reference_image: np.ndarray,
) -> np.ndarray:
    """
    Produce an output frame that looks like the character sketch performing
    the pose captured in pose_frame.

    TODO: Replace this stub with your chosen synthesis model:
      - LivePortrait: from liveportrait import LivePortrait; lp.drive(ref, pose)
      - AnimateDiff + ControlNet DWPose: pipe(image=ref, control=pose)
      - Viggle AI REST call
    """
    # Stub: alpha-blend pose skeleton onto reference for visual proof-of-concept
    ref_resized = cv2.resize(reference_image, (pose_frame.shape[1], pose_frame.shape[0]))
    blended = cv2.addWeighted(ref_resized, 0.85, pose_frame, 0.15, 0)
    return blended


# ── Main orchestrator ──────────────────────────────────────────────────────────

def run_motion_transfer(
    driver_video: str,
    reference_image: str,
    output_video: str,
) -> None:
    logger.info(f"Motion transfer: {Path(driver_video).name} → {Path(output_video).name}")

    cap = cv2.VideoCapture(driver_video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    reference = np.array(Image.open(reference_image).convert("RGB"))

    writer = imageio.get_writer(output_video, fps=fps, codec="libx264", quality=8)
    frame_count = 0

    for pose_frame in extract_pose_frames(driver_video):
        out_frame = synthesise_frame(pose_frame, reference)
        writer.append_data(cv2.cvtColor(out_frame, cv2.COLOR_BGR2RGB))
        frame_count += 1

    writer.close()
    logger.success(f"Motion transfer complete: {frame_count} frames → {output_video}")
