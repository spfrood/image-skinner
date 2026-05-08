"""
Converts per-frame MediaPipe Holistic landmarks (stored in body.json) into
DWPose-style conditioning images for ControlNet.

Each output image is a black canvas with coloured body limbs, face dots, and
hand keypoints drawn at the positions from the MediaPipe data — matching the
visual format that lllyasviel/sd-controlnet-openpose was trained on.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image


# ── MediaPipe → OpenPose-18 body keypoint mapping ─────────────────────────────
# OpenPose index → MediaPipe pose landmark index (None = synthetic)
#   1 (Neck) is the midpoint of landmarks 11 (L shoulder) and 12 (R shoulder)

_MP_TO_OP18: dict[int, Optional[int]] = {
    0:  0,    # Nose
    1:  None, # Neck — synthetic midpoint
    2:  12,   # RShoulder
    3:  14,   # RElbow
    4:  16,   # RWrist
    5:  11,   # LShoulder
    6:  13,   # LElbow
    7:  15,   # LWrist
    8:  24,   # RHip
    9:  26,   # RKnee
    10: 28,   # RAnkle
    11: 23,   # LHip
    12: 25,   # LKnee
    13: 27,   # LAnkle
    14: 5,    # REye
    15: 2,    # LEye
    16: 8,    # REar
    17: 7,    # LEar
}

# Limb connections between OpenPose-18 keypoints
_OP18_LIMBS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (1, 5), (5, 6), (6, 7),
    (1, 8), (8, 9), (9, 10),
    (1, 11), (11, 12), (12, 13),
    (0, 14), (14, 16),
    (0, 15), (15, 17),
]

# Per-limb BGR colours matching the DWPose / OpenPose reference palette
_LIMB_COLORS_BGR = [
    (85, 0, 255), (0, 0, 255), (0, 85, 255), (0, 170, 255),
    (0, 255, 255), (0, 255, 170), (0, 255, 85),
    (0, 255, 0),   (85, 255, 0), (170, 255, 0),
    (255, 255, 0), (255, 170, 0), (255, 85, 0),
    (255, 0, 0),   (255, 0, 85),
    (255, 0, 170), (255, 0, 255),
]

# Hand connections (MediaPipe 21-point hand skeleton)
_HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]
_HAND_COLOR_BGR = (0, 255, 128)


# ── Core rendering ─────────────────────────────────────────────────────────────

def _to_px(val: float, dim: int) -> int:
    """Normalised [0,1] coordinate → pixel."""
    return int(val * dim)


def render_dwpose_frame(
    frame_data: dict,
    width: int,
    height: int,
    joint_radius: int = 4,
    limb_thickness: int = 3,
) -> np.ndarray:
    """
    Render one DWPose conditioning image from a single frame's landmark data.

    frame_data keys (all optional): "pose", "left_hand", "right_hand"
    Each value is a list of dicts with "x", "y", "z", optionally "v" (visibility).

    Returns a uint8 BGR numpy array of shape (height, width, 3).
    """
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    pose = frame_data.get("pose", [])

    if pose:
        # Build OpenPose-18 keypoints in pixel coords
        kps: list[Optional[tuple[int, int]]] = [None] * 18
        for op_idx, mp_idx in _MP_TO_OP18.items():
            if mp_idx is not None and mp_idx < len(pose):
                lm = pose[mp_idx]
                vis = lm.get("v", 1.0)
                if vis > 0.3:
                    kps[op_idx] = (
                        _to_px(lm["x"], width),
                        _to_px(lm["y"], height),
                    )
            elif op_idx == 1 and len(pose) > 12:
                # Neck: midpoint of left/right shoulders
                ls, rs = pose[11], pose[12]
                lv = min(ls.get("v", 1.0), rs.get("v", 1.0))
                if lv > 0.3:
                    kps[1] = (
                        _to_px((ls["x"] + rs["x"]) / 2, width),
                        _to_px((ls["y"] + rs["y"]) / 2, height),
                    )

        # Draw limbs
        for i, (a, b) in enumerate(_OP18_LIMBS):
            if kps[a] and kps[b]:
                color = _LIMB_COLORS_BGR[i % len(_LIMB_COLORS_BGR)]
                cv2.line(canvas, kps[a], kps[b], color, limb_thickness, cv2.LINE_AA)

        # Draw joints on top
        for pt in kps:
            if pt:
                cv2.circle(canvas, pt, joint_radius, (255, 255, 255), -1, cv2.LINE_AA)

    # Draw hands
    for hand_key, color in [("left_hand", _HAND_COLOR_BGR), ("right_hand", _HAND_COLOR_BGR)]:
        hand = frame_data.get(hand_key, [])
        if not hand:
            continue
        pts = [
            (_to_px(lm["x"], width), _to_px(lm["y"], height))
            for lm in hand
        ]
        for a, b in _HAND_CONNECTIONS:
            if a < len(pts) and b < len(pts):
                cv2.line(canvas, pts[a], pts[b], color, 2, cv2.LINE_AA)
        for pt in pts:
            cv2.circle(canvas, pt, 3, (255, 255, 255), -1, cv2.LINE_AA)

    return canvas


# ── Batch conversion ───────────────────────────────────────────────────────────

def skeleton_json_to_dwpose_images(
    skeleton_json_path: str,
    width: int,
    height: int,
) -> list[Image.Image]:
    """
    Load the body.json produced by extract_body_skeleton() and return a list
    of PIL Images suitable for passing as ControlNet conditioning_frames.
    """
    frames_data: list[dict] = json.loads(
        Path(skeleton_json_path).read_text(encoding="utf-8")
    )
    images: list[Image.Image] = []
    for frame in frames_data:
        bgr = render_dwpose_frame(frame, width, height)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        images.append(Image.fromarray(rgb))
    return images


# ── Face bounding-box helper (used by compositing in motion_transfer.py) ──────

def get_face_boxes_from_skeleton(
    skeleton_json_path: str,
    width: int,
    height: int,
) -> list[Optional[tuple[int, int, int]]]:
    """
    Return per-frame face centre + radius as (cx, cy, r) tuples (or None).
    Derived from shoulder width and nose position — no face detector required.
    """
    frames_data: list[dict] = json.loads(
        Path(skeleton_json_path).read_text(encoding="utf-8")
    )
    boxes: list[Optional[tuple[int, int, int]]] = []
    for frame in frames_data:
        pose = frame.get("pose", [])
        if len(pose) < 13:
            boxes.append(None)
            continue
        nose = pose[0]
        ls, rs = pose[11], pose[12]
        vis = min(nose.get("v", 1.0), ls.get("v", 1.0), rs.get("v", 1.0))
        if vis < 0.3:
            boxes.append(None)
            continue
        shoulder_px = math.dist(
            (_to_px(ls["x"], width), _to_px(ls["y"], height)),
            (_to_px(rs["x"], width), _to_px(rs["y"], height)),
        )
        radius = max(int(shoulder_px * 0.5), 20)
        cx = _to_px(nose["x"], width)
        cy = _to_px(nose["y"], height)
        boxes.append((cx, cy, radius))
    return boxes
