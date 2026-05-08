"""
Full-body character animation via AnimateDiff + ControlNet OpenPose + IP-Adapter.

Pipeline:
  conditioning_frames  (DWPose stick figures)  ──► ControlNet
  reference_image      (character sketch)       ──► IP-Adapter   } AnimateDiff
  character_prompt                              ──► CLIP text     }

The IP-Adapter anchors the generated frames to the visual style of the sketch
so the output looks like the character rather than a generic person.

AnimateDiff has a hard limit of 16–32 frames per call (depending on the motion
adapter).  Videos longer than CHUNK_SIZE frames are processed in overlapping
windows; the overlap region is linearly cross-faded to avoid seams.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from loguru import logger
from PIL import Image


# ── Tunables ───────────────────────────────────────────────────────────────────

CHUNK_SIZE = 16     # frames per AnimateDiff inference call
OVERLAP    = 4      # overlap between consecutive chunks for cross-fade blending
INFER_SIZE = 512    # spatial resolution fed to the diffusion model

SD_BASE_MODEL = os.getenv(
    "SD_BASE_MODEL",
    "SG161222/Realistic_Vision_V5.1_noVAE",  # anime-friendly SD1.5 checkpoint
)
MOTION_ADAPTER_ID  = "guoyww/animatediff-motion-adapter-v1-5-2"
CONTROLNET_ID      = "lllyasviel/sd-controlnet-openpose"
IP_ADAPTER_REPO    = "h94/IP-Adapter"
IP_ADAPTER_WEIGHTS = "ip-adapter_sd15.bin"

DEFAULT_NEGATIVE_PROMPT = (
    "blurry, bad anatomy, bad hands, extra limbs, missing limbs, "
    "ugly, watermark, low quality, deformed face, duplicate"
)


# ── Lazy model cache ───────────────────────────────────────────────────────────
# Models are loaded once per process and reused across jobs.

_pipe = None


def _load_pipeline():
    global _pipe
    if _pipe is not None:
        return _pipe

    from diffusers import (
        AnimateDiffControlNetPipeline,
        AutoencoderKL,
        ControlNetModel,
        MotionAdapter,
    )

    logger.info("Loading AnimateDiff motion adapter…")
    adapter = MotionAdapter.from_pretrained(
        MOTION_ADAPTER_ID, torch_dtype=torch.float16
    )

    logger.info("Loading ControlNet OpenPose…")
    controlnet = ControlNetModel.from_pretrained(
        CONTROLNET_ID, torch_dtype=torch.float16
    )

    logger.info(f"Loading base SD model: {SD_BASE_MODEL}…")
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/sd-vae-ft-mse", torch_dtype=torch.float16
    )
    pipe = AnimateDiffControlNetPipeline.from_pretrained(
        SD_BASE_MODEL,
        motion_adapter=adapter,
        controlnet=controlnet,
        vae=vae,
        torch_dtype=torch.float16,
    )

    # Memory optimisations for 24 GB VRAM (RTX 4090)
    pipe.enable_xformers_memory_efficient_attention()
    pipe.to("cuda")

    logger.info("Loading IP-Adapter for appearance conditioning…")
    pipe.load_ip_adapter(
        IP_ADAPTER_REPO,
        subfolder="models",
        weight_name=IP_ADAPTER_WEIGHTS,
    )
    pipe.set_ip_adapter_scale(0.7)

    _pipe = pipe
    logger.success("AnimateDiff pipeline ready.")
    return _pipe


# ── Chunked inference ──────────────────────────────────────────────────────────

def _run_chunk(
    pipe,
    conditioning_frames: list[Image.Image],
    reference_image: Image.Image,
    prompt: str,
    negative_prompt: str,
    num_inference_steps: int,
    guidance_scale: float,
    seed: int,
) -> list[np.ndarray]:
    """Run one AnimateDiff window and return frames as uint8 RGB numpy arrays."""
    generator = torch.Generator("cuda").manual_seed(seed)
    # AnimateDiffControlNetPipeline uses `image` for the per-frame ControlNet
    # conditioning input (not `conditioning_frames`).
    # output.frames is List[List[PIL.Image]] — outer=batch, inner=frames.
    output = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        ip_adapter_image=reference_image,
        image=conditioning_frames,           # ControlNet conditioning frames
        num_frames=len(conditioning_frames),
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        conditioning_scale=0.8,              # ControlNet influence weight
        generator=generator,
        width=INFER_SIZE,
        height=INFER_SIZE,
    )
    frames = output.frames[0]  # first (and only) batch item
    return [np.array(f) for f in frames]


def _crossfade(a: np.ndarray, b: np.ndarray, n: int, i: int) -> np.ndarray:
    """Linear blend: i=0 → pure a, i=n-1 → pure b."""
    t = i / max(n - 1, 1)
    return (a * (1 - t) + b * t).astype(np.uint8)


def _infer_chunked(
    pipe,
    conditioning_frames: list[Image.Image],
    reference_image: Image.Image,
    prompt: str,
    negative_prompt: str,
    num_inference_steps: int,
    guidance_scale: float,
    seed: int,
) -> list[np.ndarray]:
    """Process an arbitrarily-long sequence in overlapping windows."""
    total = len(conditioning_frames)
    stride = CHUNK_SIZE - OVERLAP
    result: list[Optional[np.ndarray]] = [None] * total

    start = 0
    while start < total:
        end = min(start + CHUNK_SIZE, total)
        chunk_frames = conditioning_frames[start:end]

        # Pad short final chunk to at least 2 frames (AnimateDiff minimum)
        while len(chunk_frames) < 2:
            chunk_frames.append(chunk_frames[-1])

        logger.debug(f"AnimateDiff chunk [{start}:{end}] ({len(chunk_frames)} frames)")
        chunk_out = _run_chunk(
            pipe, chunk_frames, reference_image,
            prompt, negative_prompt, num_inference_steps, guidance_scale,
            seed + start,   # deterministic but varied per chunk
        )

        for local_i, global_i in enumerate(range(start, end)):
            if global_i >= total:
                break
            new_frame = chunk_out[local_i]
            if result[global_i] is None or global_i >= start + OVERLAP:
                result[global_i] = new_frame
            else:
                # Overlap region: blend previous chunk's tail with this chunk's head
                result[global_i] = _crossfade(
                    result[global_i], new_frame,
                    OVERLAP, global_i - start,
                )

        if end >= total:
            break
        start += stride

    return [f for f in result if f is not None]


# ── Public API ─────────────────────────────────────────────────────────────────

def run_body_animation(
    conditioning_frames: list[Image.Image],
    reference_image: Image.Image,
    output_video_path: str,
    character_prompt: str,
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
    num_inference_steps: int = 20,
    guidance_scale: float = 7.5,
    seed: int = 42,
    fps: float = 24.0,
) -> Path:
    """
    Animate reference_image according to the DWPose conditioning_frames.

    conditioning_frames – list of DWPose stick-figure PIL Images, one per frame.
    reference_image     – the character sketch (PIL Image); fed to IP-Adapter.
    character_prompt    – text description of the character/style.
    Returns the path to the written output video.
    """
    if not conditioning_frames:
        raise ValueError("conditioning_frames must not be empty.")

    logger.info(
        f"Body animation: {len(conditioning_frames)} frames  "
        f"prompt='{character_prompt[:60]}…'"
    )
    pipe = _load_pipeline()

    # Resize conditioning frames to model input size
    cond_resized = [f.resize((INFER_SIZE, INFER_SIZE), Image.LANCZOS) for f in conditioning_frames]
    ref_resized  = reference_image.resize((INFER_SIZE, INFER_SIZE), Image.LANCZOS)

    frames_np = _infer_chunked(
        pipe, cond_resized, ref_resized,
        character_prompt, negative_prompt,
        num_inference_steps, guidance_scale, seed,
    )

    _write_video(frames_np, output_video_path, fps)
    logger.success(f"Body animation complete: {output_video_path}")
    return Path(output_video_path)


def _write_video(frames: list[np.ndarray], path: str, fps: float) -> None:
    import imageio
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(path, fps=fps, codec="libx264", quality=8)
    for frame in frames:
        writer.append_data(frame)
    writer.close()
