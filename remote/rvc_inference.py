"""
RVC (Retrieval-based Voice Conversion) Speech-to-Speech inference.

Keeps the driver's inflection/prosody while swapping vocal timbre to the
character's trained voice model (.pth) and FAISS index (.index).

Requires the RVC project to be cloned at /workspace/Retrieval-based-Voice-Conversion-WebUI
on the GPU pod. The pod startup script should handle this clone.
"""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

RVC_ROOT = Path("/workspace/Retrieval-based-Voice-Conversion-WebUI")


def _ensure_rvc_on_path() -> None:
    rvc_str = str(RVC_ROOT)
    if rvc_str not in sys.path:
        sys.path.insert(0, rvc_str)


def run_rvc(
    audio_in: str,
    audio_out: str,
    model_path: str,
    index_path: str,
    f0_method: str = "rmvpe",
    f0_up_key: int = 0,
    filter_radius: int = 3,
    rms_mix_rate: float = 0.25,
    protect: float = 0.33,
    index_rate: float = 0.75,
) -> None:
    """
    Wraps RVC's vc_single() for programmatic (non-GUI) use.

    f0_up_key=0   → no pitch transpose; preserves the driver's expression.
    index_rate    → how strongly the voice index influences timbre (0–1).
    protect       → shields voiceless consonants from over-conversion (0–0.5).
    """
    _ensure_rvc_on_path()

    # Late import — RVC modules are only present on the GPU pod
    import torch
    import soundfile as sf
    import numpy as np
    from configs.config import Config
    from infer.modules.vc.modules import VC

    config = Config()
    vc = VC(config)
    vc.get_vc(model_path)

    logger.info(
        f"RVC: model={Path(model_path).name}  f0={f0_method}  "
        f"key={f0_up_key:+d}  index_rate={index_rate}"
    )

    _, wav_opt = vc.vc_single(
        sid=0,
        input_audio_path=audio_in,
        f0_up_key=f0_up_key,
        f0_file=None,
        f0_method=f0_method,
        file_index=index_path,
        index_rate=index_rate,
        filter_radius=filter_radius,
        resample_sr=0,
        rms_mix_rate=rms_mix_rate,
        protect=protect,
    )

    sample_rate, audio = wav_opt
    sf.write(audio_out, audio, sample_rate)
    logger.success(f"RVC output → {audio_out}")
