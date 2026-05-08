"""
Glue code: orchestrates local capture → remote GPU inference → local output.

Flow:
  1. Receive a RenderJob with a driver recording + character profile.
  2. Spin up a RunPod RTX 4090 pod via GPUManager.
  3. Upload the recording + character assets.
  4. Trigger the remote worker (RVC + motion transfer).
  5. Poll until done, then pull the output video back.
  6. Terminate the pod.
"""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from loguru import logger

from backend.character_gallery import CharacterGallery
from backend.config import settings
from backend.gpu_manager import GPUManager, JobPayload
from backend.models.schemas import RenderJob


# Status callback type: receives (job_id, status_message)
StatusCallback = Callable[[str, str], None]


class RenderPipeline:
    def __init__(
        self,
        gallery: Optional[CharacterGallery] = None,
        on_status: Optional[StatusCallback] = None,
    ) -> None:
        self._gallery = gallery or CharacterGallery()
        self._on_status = on_status or (lambda job_id, msg: logger.info(f"[{job_id}] {msg}"))

    def _emit(self, job: RenderJob, msg: str) -> None:
        job.status = msg
        self._on_status(job.id, msg)

    # ── Main entry point ───────────────────────────────────────────────────────

    def run(self, job: RenderJob) -> RenderJob:
        """
        Blocking execution of the full render pipeline.
        Updates job.status, job.output_filename, job.error in-place.
        """
        try:
            self._emit(job, "running")
            profile = self._gallery.get(job.character_id)

            payload = JobPayload(
                recording_path=str(
                    Path(settings.recordings_dir) / job.recording_filename
                ),
                character_id=job.character_id,
                sketch_path=str(
                    self._gallery.asset_path(job.character_id, profile.sketch_filename)
                ),
                voice_model_path=str(
                    self._gallery.asset_path(
                        job.character_id, profile.voice_model_filename
                    )
                ),
                voice_index_path=str(
                    self._gallery.asset_path(
                        job.character_id, profile.voice_index_filename
                    )
                ),
                output_filename=f"{job.id}_output.mp4",
            )
            job.output_filename = payload.output_filename

            with GPUManager() as gpu:
                # 1. Create pod
                self._emit(job, "provisioning_pod")
                handle = gpu.create_pod()
                job.pod_id = handle.pod_id

                # 2. Upload assets
                self._emit(job, "syncing_assets")
                gpu.sync_assets_to_pod(payload, handle)

                # 3. Trigger inference on the pod
                self._emit(job, "running_inference")
                self._trigger_inference(gpu, handle, payload)

                # 4. Wait for output file to appear on the pod
                self._emit(job, "waiting_for_output")
                self._poll_until_output_ready(
                    gpu, handle, payload.output_filename
                )

                # 5. Fetch result
                self._emit(job, "fetching_output")
                gpu.fetch_output_from_pod(handle, payload.output_filename)

            job.status = "done"
            job.completed_at = datetime.utcnow()
            logger.success(f"Job {job.id} completed → {job.output_filename}")

        except Exception as exc:
            logger.exception(f"Job {job.id} failed: {exc}")
            job.status = "failed"
            job.error = str(exc)
            job.completed_at = datetime.utcnow()

        return job

    # ── Inference trigger ──────────────────────────────────────────────────────

    def _trigger_inference(self, gpu: GPUManager, handle, payload: JobPayload) -> None:
        """
        Kick off the remote worker.py script with the right arguments.
        The worker is already running via docker_args; we send it a trigger
        file so it can process without an open SSH channel.
        """
        profile = self._gallery.get(payload.character_id)
        safe_prompt = profile.animation_prompt.replace("'", "\\'")
        trigger_json = (
            f'{{"recording": "recording.mp4", '
            f'"sketch": "sketch.png", '
            f'"voice_model": "voice_model.pth", '
            f'"voice_index": "voice_index.index", '
            f'"character_prompt": "{safe_prompt}", '
            f'"output": "/workspace/output/{payload.output_filename}"}}'
        )
        gpu._exec_on_pod(
            handle,
            f"echo '{trigger_json}' > /workspace/trigger.json",
        )

    def _poll_until_output_ready(
        self,
        gpu: GPUManager,
        handle,
        output_filename: str,
        timeout_s: int = 1800,   # 30 minutes max
        interval_s: int = 15,
    ) -> None:
        elapsed = 0
        output_path = f"/workspace/output/{output_filename}"
        while elapsed < timeout_s:
            result = gpu._exec_on_pod(
                handle,
                f"test -f {output_path} && echo READY || echo WAITING",
            )
            if "READY" in result:
                return
            logger.debug(
                f"Output not ready yet ({elapsed}s elapsed), retrying in {interval_s}s…"
            )
            time.sleep(interval_s)
            elapsed += interval_s

        raise TimeoutError(
            f"Output {output_filename} not produced within {timeout_s}s"
        )
