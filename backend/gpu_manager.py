"""
Manages the full lifecycle of a RunPod RTX 4090 inference pod:
  create → wait-ready → sync files → execute job → fetch result → terminate
"""

from __future__ import annotations

import os
import time
import tarfile
import tempfile
from pathlib import Path
from typing import Optional

import httpx
import runpod
from loguru import logger
from pydantic import BaseModel

from backend.config import settings


# ── Data shapes ───────────────────────────────────────────────────────────────

class PodHandle(BaseModel):
    pod_id: str
    ssh_host: Optional[str] = None
    ssh_port: Optional[int] = None
    status: str = "CREATED"


class JobPayload(BaseModel):
    recording_path: str        # local path to driver video
    character_id: str          # references a CharacterProfile
    sketch_path: str           # local path to PNG
    voice_model_path: str      # local path to .pth RVC model
    voice_index_path: str      # local path to .index file
    output_filename: str


# ── GPU lifecycle ──────────────────────────────────────────────────────────────

class GPUManager:
    """Thin orchestration layer over the RunPod Python SDK."""

    _POD_READY_STATES = {"RUNNING"}
    _POLL_INTERVAL_S = 5
    _MAX_WAIT_S = 300          # 5 minutes for pod to become ready

    def __init__(self) -> None:
        runpod.api_key = settings.runpod_api_key
        self._active_pod: Optional[PodHandle] = None

    # ── Pod lifecycle ──────────────────────────────────────────────────────────

    def create_pod(self) -> PodHandle:
        """Spin up an RTX 4090 pod and block until it is SSH-accessible."""
        logger.info("Requesting RunPod RTX 4090 instance…")
        pod = runpod.create_pod(
            name="solostudio-inference",
            image_name=settings.runpod_container_image,
            gpu_type_id=settings.runpod_gpu_type,
            cloud_type="SECURE",
            support_public_ip=True,
            start_ssh=True,
            data_center_id=None,       # let RunPod pick cheapest region
            container_disk_in_gb=settings.runpod_disk_size_gb,
            volume_in_gb=settings.runpod_volume_size_gb,
            volume_mount_path="/workspace",
            env={
                "PYTHONUNBUFFERED": "1",
                "HF_HOME": "/workspace/hf_cache",
            },
            # Run the remote worker on startup
            docker_args=(
                "bash -c 'pip install -q runpod loguru && "
                "python /workspace/worker.py'"
            ),
        )

        handle = PodHandle(pod_id=pod["id"], status=pod.get("desiredStatus", "CREATED"))
        logger.info(f"Pod created: {handle.pod_id}")
        self._active_pod = self._wait_for_ready(handle)
        return self._active_pod

    def _wait_for_ready(self, handle: PodHandle) -> PodHandle:
        elapsed = 0
        while elapsed < self._MAX_WAIT_S:
            pod_info = runpod.get_pod(handle.pod_id)
            status = pod_info.get("desiredStatus", "")
            runtime = pod_info.get("runtime") or {}
            ports = runtime.get("ports", [])

            logger.debug(f"Pod {handle.pod_id} status={status} elapsed={elapsed}s")

            if status in self._POD_READY_STATES and ports:
                ssh_port_entry = next(
                    (p for p in ports if p.get("privatePort") == 22), None
                )
                if ssh_port_entry:
                    handle.ssh_host = ssh_port_entry.get("ip")
                    handle.ssh_port = ssh_port_entry.get("publicPort")
                    handle.status = status
                    logger.success(
                        f"Pod ready: {handle.ssh_host}:{handle.ssh_port}"
                    )
                    return handle

            time.sleep(self._POLL_INTERVAL_S)
            elapsed += self._POLL_INTERVAL_S

        raise TimeoutError(
            f"Pod {handle.pod_id} did not become ready within {self._MAX_WAIT_S}s"
        )

    def terminate_pod(self, pod_id: Optional[str] = None) -> None:
        target = pod_id or (self._active_pod.pod_id if self._active_pod else None)
        if not target:
            logger.warning("terminate_pod called but no active pod found.")
            return
        runpod.terminate_pod(target)
        logger.success(f"Pod {target} terminated.")
        if self._active_pod and self._active_pod.pod_id == target:
            self._active_pod = None

    # ── File transfer ──────────────────────────────────────────────────────────

    def sync_assets_to_pod(self, job: JobPayload, handle: PodHandle) -> None:
        """Bundle character assets + recording and upload via RunPod file API."""
        logger.info("Syncing assets to pod…")
        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
            archive_path = tmp.name

        with tarfile.open(archive_path, "w:gz") as tar:
            for label, local_path in [
                ("recording.mp4",   job.recording_path),
                ("sketch.png",      job.sketch_path),
                ("voice_model.pth", job.voice_model_path),
                ("voice_index.index", job.voice_index_path),
            ]:
                tar.add(local_path, arcname=label)

        self._upload_file_to_pod(handle, archive_path, "/workspace/input.tar.gz")
        os.unlink(archive_path)

        # Extract on the pod via RunPod exec API
        self._exec_on_pod(
            handle,
            "tar -xzf /workspace/input.tar.gz -C /workspace/",
        )
        logger.success("Assets synced.")

    def fetch_output_from_pod(self, handle: PodHandle, output_filename: str) -> Path:
        """Download the rendered video from the pod to the local outputs dir."""
        local_out = Path(settings.outputs_dir) / output_filename
        local_out.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Fetching output → {local_out}")
        self._download_file_from_pod(
            handle,
            f"/workspace/output/{output_filename}",
            str(local_out),
        )
        logger.success(f"Output saved: {local_out}")
        return local_out

    # ── RunPod REST helpers ────────────────────────────────────────────────────
    # RunPod exposes a pod-level HTTP proxy when start_ssh=True; we use its
    # /runsync or file endpoints here as a fallback to SSH-based transfers.

    def _runpod_api_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {settings.runpod_api_key}"}

    def _upload_file_to_pod(
        self, handle: PodHandle, local_path: str, remote_path: str
    ) -> None:
        url = f"https://api.runpod.io/v2/{handle.pod_id}/upload"
        with open(local_path, "rb") as f:
            resp = httpx.post(
                url,
                headers=self._runpod_api_headers(),
                files={"file": f},
                data={"path": remote_path},
                timeout=300,
            )
        resp.raise_for_status()

    def _download_file_from_pod(
        self, handle: PodHandle, remote_path: str, local_path: str
    ) -> None:
        url = f"https://api.runpod.io/v2/{handle.pod_id}/download"
        resp = httpx.get(
            url,
            headers=self._runpod_api_headers(),
            params={"path": remote_path},
            timeout=300,
        )
        resp.raise_for_status()
        with open(local_path, "wb") as f:
            f.write(resp.content)

    def _exec_on_pod(self, handle: PodHandle, command: str) -> str:
        url = f"https://api.runpod.io/v2/{handle.pod_id}/exec"
        resp = httpx.post(
            url,
            headers=self._runpod_api_headers(),
            json={"command": command},
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json().get("output", "")

    # ── Context manager convenience ────────────────────────────────────────────

    def __enter__(self) -> "GPUManager":
        return self

    def __exit__(self, *_) -> None:
        if self._active_pod:
            self.terminate_pod()
