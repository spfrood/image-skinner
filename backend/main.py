"""FastAPI backend — exposes REST endpoints consumed by the Gradio frontend."""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path
from typing import Optional

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from loguru import logger

from backend.character_gallery import CharacterGallery
from backend.config import settings
from backend.models.schemas import (
    CharacterProfile,
    RenderJob,
    RenderRequest,
    RenderResponse,
)
from backend.pipeline import RenderPipeline

app = FastAPI(title="SoloStudio-AI", version="0.1.0")

gallery = CharacterGallery()
_jobs: dict[str, RenderJob] = {}   # in-memory; replace with SQLite for persistence


# ── Character Gallery ──────────────────────────────────────────────────────────

@app.get("/characters", response_model=list[CharacterProfile])
def list_characters():
    return gallery.list_all()


@app.get("/characters/{character_id}", response_model=CharacterProfile)
def get_character(character_id: str):
    try:
        return gallery.get(character_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Character not found")


@app.post("/characters", response_model=CharacterProfile)
async def create_character(
    name: str = Form(...),
    notes: Optional[str] = Form(None),
    sketch: UploadFile = File(...),
    voice_model: UploadFile = File(...),
    voice_index: UploadFile = File(...),
    voice_sample: UploadFile = File(...),
):
    tmp = Path(settings.assets_dir) / "tmp_upload"
    tmp.mkdir(parents=True, exist_ok=True)
    saved = {}
    try:
        for field_name, upload in [
            ("sketch", sketch),
            ("voice_model", voice_model),
            ("voice_index", voice_index),
            ("voice_sample", voice_sample),
        ]:
            dest = tmp / upload.filename
            with dest.open("wb") as f:
                shutil.copyfileobj(upload.file, f)
            saved[field_name] = dest

        profile = gallery.create(
            name=name,
            sketch_src=saved["sketch"],
            voice_model_src=saved["voice_model"],
            voice_index_src=saved["voice_index"],
            voice_sample_src=saved["voice_sample"],
            notes=notes,
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    return profile


@app.delete("/characters/{character_id}", status_code=204)
def delete_character(character_id: str):
    gallery.delete(character_id)


@app.get("/characters/{character_id}/sketch")
def get_sketch(character_id: str):
    try:
        profile = gallery.get(character_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Character not found")
    path = gallery.asset_path(character_id, profile.sketch_filename)
    return FileResponse(str(path), media_type="image/png")


# ── Recording upload ───────────────────────────────────────────────────────────

@app.post("/upload-recording")
async def upload_recording(file: UploadFile = File(...)):
    dest = Path(settings.recordings_dir) / file.filename
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    return {"filename": file.filename}


# ── Render jobs ────────────────────────────────────────────────────────────────

@app.post("/render", response_model=RenderResponse)
async def start_render(req: RenderRequest, background_tasks: BackgroundTasks):
    try:
        gallery.get(req.character_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Character not found")

    recording_path = Path(settings.recordings_dir) / req.recording_filename
    if not recording_path.exists():
        raise HTTPException(status_code=404, detail="Recording file not found")

    job = RenderJob(
        character_id=req.character_id,
        recording_filename=req.recording_filename,
    )
    _jobs[job.id] = job

    def _run_job(job_id: str) -> None:
        pipeline = RenderPipeline(gallery=gallery)
        _jobs[job_id] = pipeline.run(_jobs[job_id])

    background_tasks.add_task(_run_job, job.id)
    return RenderResponse(job_id=job.id, status=job.status)


@app.get("/render/{job_id}", response_model=RenderResponse)
def get_render_status(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return RenderResponse(
        job_id=job.id,
        status=job.status,
        output_filename=job.output_filename,
        error=job.error,
    )


@app.get("/outputs/{filename}")
def download_output(filename: str):
    path = Path(settings.outputs_dir) / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Output not found")
    return FileResponse(str(path), media_type="video/mp4", filename=filename)
