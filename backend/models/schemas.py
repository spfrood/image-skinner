from __future__ import annotations
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field
import uuid


class CharacterProfile(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    sketch_filename: str           # relative to characters_dir
    voice_model_filename: str      # .pth RVC model
    voice_index_filename: str      # .index faiss file
    voice_sample_filename: str     # original WAV sample for reference
    created_at: datetime = Field(default_factory=datetime.utcnow)
    thumbnail_filename: Optional[str] = None
    notes: Optional[str] = None


class RenderJob(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    character_id: str
    recording_filename: str
    output_filename: Optional[str] = None
    status: str = "queued"         # queued | running | done | failed
    pod_id: Optional[str] = None
    error: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None


class RenderRequest(BaseModel):
    character_id: str
    recording_filename: str        # file already uploaded via /upload-recording


class RenderResponse(BaseModel):
    job_id: str
    status: str
    output_filename: Optional[str] = None
    error: Optional[str] = None
