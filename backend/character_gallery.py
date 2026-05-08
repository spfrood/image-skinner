"""
CRUD operations for CharacterProfiles stored as JSON sidecar files
alongside their binary assets in assets/characters/<id>/.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Optional

from loguru import logger

from backend.config import settings
from backend.models.schemas import CharacterProfile


class CharacterGallery:
    def __init__(self) -> None:
        self._root = Path(settings.characters_dir)
        self._root.mkdir(parents=True, exist_ok=True)

    def _profile_dir(self, character_id: str) -> Path:
        return self._root / character_id

    def _meta_path(self, character_id: str) -> Path:
        return self._profile_dir(character_id) / "profile.json"

    # ── Write ──────────────────────────────────────────────────────────────────

    def create(
        self,
        name: str,
        sketch_src: Path,
        voice_model_src: Path,
        voice_index_src: Path,
        voice_sample_src: Path,
        notes: Optional[str] = None,
        animation_prompt: Optional[str] = None,
    ) -> CharacterProfile:
        profile = CharacterProfile(
            name=name,
            sketch_filename=sketch_src.name,
            voice_model_filename=voice_model_src.name,
            voice_index_filename=voice_index_src.name,
            voice_sample_filename=voice_sample_src.name,
            notes=notes,
            **({"animation_prompt": animation_prompt} if animation_prompt else {}),
        )
        dest = self._profile_dir(profile.id)
        dest.mkdir(parents=True, exist_ok=True)

        for src in [sketch_src, voice_model_src, voice_index_src, voice_sample_src]:
            shutil.copy2(src, dest / src.name)

        self._meta_path(profile.id).write_text(
            profile.model_dump_json(indent=2), encoding="utf-8"
        )
        logger.success(f"Character '{name}' saved (id={profile.id})")
        return profile

    def update_notes(self, character_id: str, notes: str) -> CharacterProfile:
        profile = self.get(character_id)
        profile.notes = notes
        self._meta_path(character_id).write_text(
            profile.model_dump_json(indent=2), encoding="utf-8"
        )
        return profile

    def delete(self, character_id: str) -> None:
        d = self._profile_dir(character_id)
        if d.exists():
            shutil.rmtree(d)
            logger.info(f"Character {character_id} deleted.")

    # ── Read ───────────────────────────────────────────────────────────────────

    def get(self, character_id: str) -> CharacterProfile:
        meta = self._meta_path(character_id)
        if not meta.exists():
            raise FileNotFoundError(f"Character {character_id} not found.")
        return CharacterProfile.model_validate_json(meta.read_text(encoding="utf-8"))

    def list_all(self) -> list[CharacterProfile]:
        profiles: list[CharacterProfile] = []
        for meta in sorted(self._root.glob("*/profile.json")):
            try:
                profiles.append(
                    CharacterProfile.model_validate_json(
                        meta.read_text(encoding="utf-8")
                    )
                )
            except Exception as exc:
                logger.warning(f"Skipping corrupt profile {meta}: {exc}")
        return profiles

    def asset_path(self, character_id: str, filename: str) -> Path:
        return self._profile_dir(character_id) / filename
