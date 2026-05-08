from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # RunPod
    runpod_api_key: str = ""
    runpod_gpu_type: str = "NVIDIA GeForce RTX 4090"
    runpod_container_image: str = (
        "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04"
    )
    runpod_disk_size_gb: int = 50
    runpod_volume_size_gb: int = 20

    # Local paths
    assets_dir: str = "assets"
    recordings_dir: str = "assets/recordings"
    outputs_dir: str = "assets/outputs"
    characters_dir: str = "assets/characters"

    # API server
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    def ensure_dirs(self) -> None:
        for d in [
            self.assets_dir,
            self.recordings_dir,
            self.outputs_dir,
            self.characters_dir,
        ]:
            Path(d).mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_dirs()
