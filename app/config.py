from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Whisper (speech-to-text)
    whisper_base_url: str = "http://192.168.0.36:8001/v1"
    whisper_model: str = "/models/whisper-v3-large"
    whisper_api_key: str = "EMPTY"
    whisper_language: str = "ru"
    whisper_timeout_s: float = 600.0

    # Gemma (sentiment)
    gemma_base_url: str = "http://192.168.0.36:8000/v1"
    gemma_model: str = "/models/gemma-4-26B-A4B-it-NVFP4"
    gemma_api_key: str = "EMPTY"
    gemma_timeout_s: float = 120.0
    gemma_concurrency: int = 8

    # Stereo channel -> speaker
    operator_channel: int = 1
    client_channel: int = 0

    max_upload_mb: int = 200

    # HTTP server
    host: str = "0.0.0.0"
    port: int = 9000


settings = Settings()
