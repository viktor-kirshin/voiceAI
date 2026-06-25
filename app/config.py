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
    port: int = 999

    # Quality scoring
    quality_beta: float = 1.5  # вес негатива в значении сегмента
    quality_recency_k: float = 1.0986  # ln3: концовка ×3 к началу
    quality_w_operator: float = 0.5
    quality_w_client: float = 1.0
    quality_a: float = 0.6  # вес тональности
    quality_b: float = 0.2  # вес доли негатива
    quality_c: float = 0.2  # вес тренда
    quality_neg_w_operator: float = 2.0  # тяжесть негатива оператора в S_neg
    quality_neg_w_client: float = 1.0  # тяжесть негатива клиента в S_neg
    quality_trend_min_segments: int = 4
    quality_threshold_green: float = 0.66
    quality_threshold_yellow: float = 0.45
    quality_operator_neg_ratio_red: float = 0.2
    quality_client_neg_ratio_red: float = 0.5


settings = Settings()
