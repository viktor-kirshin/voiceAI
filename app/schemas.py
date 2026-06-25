from __future__ import annotations

from enum import Enum

from pydantic import BaseModel


class Speaker(str, Enum):
    operator = "operator"
    client = "client"
    unknown = "unknown"


class Sentiment(str, Enum):
    positive = "положительный"
    neutral = "нейтральный"
    negative = "отрицательный"


class Segment(BaseModel):
    index: int
    speaker: Speaker
    start: float
    end: float
    text: str
    sentiment: Sentiment | None = None


class SpeakerStats(BaseModel):
    segments: int = 0
    positive: int = 0
    neutral: int = 0
    negative: int = 0


class AnalyzeResponse(BaseModel):
    filename: str
    duration: float | None = None
    language: str | None = None
    channels: int
    segments: list[Segment]
    stats: dict[str, SpeakerStats]
