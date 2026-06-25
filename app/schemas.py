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


class QualityLabel(str, Enum):
    green = "green"
    yellow = "yellow"
    red = "red"
    unknown = "unknown"


class Signals(BaseModel):
    sentiment: float
    neg_share: float
    trend: float | None = None


class Quality(BaseModel):
    label: QualityLabel
    score: float | None = None
    signals: Signals | None = None


class AnalyzeResponse(BaseModel):
    filename: str
    duration: float | None = None
    language: str | None = None
    channels: int
    quality: Quality
    segments: list[Segment]
    stats: dict[str, SpeakerStats]
