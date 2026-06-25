from __future__ import annotations

from dataclasses import dataclass

from openai import AsyncOpenAI

from .config import settings


@dataclass
class RawSegment:
    start: float
    end: float
    text: str


@dataclass
class ChannelTranscription:
    language: str | None
    segments: list[RawSegment]


async def transcribe_channel(
    client: AsyncOpenAI, wav_bytes: bytes, filename: str
) -> ChannelTranscription:
    result = await client.audio.transcriptions.create(
        file=(filename, wav_bytes),
        model=settings.whisper_model,
        language=settings.whisper_language,
        response_format="verbose_json",
        temperature=0.01,
    )

    segments = [
        RawSegment(s.start, s.end, s.text.strip())
        for s in (result.segments or [])
        if s.text and s.text.strip()
    ]
    if not segments and result.text.strip():
        segments = [RawSegment(0.0, result.duration or 0.0, result.text.strip())]

    return ChannelTranscription(result.language, segments)
