from __future__ import annotations

import asyncio
import logging

from openai import AsyncOpenAI

from .audio import split_channels
from .config import settings
from .gemma_client import classify_segment
from .schemas import AnalyzeResponse, Segment, Sentiment, SpeakerStats
from .whisper_client import transcribe_channel

logger = logging.getLogger(__name__)


async def analyze_audio(
    raw: bytes, filename: str, whisper: AsyncOpenAI, gemma: AsyncOpenAI
) -> AnalyzeResponse:
    channels, duration, n_channels = split_channels(
        raw, settings.operator_channel, settings.client_channel
    )

    transcriptions = [
        await transcribe_channel(whisper, ch.wav_bytes, f"{ch.speaker.value}.wav")
        for ch in channels
    ]

    segments: list[Segment] = []
    for ch, tr in zip(channels, transcriptions):
        for s in tr.segments:
            segments.append(
                Segment(
                    index=0, speaker=ch.speaker, start=s.start, end=s.end, text=s.text
                )
            )
    segments.sort(key=lambda s: (s.start, s.end))
    for i, seg in enumerate(segments):
        seg.index = i

    await _classify(gemma, segments)

    language = next((tr.language for tr in transcriptions if tr.language), None)
    return AnalyzeResponse(
        filename=filename,
        duration=round(duration, 3),
        language=language,
        channels=n_channels,
        segments=segments,
        stats=_build_stats(segments),
    )


async def _classify(gemma: AsyncOpenAI, segments: list[Segment]) -> None:
    sem = asyncio.Semaphore(settings.gemma_concurrency)

    async def worker(seg: Segment) -> None:
        async with sem:
            try:
                seg.sentiment = await classify_segment(gemma, seg.text)
            except Exception:
                logger.exception("Не удалось классифицировать сегмент %s", seg.index)

    await asyncio.gather(*(worker(seg) for seg in segments))


def _build_stats(segments: list[Segment]) -> dict[str, SpeakerStats]:
    stats: dict[str, SpeakerStats] = {}
    for seg in segments:
        bucket = stats.setdefault(seg.speaker.value, SpeakerStats())
        bucket.segments += 1
        if seg.sentiment is Sentiment.positive:
            bucket.positive += 1
        elif seg.sentiment is Sentiment.negative:
            bucket.negative += 1
        elif seg.sentiment is Sentiment.neutral:
            bucket.neutral += 1
    return stats
