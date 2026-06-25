from __future__ import annotations

import io
from dataclasses import dataclass

import soundfile as sf

from .schemas import Speaker


@dataclass
class Channel:
    speaker: Speaker
    wav_bytes: bytes


class AudioError(ValueError):
    pass


def _to_wav(samples, samplerate: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, samples, samplerate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def split_channels(
    raw: bytes, operator_channel: int, client_channel: int
) -> tuple[list[Channel], float, int]:
    try:
        data, samplerate = sf.read(io.BytesIO(raw), dtype="int16", always_2d=True)
    except Exception as exc:
        raise AudioError(f"Не удалось декодировать аудио: {exc}") from exc

    n_frames, n_channels = data.shape
    duration = n_frames / samplerate if samplerate else 0.0

    if n_channels == 1:
        return [Channel(Speaker.unknown, _to_wav(data[:, 0], samplerate))], duration, 1

    labels = {operator_channel: Speaker.operator, client_channel: Speaker.client}
    channels = [
        Channel(labels.get(i, Speaker.unknown), _to_wav(data[:, i], samplerate))
        for i in range(n_channels)
    ]
    return channels, duration, n_channels
