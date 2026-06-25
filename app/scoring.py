from __future__ import annotations

import math

from .config import settings
from .schemas import Quality, QualityLabel, Segment, Sentiment, Signals, Speaker

VALUE = {Sentiment.positive: 1.0, Sentiment.neutral: 0.0, Sentiment.negative: -1.0}
SEVERITY = {QualityLabel.red: 0, QualityLabel.yellow: 1, QualityLabel.green: 2}


def _speaker_weight(speaker: Speaker) -> float:
    if speaker is Speaker.operator:
        return settings.quality_w_operator
    if speaker is Speaker.client:
        return settings.quality_w_client
    return 1.0


def _neg_weight(speaker: Speaker) -> float:
    if speaker is Speaker.operator:
        return settings.quality_neg_w_operator
    if speaker is Speaker.client:
        return settings.quality_neg_w_client
    return 1.0


def compute_quality(segments: list[Segment]) -> Quality:
    scored = [s for s in segments if s.sentiment is not None]
    if not scored:
        return Quality(label=QualityLabel.unknown)

    beta = settings.quality_beta
    duration = max((s.end for s in scored), default=0.0) or 1.0
    recency = [math.exp(settings.quality_recency_k * s.end / duration) for s in scored]
    weights = [r * _speaker_weight(s.speaker) for r, s in zip(recency, scored)]
    total_w = sum(weights) or 1.0

    # Сигнал 1: взвешенная тональность с асимметрией негатива.
    asym = [-beta if s.sentiment is Sentiment.negative else VALUE[s.sentiment] for s in scored]
    q = sum(w * v for w, v in zip(weights, asym)) / total_w
    s_sent = (q + beta) / (1 + beta)

    # Сигнал 2: доля негатива; негатив оператора тяжелее негатива клиента.
    ref = sum(recency) or 1.0
    neg_pen = sum(
        r * _neg_weight(s.speaker)
        for r, s in zip(recency, scored)
        if s.sentiment is Sentiment.negative
    )
    s_neg = max(0.0, 1.0 - neg_pen / ref)

    # Сигнал 3: тренд по клиенту (симметричные значения).
    client_vals = [VALUE[s.sentiment] for s in scored if s.speaker is Speaker.client]
    a, b, c = settings.quality_a, settings.quality_b, settings.quality_c
    s_trend: float | None = None
    if len(client_vals) >= settings.quality_trend_min_segments:
        third = len(client_vals) // 3
        delta = sum(client_vals[-third:]) / third - sum(client_vals[:third]) / third
        s_trend = min(1.0, max(0.0, (delta + 2) / 4))
    else:
        a, b, c = a / (a + b), b / (a + b), 0.0

    score = a * s_sent + b * s_neg + c * (s_trend or 0.0)
    label = _apply_guardrails(_threshold_label(score), scored)

    return Quality(
        label=label,
        score=round(score, 3),
        signals=Signals(
            sentiment=round(s_sent, 3),
            neg_share=round(s_neg, 3),
            trend=round(s_trend, 3) if s_trend is not None else None,
        ),
    )


def _threshold_label(score: float) -> QualityLabel:
    if score >= settings.quality_threshold_green:
        return QualityLabel.green
    if score >= settings.quality_threshold_yellow:
        return QualityLabel.yellow
    return QualityLabel.red


def _apply_guardrails(label: QualityLabel, scored: list[Segment]) -> QualityLabel:
    ops = [s for s in scored if s.speaker is Speaker.operator]
    clients = [s for s in scored if s.speaker is Speaker.client]
    op_neg = sum(1 for s in ops if s.sentiment is Sentiment.negative)
    cl_neg = sum(1 for s in clients if s.sentiment is Sentiment.negative)

    # Потолок жёлтого: грубость оператора или негативная финальная реплика клиента.
    if op_neg > 0 or (clients and max(clients, key=lambda s: s.end).sentiment is Sentiment.negative):
        if SEVERITY[label] > SEVERITY[QualityLabel.yellow]:
            label = QualityLabel.yellow

    # Жёсткий красный.
    if ops and op_neg / len(ops) > settings.quality_operator_neg_ratio_red:
        label = QualityLabel.red
    if clients and cl_neg / len(clients) > settings.quality_client_neg_ratio_red:
        label = QualityLabel.red

    return label
