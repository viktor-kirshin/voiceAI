from __future__ import annotations

from typing import Literal

from openai import AsyncOpenAI
from pydantic import BaseModel

from .config import settings
from .schemas import Sentiment

SYSTEM_PROMPT = (
    "Ты определяешь тональность реплики из разговора колл-центра. "
    "Классифицируй текст в одну из категорий:\n"
    "- положительный: благодарность, удовлетворённость, дружелюбие, похвала;\n"
    "- отрицательный: недовольство, раздражение, агрессия, грубость, жалоба, мат;\n"
    "- нейтральный: обычная информация без выраженных эмоций.\n"
    "Верни результат в поле sentiment."
)


class SentimentSchema(BaseModel):
    sentiment: Literal["положительный", "нейтральный", "отрицательный"]


async def classify_segment(client: AsyncOpenAI, text: str) -> Sentiment:
    result = await client.chat.completions.parse(
        model=settings.gemma_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        temperature=0,
        response_format=SentimentSchema,
    )
    return Sentiment(result.choices[0].message.parsed.sentiment)
