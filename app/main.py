from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from openai import APIConnectionError, APIStatusError, AsyncOpenAI

from .audio import AudioError
from .config import settings
from .pipeline import analyze_audio
from .schemas import AnalyzeResponse

logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.whisper = AsyncOpenAI(
        base_url=settings.whisper_base_url,
        api_key=settings.whisper_api_key,
        timeout=settings.whisper_timeout_s,
    )
    app.state.gemma = AsyncOpenAI(
        base_url=settings.gemma_base_url,
        api_key=settings.gemma_api_key,
        timeout=settings.gemma_timeout_s,
    )
    yield
    await app.state.whisper.close()
    await app.state.gemma.close()


app = FastAPI(title="Voice Call-Center Analyzer", version="0.1.0", lifespan=lifespan)

INDEX_HTML = Path(__file__).resolve().parent.parent / "index.html"


@app.get("/")
async def index():
    return FileResponse(INDEX_HTML)


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: Request, file: UploadFile = File(...)) -> AnalyzeResponse:
    raw = await file.read()
    if not raw:
        raise HTTPException(400, "Пустой файл.")
    if len(raw) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(413, f"Файл больше {settings.max_upload_mb} МБ.")

    try:
        return await analyze_audio(
            raw,
            file.filename or "upload",
            request.app.state.whisper,
            request.app.state.gemma,
        )
    except AudioError as exc:
        raise HTTPException(415, str(exc)) from exc
    except APIStatusError as exc:
        raise HTTPException(502, f"Ошибка модели ({exc.status_code}).") from exc
    except APIConnectionError as exc:
        raise HTTPException(504, f"Модель недоступна: {exc}") from exc


def run() -> None:

    uvicorn.run(app, host=settings.host, port=settings.port)
