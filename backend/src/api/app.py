"""FastAPI-приложение.

    cd backend && uvicorn src.api.app:app --reload
    # документация: http://localhost:8000/docs

Вся логика — в Python-модулях проекта; фронтенд (Next.js) только показывает
их результаты. Графики из reports/figures раздаются как статика по /figures/.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .. import config
from .routes import chat, faq, insights
from .state import AppState


def create_app(faq_df: pd.DataFrame | None = None, db_path: str | Path | None = None,
               bot_config_path: str | Path | None = None, log_chats: bool = True,
               warmup: list[str] | None = None) -> FastAPI:
    """Фабрика приложения; параметры нужны тестам (игрушечная база вместо настоящей)."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        state = AppState(faq_df, db_path=db_path, bot_config_path=bot_config_path, log_chats=log_chats)
        for method in [state.default_method, *(warmup if warmup is not None else config.API_WARMUP_METHODS)]:
            state.bot(method)  # первый запрос пользователя не ждёт построения индекса
        app.state.aitu = state
        yield

    app = FastAPI(title="AITU FAQ Retrieval API", version="1.0.0", lifespan=lifespan,
                  description="Поиск ответа в FAQ приёмной комиссии Astana IT University.")
    app.add_middleware(CORSMiddleware, allow_origins=config.API_CORS_ORIGINS, allow_credentials=False,
                       allow_methods=["GET", "POST"], allow_headers=["*"])
    app.include_router(chat.router)
    app.include_router(faq.router)
    app.include_router(insights.router)

    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    app.mount("/figures", StaticFiles(directory=config.FIGURES_DIR), name="figures")

    @app.get("/", include_in_schema=False)
    def root() -> dict:
        return {"service": "AITU FAQ Retrieval API", "docs": "/docs", "health": "/api/health"}

    return app


app = create_app()
