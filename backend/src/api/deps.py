"""Общие зависимости роутов."""

from __future__ import annotations

from fastapi import Request

from .. import config
from .state import AppState


def get_state(request: Request) -> AppState:
    return request.app.state.aitu


def topic_name(slug) -> str | None:
    if slug is None or (isinstance(slug, float) and slug != slug):
        return None
    return config.TOPICS.get(str(slug), str(slug))
