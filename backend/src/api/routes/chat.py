"""POST /api/chat — ответ бота; POST /api/compare — один вопрос сразу несколькими методами."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from ...retrievers import LEXICAL_METHODS, METHODS
from ..deps import get_state, topic_name
from ..schemas import ChatRequest, ChatResponse, CompareRequest, CompareResponse
from ..state import AppState

router = APIRouter(prefix="/api", tags=["chat"])


def to_schema(resp) -> ChatResponse:
    d = resp.to_dict()
    for h in d["hits"]:
        h["topic_name"] = topic_name(h.get("topic"))
        h["faq_id"] = int(h["faq_id"])
        h["answer_cluster"] = None if h["answer_cluster"] is None else int(h["answer_cluster"])
    return ChatResponse(**d)


def _check_method(method: str | None) -> None:
    if method is not None and method not in METHODS:
        raise HTTPException(400, f"Неизвестный метод {method!r}. Доступны: {', '.join(METHODS)}")


@router.post("/chat", response_model=ChatResponse, summary="Ответ чат-бота")
def chat(req: ChatRequest, state: AppState = Depends(get_state)) -> ChatResponse:
    _check_method(req.method)
    try:
        resp = state.answer(req.query, method=req.method, k=req.k)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except (OSError, RuntimeError) as exc:  # модель не скачалась, нет сети и т.п.
        raise HTTPException(503, f"Метод {req.method or state.default_method} недоступен: {exc}") from exc
    return to_schema(resp)


@router.post("/compare", response_model=CompareResponse, summary="Сравнение методов на одном вопросе")
def compare(req: CompareRequest, state: AppState = Depends(get_state)) -> CompareResponse:
    methods = req.methods or list(LEXICAL_METHODS)
    for m in methods:
        _check_method(m)
    results, errors = [], {}
    for m in dict.fromkeys(methods):
        try:
            results.append(to_schema(state.answer(req.query, method=m, k=req.k, log=False)))
        except (OSError, RuntimeError, ValueError) as exc:
            errors[m] = str(exc)
    return CompareResponse(query=req.query, results=results, errors=errors)
