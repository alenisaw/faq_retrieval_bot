"""База FAQ и темы: GET /api/faqs, /api/faqs/{id}, /api/topics, /api/methods, /api/health."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from ... import config
from ...preprocessing import normalize
from ..deps import get_state, topic_name
from ..schemas import Faq, FaqPage, Health, MethodInfo, Topic
from ..state import AppState

router = APIRouter(prefix="/api", tags=["faq"])


def _faq(row: dict) -> Faq:
    orig = row.get("orig_class")
    return Faq(faq_id=int(row["faq_id"]), question=row["question"], answer=row["answer"],
               topic=None if row.get("topic") is None else str(row["topic"]),
               topic_name=topic_name(row.get("topic")),
               answer_cluster=int(row["answer_cluster"]), is_dated=bool(row.get("is_dated", False)),
               orig_class=orig if isinstance(orig, str) and orig else None)


@router.get("/health", response_model=Health, summary="Состояние сервиса")
def health(state: AppState = Depends(get_state)) -> Health:
    return Health(status="ok", n_faqs=len(state.faq), n_clusters=int(state.faq["answer_cluster"].nunique()),
                  default_method=state.default_method, threshold=state.bot_config.get("threshold"),
                  database=state.database_ready(), loaded_methods=state.loaded_methods)


@router.get("/methods", response_model=list[MethodInfo], summary="Методы поиска")
def methods(state: AppState = Depends(get_state)) -> list[MethodInfo]:
    return [MethodInfo(**m) for m in state.methods()]


@router.get("/topics", response_model=list[Topic], summary="Темы и число FAQ в каждой")
def topics(state: AppState = Depends(get_state)) -> list[Topic]:
    counts = state.faq["topic"].astype(str).value_counts()
    return [Topic(slug=slug, name=name, n_faqs=int(counts.get(slug, 0))) for slug, name in config.TOPICS.items()]


@router.get("/faqs", response_model=FaqPage, summary="Список FAQ с фильтром по теме и тексту")
def faqs(topic: str | None = Query(None, description="slug темы"),
         q: str | None = Query(None, max_length=200, description="подстрока вопроса или ответа"),
         limit: int = Query(50, ge=1, le=200), offset: int = Query(0, ge=0),
         state: AppState = Depends(get_state)) -> FaqPage:
    df = state.faq
    if topic:
        if topic not in config.TOPICS:
            raise HTTPException(400, f"Неизвестная тема {topic!r}")
        df = df[df["topic"].astype(str) == topic]
    if q and normalize(q):
        needle = normalize(q)
        text = (df["question"] + " " + df["answer"]).map(normalize)
        df = df[text.str.contains(needle, regex=False)]
    df = df.sort_values("faq_id")
    page = df.iloc[offset: offset + limit]
    return FaqPage(total=len(df), limit=limit, offset=offset,
                   items=[_faq(r) for r in page.to_dict(orient="records")])


@router.get("/faqs/{faq_id}", response_model=Faq, summary="Один FAQ")
def faq(faq_id: int, state: AppState = Depends(get_state)) -> Faq:
    row = state.faq_row(faq_id)
    if row is None:
        raise HTTPException(404, f"FAQ {faq_id} не найден")
    return _faq(row)
