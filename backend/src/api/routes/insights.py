"""Метрики и аналитика для страницы /insights: таблица методов, SQL-отчёты, графики EDA."""

from __future__ import annotations

import math

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query

from ... import config, db
from ..deps import get_state
from ..schemas import MetricsTable, ReportInfo, ReportResult
from ..state import AppState

router = APIRouter(prefix="/api", tags=["insights"])


def _records(df: pd.DataFrame) -> list[dict]:
    """DataFrame -> JSON-совместимые строки (NaN -> null, numpy -> python)."""
    out = []
    for row in df.to_dict(orient="records"):
        clean = {}
        for k, v in row.items():
            if hasattr(v, "item"):
                v = v.item()
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                v = None
            clean[str(k)] = v
        out.append(clean)
    return out


def _require_db(state: AppState) -> None:
    if not state.database_ready():
        raise HTTPException(503, "База faq.db не создана: выполните пункт 1 меню (python main.py --setup)")


@router.get("/metrics", response_model=MetricsTable, summary="Метрики последнего прогона каждого метода")
def metrics(protocol: str | None = Query(None, pattern="^(loo|test|dev|all)$"),
            state: AppState = Depends(get_state)) -> MetricsTable:
    _require_db(state)
    with db.connect(state.db_path) as conn:
        df = db.latest_metrics(conn, protocol)
    return MetricsTable(protocol=protocol, columns=list(df.columns), rows=_records(df))


@router.get("/reports", response_model=list[ReportInfo], summary="Список SQL-отчётов")
def reports() -> list[ReportInfo]:
    return [ReportInfo(name=r.name, title=r.title, description=r.description) for r in db.load_reports().values()]


@router.get("/reports/{name}", response_model=ReportResult, summary="Выполнить SQL-отчёт")
def report(name: str, state: AppState = Depends(get_state)) -> ReportResult:
    _require_db(state)
    all_reports = db.load_reports()
    if name not in all_reports:
        raise HTTPException(404, f"Нет отчёта {name!r}")
    r = all_reports[name]
    with db.connect(state.db_path) as conn:
        df = pd.read_sql(r.sql, conn)
    return ReportResult(name=r.name, title=r.title, description=r.description,
                        columns=list(df.columns), rows=_records(df))


@router.get("/figures", response_model=list[str], summary="Графики EDA и оценки (PNG/HTML)")
def figures() -> list[str]:
    if not config.FIGURES_DIR.exists():
        return []
    return sorted(p.name for p in config.FIGURES_DIR.iterdir() if p.suffix.lower() in (".png", ".svg", ".html"))
