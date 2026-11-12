"""Pydantic-схемы запросов и ответов API (их же видит фронтенд в /docs и lib/api.ts)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500, examples=["общага есть?"])
    method: str | None = Field(None, description="метод из GET /api/methods; по умолчанию — выбранный для бота")
    k: int = Field(3, ge=1, le=10, description="сколько разных ответов вернуть")

    @field_validator("query")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("пустой вопрос")
        return v.strip()


class Hit(BaseModel):
    faq_id: int
    question: str
    answer: str
    score: float
    answer_cluster: int | None = None
    topic: str | None = None
    topic_name: str | None = None
    is_dated: bool = False


class ChatResponse(BaseModel):
    query: str
    answered: bool
    message: str
    hits: list[Hit]
    method: str
    threshold: float | None = None
    predicted_topic: str | None = None
    latency_ms: float


class CompareRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    methods: list[str] | None = Field(None, max_length=10,
                                      description="по умолчанию — все методы без нейросетей")
    k: int = Field(3, ge=1, le=10)

    @field_validator("query")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("пустой вопрос")
        return v.strip()


class CompareResponse(BaseModel):
    query: str
    results: list[ChatResponse]
    errors: dict[str, str] = Field(default_factory=dict)


class MethodInfo(BaseModel):
    name: str
    family: Literal["lexical", "neural"]
    is_default: bool
    loaded: bool
    description: str


class Topic(BaseModel):
    slug: str
    name: str
    n_faqs: int


class Faq(BaseModel):
    faq_id: int
    question: str
    answer: str
    topic: str | None = None
    topic_name: str | None = None
    answer_cluster: int
    is_dated: bool = False
    orig_class: str | None = None


class FaqPage(BaseModel):
    total: int
    limit: int
    offset: int
    items: list[Faq]


class ReportInfo(BaseModel):
    name: str
    title: str
    description: str


class ReportResult(ReportInfo):
    columns: list[str]
    rows: list[dict[str, Any]]


class MetricsTable(BaseModel):
    protocol: str | None
    columns: list[str]
    rows: list[dict[str, Any]]


class Health(BaseModel):
    status: Literal["ok"]
    n_faqs: int
    n_clusters: int
    default_method: str
    threshold: float | None
    database: bool
    loaded_methods: list[str]
