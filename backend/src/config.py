"""Общие настройки проекта: пути к данным, артефактам и словарь тем.

Файл общий для всех модулей. Владелец — бэкенд (Акерке); AI-модуль (Аян)
добавил сюда только то, что нужно поиску и оценке. Новые настройки
дописывайте сюда же, чтобы пути не дублировались по модулям.
"""

from __future__ import annotations

import os
from pathlib import Path

# --- Пути ---------------------------------------------------------------
BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = BACKEND_DIR.parent

DATA_DIR = BACKEND_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CACHE_DIR = PROCESSED_DIR / "cache"          # кэш эмбеддингов (в .gitignore)
MODELS_DIR = BACKEND_DIR / "models"          # joblib-модели (в .gitignore)

RAW_CSV_PATH = RAW_DIR / "QA_addmissionAitu.csv"
RAW_PARQUET_PATH = RAW_DIR / "faq_raw.parquet"           # тот же CSV без изменений, в parquet
FAQ_CLEAN_PATH = PROCESSED_DIR / "faq_clean.parquet"
DB_PATH = PROCESSED_DIR / "faq.db"
CLEANING_LOG_PATH = PROCESSED_DIR / "cleaning_log.json"  # что и почему удалено/исправлено
TOPIC_REVIEW_PATH = PROCESSED_DIR / "topic_review.csv"   # для ручной проверки словаря тем
CLUSTER_REVIEW_PATH = PROCESSED_DIR / "cluster_review.csv"

SQL_DIR = BACKEND_DIR / "sql"
SCHEMA_SQL_PATH = SQL_DIR / "schema.sql"
QUERIES_SQL_PATH = SQL_DIR / "queries.sql"

EVAL_DIR = BACKEND_DIR / "eval"
TEST_QUERIES_PATH = EVAL_DIR / "test_queries.csv"
QRELS_PATH = EVAL_DIR / "qrels.csv"

REPORTS_DIR = PROJECT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_DIR = REPORTS_DIR / "metrics"
ERRORS_DIR = REPORTS_DIR / "errors"

RANDOM_SEED = 42

# --- Контракт очищенных данных -----------------------------------------
# Колонки faq_clean.parquet, на которые опираются поиск, оценка и бот.
FAQ_REQUIRED_COLUMNS = ("faq_id", "question", "answer", "answer_cluster")
FAQ_OPTIONAL_COLUMNS = ("orig_class", "topic", "is_dated")

# --- Крупные темы (черновик, утверждаем втроём) -------------------------
# Ключ — slug, который пишется в колонку `topic` и в qrels.csv.
# Кандидат на 13-ю тему: "college" (колледж AITU + Foundation, ~20 строк).
TOPICS: dict[str, str] = {
    "about": "Об университете",
    "admission": "Процесс поступления",
    "scores_grants": "Баллы, предметы ЕНТ и гранты",
    "programs": "Специальности бакалавриата",
    "tuition": "Стоимость, оплата и скидки",
    "dormitory": "Общежитие",
    "creative_exam": "Творческий экзамен",
    "military": "Военная кафедра",
    "masters": "Магистратура и докторантура",
    "study_process": "Учебный процесс",
    "contacts": "Контакты и график работы",
    "student_life": "Студенческая жизнь и карьера",
}

# --- API -----------------------------------------------------------------
# Откуда фронтенд может ходить в API (CORS). Через запятую в переменной окружения.
API_CORS_ORIGINS = [o.strip() for o in os.environ.get(
    "API_CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(",") if o.strip()]
# Методы, которые API строит при старте (остальные — лениво, при первом запросе).
API_WARMUP_METHODS = [m.strip() for m in os.environ.get("API_WARMUP_METHODS", "").split(",") if m.strip()]

# --- Чат-бот -----------------------------------------------------------
# Ответ при отказе (score топ-1 ниже порога). Контакты взяты из самого FAQ.
FALLBACK_MESSAGE = (
    "Не нашёл точного ответа на этот вопрос в базе FAQ. "
    "Обратитесь в приёмную комиссию AITU: +7 (7172) 645-710, "
    "info@astanait.edu.kz, сайт https://admission.astanait.edu.kz"
)
