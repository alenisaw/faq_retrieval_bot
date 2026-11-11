"""Общие настройки проекта: пути к данным, артефактам и словарь тем.

Файл общий для всех модулей. Владелец — бэкенд (Акерке); AI-модуль (Аян)
добавил сюда только то, что нужно поиску и оценке. Новые настройки
дописывайте сюда же, чтобы пути не дублировались по модулям.
"""

from __future__ import annotations

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
FAQ_CLEAN_PATH = PROCESSED_DIR / "faq_clean.parquet"
DB_PATH = PROCESSED_DIR / "faq.db"

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

# --- Чат-бот -----------------------------------------------------------
# Ответ при отказе (score топ-1 ниже порога). Контакты взяты из самого FAQ.
FALLBACK_MESSAGE = (
    "Не нашёл точного ответа на этот вопрос в базе FAQ. "
    "Обратитесь в приёмную комиссию AITU: +7 (7172) 645-710, "
    "info@astanait.edu.kz, сайт https://admission.astanait.edu.kz"
)
