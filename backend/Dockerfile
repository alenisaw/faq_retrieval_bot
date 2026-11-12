# Бэкенд: FastAPI + AI-модуль, только CPU.
#   docker build -t aitu-faq-backend .   (из папки backend/)
# Обычно запускается через docker-compose.yml в корне репозитория.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/srv/.cache/huggingface \
    NLTK_DATA=/srv/.cache/nltk

# Код лежит в /srv/backend, поэтому reports/ проекта — это /srv/reports (см. config.py)
WORKDIR /srv/backend

# CPU-сборка torch в разы меньше полной; ставим её до requirements, чтобы
# sentence-transformers не потянул CUDA-версию
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install -r requirements.txt \
    && python -c "import nltk; [nltk.download(r, download_dir='/srv/.cache/nltk', quiet=True) for r in ('punkt_tab', 'stopwords')]"

COPY . .

RUN useradd --create-home --uid 1000 app \
    && mkdir -p /srv/reports /srv/backend/models /srv/backend/data/processed \
    && chown -R app:app /srv
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=4)"

# Свежий том без данных -> собрать faq_clean.parquet и faq.db, затем поднять API
CMD ["sh", "-c", "[ -f data/processed/faq.db ] || python -m src.pipeline; exec uvicorn src.api.app:app --host 0.0.0.0 --port 8000"]
