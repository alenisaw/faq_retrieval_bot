# AITU FAQ Retrieval Chatbot

Итоговый проект по дисциплине «Programming for Data Analysis and Database», тема 11.
Чат-бот находит лучший ответ из FAQ приёмной комиссии Astana IT University на вопрос
абитуриента. Сравниваются baseline и несколько методов поиска на тестовых запросах,
которых нет в базе; метод для бота выбирается по MRR на тесте.

Данные: [AITU Admissions Guide Dataset](https://huggingface.co/datasets/Arailym-tleubayeva/AITUAdmissionsGuideDataset)
(608 пар вопрос–ответ, русский, CC BY 4.0). Тема 11 — в списке исключений по объёму данных (требование 11).

Команда: Аян Анарбай (AI и предобработка), Акерке Орынбасарова (бэкенд, БД, деплой),
Ален Иссаев (фронтенд, EDA, отчёт). AAI-2602.

## Установка

Python 3.11+ (проверено на 3.12), CPU достаточно.

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate          # Windows;  source .venv/bin/activate — Linux/macOS
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU-сборка (в Colab пропустить)
pip install -r requirements.txt
```

CPU-сборка torch весит в разы меньше полной и для проекта её достаточно.

На Windows путь к папке `.venv` должен быть короче ~110 символов (например, репозиторий в
`C:\projects\`): внутри пакета torch есть файл с путём 130 символов от `site-packages`, и при
лимите Windows в 260 символов установка падает с `OSError: [Errno 2] No such file or directory`.
Другой вариант — включить поддержку длинных путей: https://pip.pypa.io/warnings/enable-long-paths

Ресурсы NLTK (`punkt_tab`, `stopwords`) скачиваются автоматически при первом запуске.
Модели `intfloat/multilingual-e5-small` и `paraphrase-multilingual-MiniLM-L12-v2`
(~470 МБ каждая) скачиваются с Hugging Face при первом использовании.

## Структура

```
backend/
├── data/raw/                 # исходный CSV
├── data/processed/           # faq_clean.parquet, faq.db, cache/ (эмбеддинги)
├── eval/                     # test_queries.csv, qrels.csv
├── src/
│   ├── config.py             # пути, темы, общие настройки
│   ├── preprocessing.py      # нормализация, токенизация, стоп-слова, stem / lemma
│   ├── retrievers/           # методы поиска с общим интерфейсом fit / search
│   ├── evaluation.py         # метрики, протоколы A и B, ресурсы, порог, confusion matrix
│   ├── classifier.py         # классификатор тем: Dummy / TF-IDF+LR / e5+LR
│   ├── experiments.py        # все эксперименты AI-модуля одним запуском
│   └── chatbot.py            # чат-бот поверх любого метода
├── tests/
└── requirements.txt
reports/                      # metrics/, figures/
```

## AI-модуль: запуск

Все команды — из папки `backend/`. Нужен `data/processed/faq_clean.parquet`
(пункт 1 меню: загрузка и очистка данных).

```bash
# сравнение методов: протокол B (leave-one-out по перефразам, без ручной разметки)
python -m src.evaluation --protocol loo --methods all

# все эксперименты: предобработка, методы, абляции, порог, классификатор, фильтр по теме
python -m src.experiments --protocol loo        # пока нет ручного тест-набора
python -m src.experiments --protocol test       # основной протокол A, сохраняет выбор метода для бота

# чат-бот в консоли (метод и порог берутся из models/chatbot.json)
python -m src.chatbot
python -m src.chatbot --method e5_small --ask "общага есть?"

# тесты
python -m pytest
```

Методы (`src/retrievers/__init__.py`, реестр `METHODS`):

| № | Имя | Метод |
|---|-----|-------|
| 0 | `rule_based` | ключевые слова → тема + Jaccard по леммам внутри темы (baseline) |
| 1 | `bow` | мешок слов (леммы) + косинус |
| 2 | `tfidf_word_lemma`, `tfidf_word_stem` | TF-IDF по словам, 1–2-граммы |
| 3 | `tfidf_char` | TF-IDF по символьным n-граммам 2–5 (`char_wb`) |
| 4 | `e5_small`, `minilm` | sentence embeddings + косинус |
| 5 | `hybrid_rrf` | Reciprocal Rank Fusion лексического метода и e5 |
| + | `bm25` | Okapi BM25 по леммам |

Метрики на уровне `answer_cluster`: P@1, Hit@3, Hit@5 (= Recall@k), P@3/P@5, MRR@10;
ресурсы: время индексации, средняя и p95 задержка, пиковая память, размер индекса и модели.

## AI-модуль: интеграция с остальными частями

**Данные (`data_loader.py`).** Поиск и оценка ждут `faq_clean.parquet` с колонками
`faq_id, question, answer, answer_cluster` (обязательно) и `orig_class, topic, is_dated`.
`answer_cluster` объединяет FAQ с одним ответом и перефразы одного вопроса;
`topic` — slug из `config.TOPICS`.

**Меню (`main.py`).**
- пункт 3: `build_retriever(name).fit(faq_df)` для каждого имени из `METHODS`;
- пункт 4: `run_benchmark(...)` или `experiments.run_all("test")`; confusion matrix —
  `topic_confusion` + `plot_confusion_matrix`;
- пункт 5: `ChatBot.from_saved(faq_df)` и `chat_loop(bot)` либо `bot.answer(text)`.

**БД (`db.py`).** `RunResult.run_record()` → строка таблицы `runs`, `RunResult.results` →
таблица `results` (`query_id, rank, faq_id, score, latency_ms`); `TfidfWordRetriever.top_terms()`
→ таблица `keywords`; колбэк `ChatBot(on_answer=...)` — запись в `chat_log`.

**API.** `ChatResponse.to_dict()` — готовый JSON ответа бота.

**EDA и error analysis.** `reports/metrics/per_query_<protocol>_<method>.csv` — по строке на
запрос: текст, gold-кластер и тема, топ-1 ответ, ранг правильного, score, задержка.
Эмбеддинги FAQ для UMAP: `build_retriever("e5_small").fit(faq_df).doc_embeddings`.
Сравнение режимов предобработки: `preprocessing.compare_modes(texts)`.
