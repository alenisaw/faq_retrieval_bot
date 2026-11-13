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

## Быстрый старт

После установки — одна команда из папки `backend/`:

```bash
python main.py --setup     # очистка данных -> faq_clean.parquet, faq.db; оценка методов без нейросетей
python main.py             # консольное меню (пункты 1–7)
python main.py --ask "общага есть?"
python main.py --serve     # веб-API: http://localhost:8000/docs
```

Или всё в Docker (из корня репозитория, Python ставить не нужно):

```bash
docker compose up --build                         # API на http://localhost:8000
docker compose --profile frontend up --build      # + фронтенд на http://localhost:3000
docker compose down -v                            # сбросить журнал бота и прогоны
```

## Структура

```
backend/
├── data/raw/                 # исходный CSV и его копия faq_raw.parquet
├── data/processed/           # faq_clean.parquet, faq.db, cleaning_log.json,
│                             # topic_review.csv, cluster_review.csv, cache/ (эмбеддинги)
├── eval/                     # test_queries.csv, qrels.csv, GUIDELINES.md, drafts/, reviews/
├── sql/
│   ├── schema.sql            # 11 таблиц + 2 представления
│   └── queries.sql           # 11 именованных SQL-отчётов
├── src/
│   ├── config.py             # пути, темы, общие настройки
│   ├── data_loader.py        # аудит и очистка -> faq_clean.parquet + cleaning_log.json
│   ├── topic_map.py          # 422 исходных класса -> 12 тем (правила + ручные исправления)
│   ├── entities.py           # сущности: URL, PHONE, EMAIL, MONEY, DATE (regex), ORG, LOC (spaCy)
│   ├── db.py                 # SQLite: схема, загрузка корпуса, прогоны, журнал бота, отчёты
│   ├── pipeline.py           # очистка -> faq.db одной командой
│   ├── api/                  # FastAPI: app.py, schemas.py, state.py, routes/
│   ├── preprocessing.py      # нормализация, токенизация, стоп-слова, stem / lemma
│   ├── retrievers/           # методы поиска с общим интерфейсом fit / search
│   ├── evaluation.py         # метрики, протоколы A и B, ресурсы, порог, confusion matrix
│   ├── classifier.py         # классификатор тем: Dummy / TF-IDF+LR / e5+LR
│   ├── experiments.py        # все эксперименты AI-модуля одним запуском
│   ├── annotate.py           # разметка тест-набора, перекрёстная проверка, kappa, заморозка
│   ├── chatbot.py            # чат-бот поверх любого метода
│   └── eda.py                # EDA: графики в reports/figures, сводка eda_summary.json
├── tests/
├── main.py                   # консольное меню
├── Dockerfile
└── requirements.txt
reports/                      # metrics/, figures/, sql/
frontend/                     # веб-интерфейс на Next.js (frontend/README.md)
docker-compose.yml
```

## Данные: аудит и очистка

`python -m src.data_loader` (или пункт 1 меню). Каждое правило пишет в
`data/processed/cleaning_log.json`, какие строки оно задело; сводная таблица —
`reports/metrics/cleaning_audit.csv`.

| Правило | Что делает | Строк |
|---|---|---|
| `header_row` | повтор заголовка внутри файла — удалить | 1 |
| `broken_csv_row` | строка разрезана запятой, хвост ответа в колонке класса — удалить (нет вопроса / ответ оборван) | 3 |
| `answer_in_question` | «Где кабинет 1с?, ответ: …» — разнести по колонкам | 1 |
| `question_prefix` | «Вопрос 1: …» — убрать префикс | 1 |
| `whitespace_quotes` | strip, внешние кавычки, литеральные `\n` | 180 |
| `answer_is_slug` | вместо ответа идентификатор (`AITU_iot`) — удалить | 10 |
| `translation_not_answer` | пункт учебного плана + его перевод на английский — удалить | 21 |
| `empty_answer` | «…не представлена в данном контексте» — удалить | 1 |
| `duplicate_pair` | одинаковая пара вопрос–ответ после нормализации — удалить | 37 |

Итог: 608 → 535 строк, 470 кластеров ответов (50 — с двумя и более вопросами), 12 тем.

- **`faq_id`** — индекс строки в исходном CSV: по нему любую строку можно найти в исходнике.
- **`answer_cluster`** объединяет одинаковые ответы, одинаковые вопросы и перефразы из
  `MANUAL_CLUSTER_LINKS`. Короткие ответы («85 баллов.») склеиваются только при одинаковом
  исходном классе — иначе Computer Science и Software Engineering стали бы одним кластером.
  Номер кластера = минимальный `faq_id` в нём, поэтому он стабилен при пересборке;
  на него ссылается `eval/qrels.csv`.
- **`topic`** — правила из `src/topic_map.py`: сначала по вопросу, потом по классу, по
  умолчанию «об университете». У всех строк кластера одна тема. Для ручной проверки —
  `data/processed/topic_review.csv` (колонка `topic_source` показывает сработавшее правило),
  исправления — в `TOPIC_OVERRIDES`. Колледж и Foundation отнесены к «процессу поступления».
- **`is_dated`** — ответ упоминает 2020–2025 год, дату или «Нур-Султан» (17 строк);
  бот добавляет к таким ответам предупреждение.

## База данных и SQL

`data/processed/faq.db` (SQLite) строится пунктом 1 меню. Схема — `backend/sql/schema.sql`:

| Таблица | Содержимое |
|---|---|
| `topics`, `answer_clusters`, `faqs` | корпус: 12 тем, кластеры ответов, 535 FAQ |
| `keywords` | топ-10 термов каждого FAQ по TF-IDF (леммы) |
| `entities` | ORG, LOC, URL, PHONE, EMAIL, MONEY, DATE |
| `queries`, `qrels` | ручной тест-набор из `eval/` (загружается, когда файлы появятся) |
| `methods`, `runs`, `results` | конфигурации методов, прогоны с ресурсами, топ-10 выдачи каждого запроса |
| `chat_log` | вопросы к боту из меню и API: метод, топ-1, score, был ли отказ |

Представления `v_result_clusters` и `v_query_outcome` сворачивают выдачу до кластеров,
как это делает `src/evaluation.py`, поэтому MRR из SQL совпадает с посчитанным в Python.

SQL-отчёты (`backend/sql/queries.sql`, пункт 6 меню, `GET /api/reports/{name}`):

| Имя | Что показывает | Приёмы |
|---|---|---|
| `mrr_by_method` | MRR@10, P@1, Hit@3/5 по методам | CTE, JOIN, представление |
| `p1_by_topic` | P@1 по темам и методам | JOIN, GROUP BY, HAVING |
| `method_rank_in_topic` | место метода в каждой теме | `RANK() OVER (PARTITION BY …)` |
| `top_keywords_per_topic` | топ-5 слов темы по сумме TF-IDF | `ROW_NUMBER() OVER` |
| `duplicate_answers` | один ответ на несколько вопросов | GROUP BY, HAVING, GROUP_CONCAT |
| `hard_queries` | запросы, где ошиблись все методы | `NOT EXISTS` |
| `dated_answers` | ответы с датами и суммами | JOIN `faqs` × `entities` |
| `latency_by_method` | средняя и p95 задержка | `NTILE(20)` |
| `mrr_change_between_runs` | изменение MRR между прогонами | `LAG()` |
| `chat_topics` | частые темы и доля отказов в журнале бота | LEFT JOIN, `SUM() OVER ()` |
| `resources_by_method` | время индексации, память, размер | CTE, JOIN |

```python
from src import db
with db.connect() as conn:
    df = db.run_report(conn, "mrr_by_method")    # pandas.DataFrame
```

## Веб-API

`python main.py --serve` или `uvicorn src.api.app:app --reload` (из `backend/`).
Интерактивная документация — http://localhost:8000/docs.

| Метод | Путь | Что делает |
|---|---|---|
| POST | `/api/chat` | `{"query": "общага есть?", "method": null, "k": 3}` → ответ бота, топ-k разных ответов со score |
| POST | `/api/compare` | один вопрос сразу несколькими методами (по умолчанию — все лексические) |
| GET | `/api/methods` | методы, метод по умолчанию, какие индексы уже построены |
| GET | `/api/topics` | 12 тем и число FAQ |
| GET | `/api/faqs?topic=&q=&limit=&offset=` | база FAQ с фильтрами |
| GET | `/api/faqs/{id}` | один FAQ |
| GET | `/api/metrics?protocol=loo\|test` | метрики последнего прогона каждого метода |
| GET | `/api/reports`, `/api/reports/{name}` | список SQL-отчётов и результат одного |
| GET | `/api/figures`, `/figures/{file}` | графики из `reports/figures` |
| GET | `/api/health` | состояние сервиса |

Метод по умолчанию и порог отказа берутся из `models/chatbot.json` (его сохраняет
`src.experiments --protocol test`); пока файла нет — `tfidf_char` без порога. Индекс
каждого метода строится при первом запросе к нему и кэшируется. Вопросы к `/api/chat`
пишутся в `chat_log`. Переменные окружения: `API_CORS_ORIGINS` (по умолчанию
`http://localhost:3000`), `API_WARMUP_METHODS` (методы, которые строятся при старте).

## Фронтенд

Next.js 16 + TypeScript + Tailwind в папке `frontend/`, подробности — `frontend/README.md`.

```bash
cd frontend && npm install && npm run dev      # http://localhost:3000 (API должен работать на :8000)
```

| Страница | Что на ней |
|---|---|
| `/` | вопрос → ответ с маркером на совпавших словах, тема, совпадение, метод; похожие ответы; темы базы |
| `/compare` | один вопрос всеми методами сразу, одинаковые ответы подсвечены одним цветом |
| `/insights` | метрики методов, SQL-отчёты, графики EDA |

Консольное меню остаётся обязательным интерфейсом по требованию 7; веб-интерфейс — расширение поверх того же API.

## EDA

`python -m src.eda` (или пункт 2 меню) строит графики в `reports/figures/` и сводку `reports/eda_summary.json`:
темы, длины текстов с выбросами по IQR, частые слова до и после стоп-слов, биграммы и триграммы,
облака слов по темам, размер словаря raw / stem / lemma (`reports/metrics/eda_vocab.csv`), размеры
кластеров ответов. Графики видны на странице `/insights`.

## AI-модуль: запуск

Все команды — из папки `backend/`. Нужен `data/processed/faq_clean.parquet`
(пункт 1 меню: загрузка и очистка данных).

```bash
# сравнение методов: протокол B (leave-one-out по перефразам, без ручной разметки)
python -m src.evaluation --protocol loo --methods all

# все эксперименты: предобработка, методы, абляции, порог, классификатор, фильтр по теме
python -m src.experiments --protocol loo        # протокол B: перефразы из самой базы
python -m src.experiments --protocol test       # основной протокол A, сохраняет выбор метода для бота

# разбор ошибок выбранного метода на тесте -> reports/errors/, reports/figures/err_*.png
python -m src.error_analysis

# чат-бот в консоли (метод и порог берутся из models/chatbot.json)
python -m src.chatbot
python -m src.chatbot --method e5_small --ask "общага есть?"

# тесты
python -m pytest
```

Прогоны `src.experiments` пишутся в `reports/metrics/` и в таблицы `runs`/`results` базы `faq.db`.

### Тестовый набор (протокол A)

Правила — [backend/eval/GUIDELINES.md](backend/eval/GUIDELINES.md). Всё делается через
`src.annotate`, файлы руками не правятся:

```bash
python -m src.annotate --author ayan                    # 1. каждый размечает ~60 + 10 своих запросов
python -m src.annotate --import eval/drafts/<файл>.csv  #    или добавляет проверенный черновик
python -m src.annotate --author ayan --review alen      # 2. независимая разметка чужих запросов
python -m src.annotate --agreement eval/reviews/qrels_review_ayan_on_alen.csv   # Cohen's kappa
python -m src.annotate --check                          # 3. статистика по авторам и темам
python -m src.annotate --freeze                         # 4. один раз: dev 30% / test 70%
python -m src.experiments --protocol test               # 5. финальные цифры и выбор метода для бота
```

Текущий набор: 211 запросов (181 по теме, 30 не по теме), по 15 на каждую из 12 тем,
заморожен: dev 58 / test 153. Запросы — черновики LLM в разговорном стиле абитуриентов
(`origin=llm_draft`), к эталону добавлены почти одинаковые ответы-дубли из базы.

### Результаты (тест, 132 запроса по теме)

| Метод | P@1 | Hit@3 | MRR@10 |
|---|---|---|---|
| TF-IDF, символы (выбран для бота) | 0.538 | 0.712 | 0.638 |
| TF-IDF, символы, вопрос+ответ | 0.500 | 0.667 | 0.614 |
| Правила (baseline) | 0.462 | 0.621 | 0.559 |
| BM25 | 0.462 | 0.583 | 0.542 |
| Мешок слов | 0.439 | 0.591 | 0.532 |
| TF-IDF, стеммы / леммы | 0.432 | 0.591 / 0.568 | 0.528 / 0.523 |

Порог отказа подобран на dev (0.50): на тесте бот ответил бы на 3 из 21 вопроса не по теме.
На LOO (84 запроса) нейросетевые методы впереди: e5-small с индексом «вопрос+ответ» —
P@1 0.726, MRR 0.787, но 11 мс на запрос против 1 мс у TF-IDF. На тесте их ещё нужно
прогнать там, где доступен Hugging Face: `python -m src.experiments --protocol test`.

Разбор ошибок — [reports/errors/error_analysis.md](reports/errors/error_analysis.md):
из 61 промаха top-1 у 20 правильный ответ на 2–3 месте, у 17 его нет в топ-10, 5 — равноценные
ответы-дубли; на запросах со сленгом P@1 0.49 против 0.56 без него.

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

Пункт 4 в варианте `experiments` и `src.experiments` сами пишут прогоны в `faq.db`
(`experiments.save_runs_to_db`); тест-набор перезаписывается в базе, только если он изменился.

**БД (`db.py`).** `RunResult.run_record()` → строка таблицы `runs`, `RunResult.results` →
таблица `results` (`query_id, rank, faq_id, score, latency_ms`); `TfidfWordRetriever.top_terms()`
→ таблица `keywords`; колбэк `ChatBot(on_answer=...)` — запись в `chat_log`.

**API.** `ChatResponse.to_dict()` — готовый JSON ответа бота.

**EDA и error analysis.** `reports/metrics/per_query_<protocol>_<method>.csv` — по строке на
запрос: текст, gold-кластер и тема, топ-1 ответ, ранг правильного, score, задержка.
Эмбеддинги FAQ для UMAP: `build_retriever("e5_small").fit(faq_df).doc_embeddings`.
Сравнение режимов предобработки: `preprocessing.compare_modes(texts)`.
