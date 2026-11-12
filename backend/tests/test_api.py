import pytest
from fastapi.testclient import TestClient

from src import db
from src.api.app import create_app
from src.evaluation import leave_one_out
from src.retrievers import build_retriever


@pytest.fixture()
def db_path(tmp_path, faq_df):
    path = tmp_path / "api.db"
    with db.connect(path) as conn:
        db.init_db(conn)
        db.load_corpus(conn, faq_df)
        db.save_run(conn, leave_one_out(build_retriever("tfidf_char"), faq_df, measure_memory=False))
    return path


@pytest.fixture()
def client(faq_df, db_path, tmp_path):
    app = create_app(faq_df=faq_df, db_path=db_path, bot_config_path=tmp_path / "no_config.json", warmup=[])
    with TestClient(app) as c:
        yield c


def test_health(client, faq_df):
    body = client.get("/api/health").json()
    assert body["status"] == "ok" and body["n_faqs"] == len(faq_df)
    assert body["default_method"] == "tfidf_char" and body["database"] is True
    assert "tfidf_char" in body["loaded_methods"]          # бот по умолчанию строится при старте


def test_chat_answers_and_logs(client, db_path):
    r = client.post("/api/chat", json={"query": "общага есть?", "k": 2})
    assert r.status_code == 200
    body = r.json()
    assert body["answered"] and body["hits"][0]["faq_id"] in (1, 2)
    assert body["hits"][0]["topic_name"] == "Общежитие"
    assert len(body["hits"]) == 2
    with db.connect(db_path) as conn:
        assert conn.execute("SELECT query, source FROM chat_log").fetchall() == [("общага есть?", "api")]


def test_chat_other_method(client):
    body = client.post("/api/chat", json={"query": "сколько стоит обучение", "method": "bm25"}).json()
    assert body["method"] == "bm25" and body["hits"][0]["topic"] == "tuition"


@pytest.mark.parametrize("payload, status", [
    ({"query": "   "}, 422),
    ({"query": ""}, 422),
    ({"query": "x" * 501}, 422),
    ({"query": "ok", "k": 0}, 422),
    ({"query": "ok", "method": "nope"}, 400),
])
def test_chat_validation(client, payload, status):
    assert client.post("/api/chat", json=payload).status_code == status


def test_compare(client, db_path):
    body = client.post("/api/compare", json={"query": "военная кафедра есть?",
                                             "methods": ["bow", "tfidf_char", "bow"]}).json()
    assert [r["method"] for r in body["results"]] == ["bow", "tfidf_char"]
    assert all(r["hits"][0]["topic"] == "military" for r in body["results"])
    with db.connect(db_path) as conn:   # сравнение не засоряет журнал бота
        assert conn.execute("SELECT COUNT(*) FROM chat_log").fetchone()[0] == 0


def test_faqs_filter_and_detail(client):
    body = client.get("/api/faqs", params={"topic": "dormitory"}).json()
    assert body["total"] == 3 and {f["topic"] for f in body["items"]} == {"dormitory"}
    assert client.get("/api/faqs", params={"q": "военная"}).json()["total"] == 2
    page = client.get("/api/faqs", params={"limit": 5, "offset": 10}).json()
    assert page["total"] == 12 and len(page["items"]) == 2
    assert client.get("/api/faqs", params={"topic": "nope"}).status_code == 400
    assert client.get("/api/faqs/3").json()["is_dated"] is True
    assert client.get("/api/faqs/999").status_code == 404


def test_topics_and_methods(client):
    topics = {t["slug"]: t["n_faqs"] for t in client.get("/api/topics").json()}
    assert len(topics) == 12 and topics["dormitory"] == 3 and topics["masters"] == 0
    methods = {m["name"]: m for m in client.get("/api/methods").json()}
    assert methods["tfidf_char"]["is_default"] and methods["e5_small"]["family"] == "neural"


def test_metrics_and_reports(client):
    m = client.get("/api/metrics").json()
    assert m["rows"][0]["method"] == "tfidf_char" and "mrr@10" in m["columns"]
    assert client.get("/api/metrics", params={"protocol": "bad"}).status_code == 422
    names = [r["name"] for r in client.get("/api/reports").json()]
    assert "mrr_by_method" in names and len(names) >= 10
    rep = client.get("/api/reports/mrr_by_method").json()
    assert rep["columns"][0] == "method" and rep["rows"][0]["method"] == "tfidf_char"
    assert client.get("/api/reports/unknown").status_code == 404
    assert isinstance(client.get("/api/figures").json(), list)


def test_reports_without_database(faq_df, tmp_path):
    app = create_app(faq_df=faq_df, db_path=tmp_path / "absent.db", bot_config_path=tmp_path / "x.json",
                     warmup=[])
    with TestClient(app) as c:
        assert c.get("/api/reports/mrr_by_method").status_code == 503
        assert c.post("/api/chat", json={"query": "общага есть?"}).status_code == 200   # бот работает и без БД
