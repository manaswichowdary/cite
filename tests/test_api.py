import json
from datetime import date, timedelta

from fastapi.testclient import TestClient

from cite.api import RateLimiter, SpendTracker, create_app
from cite.config import Settings
from cite.pipeline import run_question
from tests.fakes import FakeClient, FakeSearcher, Q1, Q2

SETTINGS = Settings(openai_api_key="k", tavily_api_key="t", daily_cap_usd=0.5)
ANSWER = "- DuckDB runs analytical SQL queries on local files without a separate server process [1]."


def fake_run(question, emit):
    client = FakeClient(ANSWER, [{"part": "a", "quote": Q1}, {"part": "b", "quote": Q2}])
    return run_question(question, SETTINGS, client, FakeSearcher(), emit)


def parse(text):
    events = []
    for block in text.strip().split("\n\n"):
        name, data = block.split("\n")
        events.append((name.removeprefix("event: "), json.loads(data.removeprefix("data: "))))
    return events


def make(run=fake_run, **kw):
    return TestClient(create_app(SETTINGS, run, **kw))


def test_streams_events_in_order():
    r = make().get("/ask", params={"q": "What is DuckDB?"})
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    names = [n for n, _ in parse(r.text)]
    assert names == ["search", "sources", "answer", "claim", "done"]


def test_page_and_health_are_served():
    c = make()
    assert "<title>Cite</title>" in c.get("/").text
    assert c.get("/health").json()["daily_cap_usd"] == 0.5


def test_rejects_short_questions():
    assert make().get("/ask", params={"q": "hi"}).status_code == 422


def test_rate_limit_per_address():
    t = [0.0]
    c = make(limiter=RateLimiter(limit=2, clock=lambda: t[0]))
    assert [c.get("/ask", params={"q": "question one"}).status_code for _ in range(3)] == [200, 200, 429]
    t[0] = 61.0
    assert c.get("/ask", params={"q": "question one"}).status_code == 200


def test_daily_cap_blocks_after_spend():
    tracker = SpendTracker(cap_usd=0.0001)
    c = make(tracker=tracker, limiter=RateLimiter(limit=100))
    assert c.get("/ask", params={"q": "first question"}).status_code == 200
    assert tracker.spent > 0.0001
    r = c.get("/ask", params={"q": "second question"})
    assert r.status_code == 429 and "daily" in r.json()["detail"]


def test_spend_tracker_resets_on_a_new_day():
    day = [date(2026, 1, 1)]
    tracker = SpendTracker(1.0, today=lambda: day[0])
    tracker.add(2.0)
    assert tracker.exceeded()
    day[0] += timedelta(days=1)
    assert not tracker.exceeded() and tracker.spent == 0.0


def test_errors_do_not_leak_details():
    def boom(question, emit):
        raise RuntimeError("secret sk-12345")

    r = make(run=boom).get("/ask", params={"q": "What is DuckDB?"})
    (name, data), = parse(r.text)
    assert name == "error" and "sk-12345" not in r.text and "Something went wrong" in data["message"]


def test_missing_keys_give_503_when_no_runner_is_set():
    c = TestClient(create_app(Settings(), None))
    assert c.get("/ask", params={"q": "What is DuckDB?"}).status_code == 503
