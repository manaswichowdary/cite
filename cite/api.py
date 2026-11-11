"""Web server. GET /ask?q=... streams progress as server sent events; GET / serves the page.

Two protections keep one visitor from running up the bill: a per minute limit for each address,
and a daily spending cap for the whole server.
"""

from __future__ import annotations

import json
import queue
import threading
import time
from datetime import date
from pathlib import Path
from typing import Callable

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, StreamingResponse

from .config import Settings, load_settings
from .pipeline import Result

WEB_PAGE = Path(__file__).resolve().parent.parent / "web" / "index.html"
Runner = Callable[[str, Callable[[dict], None]], Result]


class SpendTracker:
    """Adds up spend for today and says when the daily cap is reached."""

    def __init__(self, cap_usd: float, today: Callable[[], date] = date.today):
        self.cap, self._today, self._day, self._spent = cap_usd, today, today(), 0.0
        self._lock = threading.Lock()

    def _roll(self) -> None:
        if self._today() != self._day:
            self._day, self._spent = self._today(), 0.0

    def add(self, usd: float) -> None:
        with self._lock:
            self._roll()
            self._spent += usd

    def exceeded(self) -> bool:
        with self._lock:
            self._roll()
            return self._spent >= self.cap

    @property
    def spent(self) -> float:
        with self._lock:
            self._roll()
            return self._spent


class RateLimiter:
    """At most `limit` requests per address in any 60 seconds."""

    def __init__(self, limit: int = 5, clock: Callable[[], float] = time.monotonic):
        self.limit, self._clock, self._seen = limit, clock, {}
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        with self._lock:
            now = self._clock()
            recent = [t for t in self._seen.get(key, []) if now - t < 60]
            if len(recent) >= self.limit:
                self._seen[key] = recent
                return False
            recent.append(now)
            self._seen[key] = recent
            return True


def default_runner(settings: Settings) -> Runner:
    from .agent import make_client      # built on first use, so importing this module needs no keys
    from .pipeline import run_question
    from .search import Searcher
    from .trace import write_trace

    state: dict = {}
    lock = threading.Lock()

    def run(question: str, emit: Callable[[dict], None]) -> Result:
        with lock:
            if not state:
                state["client"], state["searcher"] = make_client(settings), Searcher(settings)
        result = run_question(question, settings, state["client"], state["searcher"], emit)
        write_trace(result)
        return result

    return run


def create_app(settings: Settings | None = None, run: Runner | None = None,
               limiter: RateLimiter | None = None, tracker: SpendTracker | None = None) -> FastAPI:
    settings = settings or load_settings()
    app = FastAPI(title="Cite")
    limiter = limiter or RateLimiter()
    tracker = tracker or SpendTracker(settings.daily_cap_usd)
    runner: Runner | None = run

    @app.get("/health")
    def health() -> dict:
        return {"ok": True, "spent_today_usd": round(tracker.spent, 4), "daily_cap_usd": tracker.cap}

    @app.get("/")
    def index():
        return FileResponse(WEB_PAGE)

    @app.get("/ask")
    def ask(request: Request, q: str = Query(..., min_length=3, max_length=300)):
        nonlocal runner
        ip = request.client.host if request.client else "unknown"
        if not limiter.allow(ip):
            raise HTTPException(429, "Too many questions. Wait a minute and try again.")
        if tracker.exceeded():
            raise HTTPException(429, "The daily spending cap is reached. Try again tomorrow.")
        if runner is None:
            if not (settings.has_openai and settings.has_tavily):
                raise HTTPException(503, "The server is missing its API keys.")
            runner = default_runner(settings)
        active = runner

        def stream():
            events: queue.Queue = queue.Queue()

            def worker() -> None:
                try:
                    tracker.add(active(q, events.put).cost_usd)
                except Exception as err:   # never send details to the browser: they can hold secrets
                    print(f"ask failed: {type(err).__name__}")
                    events.put({"type": "error", "message": "Something went wrong. Please try again."})
                finally:
                    events.put(None)

            threading.Thread(target=worker, daemon=True).start()
            while (event := events.get()) is not None:
                yield f"event: {event['type']}\ndata: {json.dumps(event)}\n\n"

        return StreamingResponse(stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    return app


app = None   # `uvicorn cite.api:create_app --factory` builds the real app


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(create_app(), host="127.0.0.1", port=8000)
