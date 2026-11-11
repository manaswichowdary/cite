"""One question, start to finish: search and answer, then check the citations."""

from __future__ import annotations

import importlib.util
import sys
import threading
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable

from .agent import Agent, AgentAnswer
from .checker import CheckReport, OpenAIJudge, check_answer
from .config import Settings
from .retrieval import Retriever, make_retriever

_retrievers: dict[str, Retriever] = {}
_retriever_lock = threading.Lock()


def get_retriever(name: str) -> Retriever:
    """One retriever per kind for the whole process, so the embedding model loads only once."""
    with _retriever_lock:
        if name not in _retrievers:
            if name != "words" and importlib.util.find_spec("sentence_transformers") is None:
                print(f"cite: {name} retrieval needs `pip install -r requirements-rag.txt`, "
                      "using word matching for now", file=sys.stderr)
                _retrievers[name] = make_retriever("words")
            else:
                _retrievers[name] = make_retriever(name)
        return _retrievers[name]


@dataclass
class Result:
    question: str
    answer: AgentAnswer
    report: CheckReport
    latency_s: float

    @property
    def cost_usd(self) -> float:
        return self.answer.cost_usd + self.report.cost_usd

    def summary(self) -> dict:
        r = self.report
        return {"cited": r.cited, "supported": r.count("supported"), "partial": r.count("partial"),
                "unsupported": r.count("unsupported"), "bad_citation": r.count("bad_citation"),
                "uncited": r.count("uncited"), "support_rate": round(r.support_rate, 4)}

    def to_dict(self) -> dict:
        a = self.answer
        return {
            "question": self.question, "answer": a.text, "stopped": a.stopped,
            "steps": a.steps, "searches": a.searches, "queries": a.queries,
            "sources": [{"n": n, "title": s.title, "url": s.url} for n, s in sorted(a.sources.items())],
            "claims": [asdict(c) for c in self.report.claims],
            "summary": self.summary(), "judge_calls": self.report.judge_calls,
            "cost_usd": round(self.cost_usd, 6), "latency_s": round(self.latency_s, 2),
        }


def run_question(question: str, settings: Settings, client: Any, searcher: Any,
                 on_event: Callable[[dict], None] | None = None,
                 retriever: Retriever | None = None) -> Result:
    emit = on_event or (lambda event: None)
    start = time.perf_counter()
    answer = Agent(client, searcher, settings).ask(question, on_event)
    if answer.stopped == "budget" or not answer.text.strip():
        report = CheckReport()
    else:
        report = check_answer(answer.text, answer.sources, OpenAIJudge(client, settings.model),
                              on_claim=lambda c: emit({"type": "claim", **asdict(c)}),
                              retriever=retriever or get_retriever(settings.retriever))
    result = Result(question, answer, report, time.perf_counter() - start)
    emit({"type": "done", "summary": result.summary(),
          "cost_usd": round(result.cost_usd, 6), "latency_s": round(result.latency_s, 2)})
    return result
