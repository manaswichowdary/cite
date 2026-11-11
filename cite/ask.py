"""Ask one question from the terminal: python -m cite.ask "your question"."""

from __future__ import annotations

import sys

from .agent import make_client
from .config import load_settings
from .pipeline import Result, run_question
from .search import Searcher
from .trace import write_trace

LABELS = {"supported": "OK", "partial": "PARTIAL", "unsupported": "NOT SUPPORTED",
          "bad_citation": "BAD CITATION", "uncited": "UNCITED"}


def print_result(result: Result) -> None:
    a, r = result.answer, result.report
    print(f"\n{a.text}\n")
    for n, s in sorted(a.sources.items()):
        print(f"[{n}] {s.title}\n    {s.url}")
    print(f"\nCitation check: {r.count('supported')} of {r.cited} cited claims supported "
          f"({r.support_rate:.0%}), {r.count('partial')} partial, {r.count('unsupported')} unsupported, "
          f"{r.count('bad_citation')} bad citations, {r.count('uncited')} sentences with no citation")
    for c in r.claims:
        print(f"  [{LABELS[c.status]}] {c.sentence}")
        if c.note:
            print(f"      {c.note}")
        if c.status == "supported":
            for q in c.quotes:
                print(f'      source {c.source} says: "{q}"')


def main() -> int:
    question = " ".join(sys.argv[1:]).strip()
    if not question:
        print('Usage: python -m cite.ask "your question"')
        return 2
    settings = load_settings()
    if not (settings.has_openai and settings.has_tavily):
        print("Both keys are needed. Run: python -m cite.check")
        return 1
    searcher = Searcher(settings)
    result = run_question(question, settings, make_client(settings), searcher)
    print_result(result)
    write_trace(result)
    print(f"\nModel calls: {result.answer.steps}, searches: {result.answer.searches}, "
          f"judge calls: {result.report.judge_calls} (Tavily credits used: {searcher.credits_used}, "
          f"cache hits: {searcher.cache_hits})")
    print(f"Estimated cost: ${result.cost_usd:.4f}   Time: {result.latency_s:.1f}s   "
          f"Stopped because: {result.answer.stopped}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
