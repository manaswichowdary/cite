"""Two evals.

planted:   claims with known errors are checked against passages with fictional facts. Measures how
           many wrong claims the checker catches, and how many true claims it wrongly rejects.
questions: the whole agent answers 30 questions. Measures how many cited claims the checker can verify.

Usage:
    python -m cite.evals planted
    python -m cite.evals questions --limit 5 --max-usd 0.50
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Callable

from .checker import Judge, OpenAIJudge, check_answer
from .config import Settings, load_settings
from .pipeline import run_question
from .search import SearchResult

KINDS = ("wrong_number", "wrong_name", "extra_fact", "negation")


# ---------- planted errors ----------

def _as_claim(text: str) -> str:
    return text.rstrip(". ") + " [1]."     # the checker expects a citation


def score_planted(cases: list[dict], judge: Judge) -> dict:
    by_kind = {k: {"total": 0, "caught": 0} for k in KINDS}
    true_total = true_accepted = judge_calls = 0
    cost = 0.0
    misses, false_alarms = [], []
    for case in cases:
        page = SearchResult("Planted passage", f"https://example.test/{case['id']}",
                            case["passage"], case["passage"], 1.0)
        for kind in ("true",) + KINDS:
            report = check_answer(_as_claim(case[kind]), {1: page}, judge)
            cost += report.cost_usd
            judge_calls += report.judge_calls
            if not report.claims:
                continue
            claim = report.claims[0]
            if kind == "true":
                true_total += 1
                if claim.status == "supported":
                    true_accepted += 1
                else:
                    false_alarms.append({"id": case["id"], "claim": case[kind], "status": claim.status,
                                         "note": claim.note})
            else:
                by_kind[kind]["total"] += 1
                if claim.status != "supported":
                    by_kind[kind]["caught"] += 1
                else:
                    misses.append({"id": case["id"], "kind": kind, "claim": case[kind],
                                   "quotes": claim.quotes})
    planted_total = sum(v["total"] for v in by_kind.values())
    caught = sum(v["caught"] for v in by_kind.values())
    rate = lambda a, b: round(a / b, 4) if b else 0.0    # noqa: E731
    return {
        "passages": len(cases), "planted_claims": planted_total, "planted_caught": caught,
        "catch_rate": rate(caught, planted_total),
        "catch_rate_by_kind": {k: {**v, "rate": rate(v["caught"], v["total"])} for k, v in by_kind.items()},
        "true_claims": true_total, "true_accepted": true_accepted,
        "true_acceptance_rate": rate(true_accepted, true_total),
        "misses": misses, "false_alarms": false_alarms,
        "judge_calls": judge_calls, "cost_usd": round(cost, 4),
    }


# ---------- the 30 question run ----------

def percentile(values: list[float], p: float) -> float:
    """Nearest rank percentile: the smallest value that at least p percent of the values are at or below."""
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[max(1, math.ceil(p / 100 * len(ordered))) - 1]


def summarize(results: list[dict]) -> dict:
    ok = [r for r in results if "error" not in r]
    cited = sum(r["summary"]["cited"] for r in ok)
    supported = sum(r["summary"]["supported"] for r in ok)
    claims = sum(len(r["claims"]) for r in ok)
    stopped: dict[str, int] = {}
    for r in ok:
        stopped[r["stopped"]] = stopped.get(r["stopped"], 0) + 1
    clean = sum(1 for r in ok if r["claims"] and all(c["status"] == "supported" for c in r["claims"]))
    n = len(ok) or 1
    return {
        "questions": len(results), "answered": len(ok), "errors": len(results) - len(ok),
        "claims": claims, "cited_claims": cited, "supported_claims": supported,
        "support_rate": round(supported / cited, 4) if cited else 0.0,
        "uncited_sentences": sum(r["summary"]["uncited"] for r in ok),
        "answers_with_every_claim_supported": clean,
        "avg_searches": round(sum(r["searches"] for r in ok) / n, 2),
        "latency_p50_s": round(percentile([r["latency_s"] for r in ok], 50), 2),
        "latency_p95_s": round(percentile([r["latency_s"] for r in ok], 95), 2),
        "avg_cost_usd": round(sum(r["cost_usd"] for r in ok) / n, 5),
        "total_cost_usd": round(sum(r["cost_usd"] for r in ok), 4),
        "stopped": stopped,
    }


def read_results(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def run_questions(questions: list[dict], settings: Settings, client, searcher, out_dir: str | Path,
                  limit: int | None = None, max_usd: float = 1.0,
                  run: Callable = run_question, log: Callable[[str], None] = print) -> dict:
    """Answer each question once. Already answered questions are skipped, so a crash costs nothing."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "questions.jsonl"
    done = {r["id"] for r in read_results(path) if "error" not in r}
    spent, ran, failures = 0.0, 0, 0
    for q in questions:
        if q["id"] in done:
            continue
        if (limit is not None and ran >= limit) or spent >= max_usd:
            log(f"Stopping: ran {ran} questions, spent ${spent:.4f}")
            break
        try:
            result = run(q["question"], settings, client, searcher)
            row = {"id": q["id"], **result.to_dict()}
            spent += result.cost_usd
            failures = 0
            s = row["summary"]
            log(f"[{q['id']:>2}] {s['supported']}/{s['cited']} supported, {s['uncited']} uncited, "
                f"${row['cost_usd']:.4f}, {row['latency_s']}s  {q['question']}")
        except Exception as err:
            row = {"id": q["id"], "question": q["question"], "error": type(err).__name__}
            failures += 1
            log(f"[{q['id']:>2}] failed: {type(err).__name__}")
        with path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        ran += 1
        if failures >= 3:
            log("Stopping: 3 failures in a row. Check your keys and connection.")
            break
    latest: dict[int, dict] = {}
    for r in read_results(path):
        if r["id"] not in latest or "error" not in r:
            latest[r["id"]] = r
    metrics = summarize(list(latest.values()))
    (out / "questions_metrics.json").write_text(json.dumps(metrics, indent=1))
    return metrics


# ---------- README results ----------

RESULTS_START, RESULTS_END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
KIND_NAMES = {"wrong_number": "Wrong number", "wrong_name": "Wrong name",
              "extra_fact": "Extra fact the source does not state", "negation": "Negation"}


def _pct(a: int, b: int) -> str:
    return f"{a} of {b} ({a / b:.0%})" if b else "n/a"


def build_results_block(planted: dict, metrics: dict, holdout: dict | None = None) -> str:
    rows = [f"| {KIND_NAMES[k]} | {_pct(v['caught'], v['total'])} |"
            for k, v in planted["catch_rate_by_kind"].items()]
    return "\n".join([
        "**Citation checker on planted errors** (20 passages, 80 wrong claims, 20 true claims)",
        "",
        "| Kind of error | Caught |", "|---|---|", *rows,
        f"| **All planted errors** | **{_pct(planted['planted_caught'], planted['planted_claims'])}** |",
        f"| True claims accepted | {_pct(planted['true_accepted'], planted['true_claims'])} |",
        "",
        f"**Agent on {metrics['questions']} questions**",
        "",
        "| Measure | Result |", "|---|---|",
        f"| Cited sentences fully supported | {_pct(metrics['supported_claims'], metrics['cited_claims'])} |",
        f"| Answers where every sentence was supported | {metrics['answers_with_every_claim_supported']} of {metrics['answered']} |",
        f"| Sentences with no citation | {metrics['uncited_sentences']} |",
        f"| Searches per question | {metrics['avg_searches']} |",
        f"| Time per question (median, slowest 5%) | {metrics['latency_p50_s']} s, {metrics['latency_p95_s']} s |",
        f"| Cost per question | ${metrics['avg_cost_usd']:.5f} |",
        *(_holdout_rows(holdout) if holdout else []),
    ])


def _holdout_rows(h: dict) -> list[str]:
    return [
        "",
        f"**Agent on {h['questions']} fresh questions** (run once after the last change, never tuned on)",
        "",
        "| Measure | Result |", "|---|---|",
        f"| Cited sentences fully supported | {_pct(h['supported_claims'], h['cited_claims'])} |",
        f"| Answers where every sentence was supported | {h['answers_with_every_claim_supported']} of {h['answered']} |",
        f"| Sentences with no citation | {h['uncited_sentences']} |",
        f"| Time per question (median, slowest 5%) | {h['latency_p50_s']} s, {h['latency_p95_s']} s |",
        f"| Cost per question | ${h['avg_cost_usd']:.5f} |",
    ]


def update_readme(readme: Path, planted_path: Path, metrics_path: Path,
                  holdout_path: Path | None = None) -> None:
    text = readme.read_text()
    start, end = text.index(RESULTS_START), text.index(RESULTS_END)
    holdout = json.loads(holdout_path.read_text()) if holdout_path and holdout_path.exists() else None
    block = build_results_block(json.loads(planted_path.read_text()), json.loads(metrics_path.read_text()),
                                holdout)
    readme.write_text(text[:start + len(RESULTS_START)] + "\n" + block + "\n" + text[end:])


# ---------- command line ----------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m cite.evals")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("planted", help="measure the citation checker on planted errors")
    p1.add_argument("--cases", default="evals/planted_cases.json")
    p1.add_argument("--out", default="evals/results")
    p2 = sub.add_parser("questions", help="run the agent on the 30 question set")
    p2.add_argument("--questions", default="evals/questions.json")
    p2.add_argument("--out", default="evals/results")
    p2.add_argument("--limit", type=int, default=None)
    p2.add_argument("--max-usd", type=float, default=1.0)
    p4 = sub.add_parser("readme", help="write the measured numbers into the README Results block")
    p4.add_argument("--out", default="evals/results")
    p4.add_argument("--readme", default="README.md")
    p4.add_argument("--holdout", default="evals/results_holdout")
    p5 = sub.add_parser("audit", help="print flagged sentences with their source links, for a hand check")
    p5.add_argument("--out", default="evals/results")
    p5.add_argument("--limit", type=int, default=10)
    p3 = sub.add_parser("show", help="print the claims that were not fully supported")
    p3.add_argument("--out", default="evals/results")
    p3.add_argument("--limit", type=int, default=40)
    args = ap.parse_args(argv)

    if args.cmd == "readme":
        out = Path(args.out)
        update_readme(Path(args.readme), out / "planted.json", out / "questions_metrics.json",
                      Path(args.holdout) / "questions_metrics.json")
        print(f"Updated {args.readme}")
        return 0

    if args.cmd == "audit":
        shown = 0
        for r in read_results(Path(args.out) / "questions.jsonl"):
            urls = {s["n"]: s["url"] for s in r.get("sources", [])}
            for c in r.get("claims", []):
                if c["status"] == "supported":
                    continue
                shown += 1
                print(f"\n{shown}. {c['sentence']}\n   Checker said: {c['status']}, {c['note']}")
                for n in c["cites"]:
                    print(f"   [{n}] {urls.get(n, 'missing')}")
                if shown >= args.limit:
                    return 0
        return 0

    if args.cmd == "show":
        shown = 0
        for r in read_results(Path(args.out) / "questions.jsonl"):
            for c in r.get("claims", []):
                if c["status"] in ("supported",):
                    continue
                print(f"[{r['id']}] {c['status'].upper()}: {c['sentence']}\n      {c['note']}")
                shown += 1
                if shown >= args.limit:
                    return 0
        return 0

    settings = load_settings()
    from .agent import make_client
    if not settings.has_openai or (args.cmd == "questions" and not settings.has_tavily):
        print("Missing keys. Run: python -m cite.check")
        return 1
    client = make_client(settings)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.cmd == "planted":
        report = score_planted(json.loads(Path(args.cases).read_text()), OpenAIJudge(client, settings.model))
        (out / "planted.json").write_text(json.dumps(report, indent=1))
        print(f"Planted wrong claims caught: {report['planted_caught']} of {report['planted_claims']} "
              f"({report['catch_rate']:.0%})")
        for kind, v in report["catch_rate_by_kind"].items():
            print(f"  {kind:<13} {v['caught']:>2} of {v['total']:>2} ({v['rate']:.0%})")
        print(f"True claims accepted: {report['true_accepted']} of {report['true_claims']} "
              f"({report['true_acceptance_rate']:.0%})")
        print(f"Misses: {len(report['misses'])}   False alarms: {len(report['false_alarms'])}   "
              f"Judge calls: {report['judge_calls']}   Estimated cost: ${report['cost_usd']:.4f}")
        for m in report["misses"]:
            print(f"  MISSED {m['id']} {m['kind']}: {m['claim']}\n         quotes: {m['quotes']}")
        for f in report["false_alarms"]:
            print(f"  FALSE ALARM {f['id']} ({f['status']}): {f['claim']}\n         {f['note']}")
        print(f"Details saved to {out / 'planted.json'}")
        return 0

    from .search import Searcher
    searcher = Searcher(settings)
    metrics = run_questions(json.loads(Path(args.questions).read_text()), settings, client, searcher,
                            out, args.limit, args.max_usd)
    print("\n" + json.dumps(metrics, indent=1))
    print(f"Tavily credits used this run: {searcher.credits_used}, cache hits: {searcher.cache_hits}")
    print(f"Details saved to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
