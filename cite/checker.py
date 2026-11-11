"""Citation checker: does each cited source really say what the answer claims?

For every sentence that cites [n], we pull the most relevant passages from source n and ask a
judge model whether they support the sentence. The judge must copy an exact quote. We then check
that the quote really appears in the page. A "supported" verdict with a made up quote is thrown out,
so the judge cannot talk its way past the check.
"""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from .pricing import cost_of
from .search import SearchResult

CITE_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
MIN_QUOTE_CHARS = 6
STOPWORDS = {"that", "this", "with", "from", "have", "been", "which", "their", "there", "these",
             "those", "they", "them", "then", "than", "into", "also", "such", "when", "what",
             "where", "will", "would", "could", "should", "about", "other", "more", "most",
             "some", "only", "very", "over", "your", "like", "used", "using", "can", "and",
             "the", "for", "are", "was", "were", "its", "not", "but", "you", "any", "all"}


@dataclass
class Part:
    part: str               # one separate fact inside the sentence
    quote: str = ""         # words the judge copied from the page for it


@dataclass
class JudgeResult:
    parts: list[Part] = field(default_factory=list)
    cost_usd: float = 0.0


class Judge(Protocol):
    def judge(self, claim: str, passages: list[str]) -> JudgeResult: ...


@dataclass
class ClaimCheck:
    sentence: str
    cites: list[int]
    status: str             # supported | partial | unsupported | bad_citation | uncited
    source: int | None = None
    quotes: list[str] = field(default_factory=list)
    note: str = ""


@dataclass
class CheckReport:
    claims: list[ClaimCheck] = field(default_factory=list)
    judge_calls: int = 0
    cost_usd: float = 0.0

    def count(self, status: str) -> int:
        return sum(1 for c in self.claims if c.status == status)

    @property
    def cited(self) -> int:
        return sum(1 for c in self.claims if c.status != "uncited")

    @property
    def support_rate(self) -> float:
        return self.count("supported") / self.cited if self.cited else 0.0


# ---------- splitting the answer into claims ----------

ABBREVIATIONS = {"dr", "mr", "mrs", "ms", "prof", "st", "vs", "etc", "inc", "ltd", "jr", "sr", "no",
                 "fig", "approx", "e.g", "i.e", "u.s", "u.k"}
BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def split_sentences(line: str) -> list[str]:
    """Split a line into sentences, but not after abbreviations like Dr. or initials like J."""
    out, start = [], 0
    for m in BOUNDARY.finditer(line):
        words = line[start:m.start()].split()
        last = words[-1].lower().rstrip(".") if words else ""
        if last in ABBREVIATIONS or re.fullmatch(r"[a-z]", last):
            continue
        out.append(line[start:m.start()])
        start = m.end()
    out.append(line[start:])
    return out


def split_claims(answer: str) -> list[tuple[str, list[int]]]:
    """Return (sentence, cited source numbers) for each sentence worth checking."""
    out = []
    for line in answer.splitlines():
        line = re.sub(r"^\s*(?:[-*\u2022]|\d+[.)])\s+", "", line).strip()
        if not line or line.endswith(":"):
            continue
        line = re.sub(r"([.!?])\s*((?:\[\d+\])+)", r"\2\1", line)   # "text. [1]" -> "text[1]."
        for sentence in split_sentences(line):
            plain = CITE_RE.sub("", sentence).strip()
            if len(plain.split()) < 5:
                continue
            cites = [int(n) for m in CITE_RE.findall(sentence) for n in re.split(r"\s*,\s*", m)]
            out.append((sentence.strip(), cites))
    return out


# ---------- finding the right passages ----------

def split_chunks(text: str, size: int = 500, overlap: int = 100) -> list[str]:
    chunks, buf = [], ""
    for piece in re.split(r"\n+", text):
        piece = piece.strip()
        if not piece:
            continue
        if buf and len(buf) + len(piece) + 1 > size:
            chunks.append(buf)
            buf = ""
        buf = f"{buf} {piece}".strip()
        while len(buf) > size:
            chunks.append(buf[:size])
            buf = buf[size - overlap:]
    if buf:
        chunks.append(buf)
    return chunks


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) >= 4 and w not in STOPWORDS}


def best_passages(claim: str, page_text: str, top: int = 8) -> list[str]:
    claim_words = _words(CITE_RE.sub("", claim))
    if not claim_words:
        return []
    need = min(2, len(claim_words))
    scored = [(len(claim_words & _words(c)), c) for c in split_chunks(page_text)]
    scored = [s for s in scored if s[0] >= need]
    scored.sort(key=lambda s: -s[0])
    return [c for _, c in scored[:top]]


def normalize(text: str) -> str:
    """Lowercase and drop punctuation, so a comma or a curly quote cannot break a quote match."""
    return re.sub(r"[\W_]+", " ", text.lower()).strip()


def _meaningful(q: str) -> bool:
    """A quote must carry something: a number or a real word. This rejects filler like "it is" but
    accepts short evidence like "self healing", "in 1989" or "Joseph Okafor"."""
    return len(q) >= MIN_QUOTE_CHARS and any(t.isdigit() or (len(t) >= 3 and t not in STOPWORDS)
                                             for t in q.split())


def quote_in_page(quote: str, page_text: str) -> bool:
    q = normalize(quote)
    return _meaningful(q) and f" {q} " in f" {normalize(page_text)} "


# ---------- details the quote must not contradict ----------

NUMBER_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")
NAME_RE = re.compile(r"\b[A-Z][A-Za-z0-9]{2,}\b")
# Only plain negations count. "cannot" and "without" are often fair paraphrases, so they are left out.
NEGATIONS = {"not", "no", "never", "none"}
ALIASES = ((re.compile(r"\bU\.?S\.?A?\b"), "United States"), (re.compile(r"\bU\.?K\.?\b"), "United Kingdom"))


def _expand(text: str) -> str:
    for pattern, full in ALIASES:
        text = pattern.sub(full, text)
    return text


def _numbers(text: str) -> set[str]:
    return {n.replace(",", "").rstrip(".") for n in NUMBER_RE.findall(text)}


def _has_negation(text: str) -> bool:
    return any(w in NEGATIONS or w.endswith("n't") for w in re.findall(r"[a-z']+", text.lower()))


def detail_problems(sentence: str, quotes: list[str], page_text: str) -> list[str]:
    """A real quote can still say something else. Check the parts a model is most likely to get wrong:
    numbers must appear in the quotes, names must appear in the page, and a claim that says "not"
    needs a quote that says "not" too."""
    plain = _expand(CITE_RE.sub("", sentence))
    page_text = _expand(page_text)
    joined = " ".join(quotes)
    problems = []
    missing = sorted(_numbers(plain) - _numbers(joined))
    if missing:
        problems.append("number not in the quote: " + ", ".join(missing))
    page = f" {normalize(page_text)} "
    names = [name for word in plain.split()[1:] for name in NAME_RE.findall(word)]   # skip the first word
    absent = sorted({n for n in names
                     if f" {n.lower()} " not in page and f" {n.lower().rstrip('s')} " not in page})
    if absent:
        problems.append("name not in the source: " + ", ".join(absent))
    if _has_negation(plain) and not _has_negation(joined):
        problems.append("the sentence says 'not' but the quote does not")
    return problems


# ---------- the judge ----------

JUDGE_PROMPT = """You check whether a claim is supported by passages from a web page.
Step 1: split the claim into its separate factual parts. Ignore filler words and style.
Step 2: for each part, copy an exact quote from the passages that states that part.
The quote must contain the same numbers and names as the part and must say the same thing, not
something different or the opposite. If the passages give a different number or name, or say something
different, leave that quote empty. If no passage states the part, leave its quote empty.
A passage that is only related to the topic does not count. Never paraphrase inside a quote.
Reply with JSON only: {"parts": [{"part": "short description", "quote": "exact words or empty"}]}"""


class OpenAIJudge:
    def __init__(self, client: Any, model: str):
        self.client, self.model = client, model

    def judge(self, claim: str, passages: list[str]) -> JudgeResult:
        body = "\n\n".join(f"Passage {i}:\n{p}" for i, p in enumerate(passages, 1))
        resp = self.client.chat.completions.create(
            model=self.model, temperature=0, response_format={"type": "json_object"},
            messages=[{"role": "system", "content": JUDGE_PROMPT},
                      {"role": "user", "content": f"Claim: {CITE_RE.sub('', claim).strip()}\n\n{body}"}],
        )
        cost = cost_of(self.model, resp.usage.prompt_tokens, resp.usage.completion_tokens)
        try:
            data = json.loads(resp.choices[0].message.content or "{}")
            parts = [Part(str(p.get("part", "")), str(p.get("quote", "") or ""))
                     for p in data.get("parts", []) if isinstance(p, dict)]
        except (ValueError, TypeError, AttributeError):
            return JudgeResult([], cost)
        return JudgeResult(parts, cost)


# ---------- putting it together ----------

def _check_one(sentence: str, cites: list[int], sources: dict[int, SearchResult],
               judge: Judge) -> tuple[ClaimCheck, int, float]:
    """Check one sentence. Returns the verdict, the number of judge calls and their cost."""
    if not cites:
        return ClaimCheck(sentence, [], "uncited", note="no citation"), 0, 0.0
    missing = [n for n in cites if n not in sources]
    if missing and len(missing) == len(cites):
        return ClaimCheck(sentence, cites, "bad_citation", note=f"source {missing[0]} does not exist"), 0, 0.0

    best = ClaimCheck(sentence, cites, "unsupported", note="no passage matched the claim")
    best_found, calls, cost = 0, 0, 0.0
    for n in (c for c in cites if c in sources):
        page = sources[n].page_text
        passages = best_passages(sentence, page)
        if not passages:
            continue
        result = judge.judge(sentence, passages)
        calls += 1
        cost += result.cost_usd
        if not result.parts:
            if best_found == 0:
                best = ClaimCheck(sentence, cites, "unsupported", n, note="the cited page does not state this")
            continue
        good = [p for p in result.parts if quote_in_page(p.quote, page)]
        if len(good) == len(result.parts):
            quotes = [p.quote for p in good]
            problems = detail_problems(sentence, quotes, page)
            if not problems:
                return ClaimCheck(sentence, cites, "supported", n, quotes), calls, cost
            if best_found < len(good):
                best_found = len(good)
                best = ClaimCheck(sentence, cites, "partial", n, quotes,
                                  "quote is real but does not match: " + "; ".join(problems))
            continue
        if len(good) > best_found:
            best_found = len(good)
            missing_parts = "; ".join(p.part for p in result.parts if p not in good)
            best = ClaimCheck(sentence, cites, "partial", n, [p.quote for p in good],
                              f"{len(good)} of {len(result.parts)} facts found. Not found: {missing_parts}")
        elif best_found == 0:
            best = ClaimCheck(sentence, cites, "unsupported", n,
                              note="no quote the judge gave is in the cited page")
    return best, calls, cost


def check_answer(answer: str, sources: dict[int, SearchResult], judge: Judge,
                 on_claim: Callable[[ClaimCheck], None] | None = None, workers: int = 4) -> CheckReport:
    """A sentence is supported only if every fact in it has a quote that really is in one cited page,
    and the quote agrees with the sentence on numbers, names and 'not'. Sentences are checked in
    parallel; results come back in the order of the answer."""
    report = CheckReport()
    jobs = split_claims(answer)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for claim, calls, cost in pool.map(lambda j: _check_one(j[0], j[1], sources, judge), jobs):
            report.claims.append(claim)
            report.judge_calls += calls
            report.cost_usd += cost
            if on_claim:
                on_claim(claim)
    return report
