"""The research agent: a plain loop around OpenAI function calling.

The model may call one tool, search_web. We run it, give the results back, and repeat until
the model writes an answer. Every source gets a stable number, and the answer cites them as [n].
Two limits stop it from running away: a step limit and a spending cap per question.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from typing import Any

from .config import Settings, load_settings
from .search import SearchError, SearchResult, Searcher

# Dollars per million tokens (input, output). Check the OpenAI pricing page before trusting these.
PRICES = {"gpt-4o-mini": (0.15, 0.60)}
UNKNOWN_MODEL_PRICE = (2.50, 10.00)   # deliberately high, so an unknown model hits the cap early
SNIPPET_CHARS = 1200                  # how much of each result the model gets to read

SYSTEM_PROMPT = """You are a careful research assistant.
Use the search_web tool to find evidence, then answer the question.
Rules:
1. Only state facts that appear in the search results you were given.
2. After every factual sentence, cite the source number in square brackets, like [1] or [2][3].
3. If the results do not contain enough evidence, say so plainly. Do not guess.
4. Keep the answer short and direct. Do not add a source list; it is added for you."""

TOOLS = [{
    "type": "function",
    "function": {
        "name": "search_web",
        "description": "Search the web. Returns numbered results with a short extract from each page.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "A focused search query."}},
            "required": ["query"],
        },
    },
}]


@dataclass
class AgentAnswer:
    question: str
    text: str
    sources: dict[int, SearchResult] = field(default_factory=dict)
    steps: int = 0                 # model calls made
    searches: int = 0              # tool calls made
    cost_usd: float = 0.0
    stopped: str = "done"          # done | step_limit | budget


def cost_of(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    price_in, price_out = PRICES.get(model, UNKNOWN_MODEL_PRICE)
    return (prompt_tokens * price_in + completion_tokens * price_out) / 1_000_000


class Agent:
    def __init__(self, client: Any, searcher: Any, settings: Settings):
        self.client = client
        self.searcher = searcher
        self.settings = settings

    def _run_tool(self, call: Any, sources: dict[int, SearchResult],
                  numbers: dict[str, int]) -> str:
        try:
            query = str(json.loads(call.function.arguments or "{}")["query"])
        except (ValueError, KeyError, TypeError):
            return "Error: the tool needs JSON arguments like {\"query\": \"...\"}."
        try:
            results = self.searcher.search(query)
        except SearchError as err:
            return f"Search failed: {err}. Try a different query or answer with what you have."
        if not results:
            return "No results."
        lines = []
        for r in results:
            n = numbers.setdefault(r.url, len(numbers) + 1)   # same URL keeps the same number
            sources[n] = r
            lines.append(f"[{n}] {r.title} ({r.url})\n{r.snippet[:SNIPPET_CHARS]}")
        return "\n\n".join(lines)

    def ask(self, question: str) -> AgentAnswer:
        s = self.settings
        out = AgentAnswer(question=question, text="")
        numbers: dict[str, int] = {}
        messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT},
                                {"role": "user", "content": question}]

        for step in range(s.max_steps):
            if out.cost_usd >= s.max_usd_per_question:
                out.stopped = "budget"
                out.text = "Stopped: the spending cap for this question was reached before an answer."
                return out
            last_step = step == s.max_steps - 1
            resp = self.client.chat.completions.create(
                model=s.model, messages=messages, tools=TOOLS,
                tool_choice="none" if last_step else "auto",   # last step must be an answer
            )
            out.steps += 1
            out.cost_usd += cost_of(s.model, resp.usage.prompt_tokens, resp.usage.completion_tokens)
            msg = resp.choices[0].message
            calls = getattr(msg, "tool_calls", None) or []

            if not calls:
                out.text = (msg.content or "").strip()
                out.stopped = "step_limit" if (last_step and out.searches) else "done"
                return out

            messages.append({
                "role": "assistant", "content": msg.content,
                "tool_calls": [{"id": c.id, "type": "function",
                                "function": {"name": c.function.name,
                                             "arguments": c.function.arguments}} for c in calls],
            })
            for c in calls:
                out.searches += 1
                messages.append({"role": "tool", "tool_call_id": c.id,
                                 "content": self._run_tool(c, out.sources, numbers)})

        out.stopped = "step_limit"   # not reached in practice: the last step forces an answer
        return out


def make_client(settings: Settings):
    from openai import OpenAI   # imported here so tests do not need the network client
    return OpenAI(api_key=settings.openai_api_key)


def main() -> int:
    question = " ".join(sys.argv[1:]).strip()
    if not question:
        print('Usage: python -m cite.agent "your question"')
        return 2
    settings = load_settings()
    if not (settings.has_openai and settings.has_tavily):
        print("Both keys are needed. Run: python -m cite.check")
        return 1
    searcher = Searcher(settings)
    answer = Agent(make_client(settings), searcher, settings).ask(question)
    print(f"\n{answer.text}\n")
    for n, r in sorted(answer.sources.items()):
        print(f"[{n}] {r.title}\n    {r.url}")
    print(f"\nModel calls: {answer.steps}, searches: {answer.searches} "
          f"(Tavily credits used: {searcher.credits_used}, cache hits: {searcher.cache_hits})")
    print(f"Estimated cost: ${answer.cost_usd:.4f}   Stopped because: {answer.stopped}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
