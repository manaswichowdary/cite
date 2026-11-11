"""What a model call costs. One place, so the agent and the checker agree."""

# Dollars per million tokens (input, output). Compare with OpenAI's pricing page before trusting these.
PRICES = {"gpt-4o-mini": (0.15, 0.60)}
UNKNOWN_MODEL_PRICE = (2.50, 10.00)   # deliberately high, so an unknown model hits the cap early


def cost_of(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    price_in, price_out = PRICES.get(model, UNKNOWN_MODEL_PRICE)
    return (prompt_tokens * price_in + completion_tokens * price_out) / 1_000_000
