"""01_basic_agent.py — template example for a new framework.

Replace every YOUR_* placeholder with real framework API.
This file is injected as Tier-3 LLM context when the converter resolves
patterns that target this framework.
"""

from your_framework import YOUR_AGENT_CLASS, YOUR_BUILDER_CLASS, YOUR_STEP_BASE, YOUR_CONTEXT_TYPE
from your_framework import your_tool_decorator
from your_framework.openai import OpenAIChatClient  # or your provider

# ── Tools ─────────────────────────────────────────────────────────────────────

@your_tool_decorator
def fetch(url: str) -> str:
    """Fetch a resource from the given URL.

    Args:
        url: The URL to fetch.
    """
    import requests
    return requests.get(url).text


@your_tool_decorator
def summarise(text: str) -> str:
    """Return a one-paragraph summary of the text.

    Args:
        text: The text to summarise.
    """
    # In production: call an LLM or summarisation service.
    return text[:500]


# ── Single-agent pattern ───────────────────────────────────────────────────────

agent = YOUR_AGENT_CLASS(
    client=OpenAIChatClient(),          # reads OPENAI_API_KEY from env
    instructions="You are a research assistant. Use your tools to fetch and summarise content.",
    tools=[fetch, summarise],
)

# Synchronous invocation:
result = agent.run("Fetch https://example.com and summarise the content.")
print(result.output)


# ── Multi-step workflow pattern ────────────────────────────────────────────────

class FetchStep(YOUR_STEP_BASE):
    """Fetches the target resource."""

    def run(self, input: dict, ctx: YOUR_CONTEXT_TYPE) -> dict:
        data = fetch(input["url"])
        return {"raw": data}


class SummariseStep(YOUR_STEP_BASE):
    """Summarises the fetched content."""

    def run(self, input: dict, ctx: YOUR_CONTEXT_TYPE) -> dict:
        summary = summarise(input["raw"])
        ctx.set_output({"summary": summary})
        return {}


workflow = (
    YOUR_BUILDER_CLASS()
    .add_step(FetchStep())
    .add_step(SummariseStep())
    .set_entry_point(FetchStep)
    .build()
)

if __name__ == "__main__":
    output = workflow.run({"url": "https://example.com"})
    print(output)
