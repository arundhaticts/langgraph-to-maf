"""
AWS Strands Example 03 — Model providers (strands.models).

NATIVE PATTERN (Strands):
    The model is INJECTED into the Agent. Swap providers by swapping the
    constructor — the rest of the agent code is unchanged. The DEFAULT provider
    is Amazon Bedrock: a bare Agent() with no model= uses a Bedrock Claude model
    and requires AWS credentials.

IR MAPPING:
    The model constructor maps to the node's model/config (provider, model_id,
    temperature, ...) — it is NOT a node of its own.

Each builder below returns an agent with a different provider. Only install the
extra a given provider needs (see the pip hints inline).
"""
from __future__ import annotations

from strands import Agent

SYSTEM_PROMPT = "You are a concise test-optimisation assistant."


# ---------------------------------------------------------------------------
# 1) Amazon Bedrock — the DEFAULT provider.
# ---------------------------------------------------------------------------
def bedrock_agent() -> Agent:
    from strands.models import BedrockModel

    model = BedrockModel(
        model_id="us.anthropic.claude-sonnet-4-20250514-v1:0",
        region_name="us-east-1",
        temperature=0.3,
    )
    return Agent(model=model, system_prompt=SYSTEM_PROMPT)


# ---------------------------------------------------------------------------
# 2) Anthropic API directly.  pip install 'strands-agents[anthropic]'
#    Needs ANTHROPIC_API_KEY.
# ---------------------------------------------------------------------------
def anthropic_agent() -> Agent:
    from strands.models.anthropic import AnthropicModel

    model = AnthropicModel(model_id="claude-sonnet-4-20250514", max_tokens=1024)
    return Agent(model=model, system_prompt=SYSTEM_PROMPT)


# ---------------------------------------------------------------------------
# 3) OpenAI.  pip install 'strands-agents[openai]'   Needs OPENAI_API_KEY.
# ---------------------------------------------------------------------------
def openai_agent() -> Agent:
    from strands.models.openai import OpenAIModel

    model = OpenAIModel(model_id="gpt-4o")
    return Agent(model=model, system_prompt=SYSTEM_PROMPT)


# ---------------------------------------------------------------------------
# 4) LiteLLM — one interface to 100+ providers.
#    pip install 'strands-agents[litellm]'
# ---------------------------------------------------------------------------
def litellm_agent() -> Agent:
    from strands.models.litellm import LiteLLMModel

    model = LiteLLMModel(model_id="gemini/gemini-1.5-pro")
    return Agent(model=model, system_prompt=SYSTEM_PROMPT)


# ---------------------------------------------------------------------------
# 5) Ollama — local inference.  pip install 'strands-agents[ollama]'
# ---------------------------------------------------------------------------
def ollama_agent() -> Agent:
    from strands.models.ollama import OllamaModel

    model = OllamaModel(host="http://localhost:11434", model_id="llama3")
    return Agent(model=model, system_prompt=SYSTEM_PROMPT)


def main() -> None:
    # Provider is fully swappable; the invocation is identical.
    agent = bedrock_agent()
    print(agent("Name one quick win for a slow test suite.").message)


if __name__ == "__main__":
    main()
