"""Shared configuration and model setup for all agents."""

from enum import Enum
from pathlib import Path

from pydantic_ai.models import Model
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.providers.google import GoogleProvider
from pydantic_settings import BaseSettings, SettingsConfigDict

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent

# PNNL AI Incubator (Depot) gateway. The gateway speaks each vendor's native wire
# protocol (OpenAI chat completions, Anthropic messages, Google generateContent) at
# this same base URL, so every model is wrapped by its matching native Model/Provider
# pair below rather than being funneled through OpenAIChatModel. Model IDs must match
# the gateway's /v1/models listing exactly.
INCUBATOR_BASE_URL = "https://ai-incubator-api.pnnl.gov"


class WebAgentMode(str, Enum):
    """Supported web app agent modes."""

    PLANNER = "planner"
    PLANNER_EXECUTOR = "planner_executor"
    #: Onboarding mode: instead of planning workflows, the chat agent helps the
    #: user turn their own Python code into new capabilities/subagents. See
    #: agents/onboarding/ and onboarding/.
    ONBOARDING = "onboarding"


class RuntimeConfig(BaseSettings):
    """Runtime configuration loaded from environment and .env file."""

    AI_INCUBATOR_KEY: str
    WEB_AGENT_MODE: WebAgentMode = WebAgentMode.PLANNER

    model_config = SettingsConfigDict(
        env_file=str(_REPO_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )


runtime_config = RuntimeConfig()

_openai_provider = OpenAIProvider(
    api_key=runtime_config.AI_INCUBATOR_KEY,
    base_url=INCUBATOR_BASE_URL,
)

_anthropic_provider = AnthropicProvider(
    api_key=runtime_config.AI_INCUBATOR_KEY,
    base_url=INCUBATOR_BASE_URL,
)

_google_provider = GoogleProvider(
    api_key=runtime_config.AI_INCUBATOR_KEY,
    base_url=INCUBATOR_BASE_URL,
)


def incubator_openai_model(model_id: str) -> OpenAIChatModel:
    """Build an OpenAIChatModel for a PNNL Depot model ID via the shared provider."""
    return OpenAIChatModel(model_id, provider=_openai_provider)


def incubator_anthropic_model(model_id: str) -> AnthropicModel:
    """Build an AnthropicModel for a PNNL Depot model ID via the shared provider."""
    return AnthropicModel(model_id, provider=_anthropic_provider)


def incubator_google_model(model_id: str) -> GoogleModel:
    """Build a GoogleModel for a PNNL Depot model ID via the shared provider."""
    return GoogleModel(model_id, provider=_google_provider)


# Display label -> Depot model ID. Drives the chat-window model dropdown (see app.py).
MODELS: dict[str, Model] = {
    "Claude Sonnet 4.6": incubator_anthropic_model("claude-sonnet-4-6-project"),
    "Claude Opus 4.8": incubator_anthropic_model("claude-opus-4-8-project"),
    "Claude Haiku 4.5": incubator_anthropic_model("claude-haiku-4-5-20251001-v1-project"),
    "GPT 5.5": incubator_openai_model("gpt-5.5-project"),
    "GPT 5.4": incubator_openai_model("gpt-5.4-project"),
    "GPT 5.1": incubator_openai_model("gpt-5.1-project"),
    "GPT o4 Mini": incubator_openai_model("o4-mini-project"),
    "Gemma 4 26B": incubator_google_model("gemma-4-26b-a4b-project"),
    "Gemini 3.5 Flash": incubator_google_model("gemini-3.5-flash-project"),
    "Claude Sonnet 5": incubator_anthropic_model("claude-sonnet-5-project"),
    "GPT 5.6 Terra": incubator_openai_model("gpt-5.6-terra-project"),
    "Gemini 3.7 Flash": incubator_google_model("gemini-3.7-flash-project"),
}

# Default model used by the planner and every subagent.
model = MODELS["Claude Sonnet 4.6"]


def load_prompt(extra_instructions: str = "") -> str:
    """Load system prompt and tool catalog, optionally with extra instructions."""
    base_prompt = (_HERE.parent / "system_prompt.md").read_text(encoding="utf-8")
    tool_catalog = (_REPO_ROOT / "tools" / "tool_catalog.yaml").read_text(encoding="utf-8")
    return (
        f"{base_prompt}\n\n"
        f"## Tool Catalog\n\n"
        f"```yaml\n{tool_catalog}\n```\n\n"
        f"{extra_instructions}".strip()
    )
