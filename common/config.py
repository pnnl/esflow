"""Shared configuration and model setup for all agents."""

from pathlib import Path

from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_settings import BaseSettings, SettingsConfigDict

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent

# PNNL AI Incubator (Depot) gateway. Every Depot model is OpenAI-compatible, so all
# models — including the Claude ones — are reached through OpenAIChatModel, not
# AnthropicModel. Model IDs must match the gateway's /v1/models listing exactly.
INCUBATOR_BASE_URL = "https://ai-incubator-api.pnnl.gov"


class RuntimeConfig(BaseSettings):
    """Runtime configuration loaded from environment and .env file."""

    AI_INCUBATOR_KEY: str

    model_config = SettingsConfigDict(
        env_file=str(_REPO_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )


runtime_config = RuntimeConfig()

_provider = OpenAIProvider(
    api_key=runtime_config.AI_INCUBATOR_KEY,
    base_url=INCUBATOR_BASE_URL,
)


def incubator_model(model_id: str) -> OpenAIChatModel:
    """Build an OpenAIChatModel for a PNNL Depot model ID via the shared provider."""
    return OpenAIChatModel(model_id, provider=_provider)


# Display label -> Depot model ID. Drives the chat-window model dropdown (see app.py).
MODELS: dict[str, OpenAIChatModel] = {
    "Claude Sonnet 4.6": incubator_model("claude-sonnet-4-6-project"),
    "Claude Opus 4.8": incubator_model("claude-opus-4-8-project"),
    "Claude Haiku 4.5": incubator_model("claude-haiku-4-5-20251001-v1-project"),
    "GPT-5.5": incubator_model("gpt-5.5-project"),
    "GPT-5.1": incubator_model("gpt-5.1-project"),
    "o4-mini": incubator_model("o4-mini-project"),
    "Gemma 4 26B": incubator_model("gemma-4-26b-a4b-project"),
}

# Default model used by the supervisor and every subagent.
model = MODELS["Claude Sonnet 4.6"]


def load_prompt(extra_instructions: str = "") -> str:
    """Load system prompt and tool catalog, optionally with extra instructions."""
    base_prompt = (_HERE.parent / "system_prompt.md").read_text(encoding="utf-8")
    tool_catalog = (_REPO_ROOT / "tool_catalog.yaml").read_text(encoding="utf-8")
    return (
        f"{base_prompt}\n\n"
        f"## Tool Catalog\n\n"
        f"```yaml\n{tool_catalog}\n```\n\n"
        f"{extra_instructions}".strip()
    )
