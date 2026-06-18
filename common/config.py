"""Shared configuration and model setup for all agents."""

from pathlib import Path

from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_settings import BaseSettings, SettingsConfigDict

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent


class RuntimeConfig(BaseSettings):
    """Runtime configuration loaded from environment and .env file."""

    AI_INCUBATOR_KEY: str

    model_config = SettingsConfigDict(
        env_file=str(_REPO_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )


runtime_config = RuntimeConfig()

model = AnthropicModel(
    "claude-haiku-4-5-20251001-v1-project",
    provider=AnthropicProvider(
        api_key=runtime_config.AI_INCUBATOR_KEY,
        base_url="https://ai-incubator-api.pnnl.gov",
    ),
)


def load_prompt(extra_instructions: str = "") -> str:
    """Load system prompt and tool catalog, optionally with extra instructions."""
    base_prompt = (_HERE.parent / "system_prompt.md").read_text(encoding="utf-8")
    tool_catalog = (_REPO_ROOT / "tool_catalog_generated.yaml").read_text(encoding="utf-8")
    return (
        f"{base_prompt}\n\n"
        f"## Tool Catalog\n\n"
        f"```yaml\n{tool_catalog}\n```\n\n"
        f"{extra_instructions}".strip()
    )
