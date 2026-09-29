"""Shared configuration and model setup for all agents."""

from enum import Enum
from pathlib import Path
from typing import Optional

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
    """Runtime configuration loaded from environment and .env file.

    All ESFlow path settings can be overridden via environment variables so that
    no file paths need to be hard-coded in source.  Sensible repo-relative
    defaults are applied when an env var is absent.
    """

    # --- Auth / gateway ---
    AI_INCUBATOR_KEY: str
    WEB_AGENT_MODE: WebAgentMode = WebAgentMode.PLANNER

    # --- Data & output paths ---
    # Root directory containing E3SM output and observation data.
    ESFLOW_DATA_DIR: Optional[str] = None
    # Directory where workflow outputs (plots, CSVs, etc.) are written.
    ESFLOW_OUTPUT_DIR: Optional[str] = None
    # Path to the tool catalog YAML.  Defaults to tools/tool_catalog.yaml inside the repo.
    ESFLOW_CATALOG_FILE: Optional[str] = None
    # Validation cache directory.  Defaults to <output_dir>/.validation_cache.
    ESFLOW_VALIDATION_CACHE_DIR: Optional[str] = None

    # --- E3SM simulation identifiers (optional defaults; the agent may still ask) ---
    ESFLOW_CASE_NAME: Optional[str] = None
    # Comma-separated list of years to validate, e.g. "1985,1986,1987"
    ESFLOW_YEARS: Optional[str] = None

    # --- Observation sub-directory layout (override if your data is not standard) ---
    ESFLOW_OBS_SUBDIR: str = "obs"
    ESFLOW_GAUGE_METADATA_FILENAME: str = "gauge_metadata.csv"
    ESFLOW_STREAMFLOW_SUBDIR: str = "streamflow"
    ESFLOW_BASIN_POLYGONS_FILENAME: str = "basin_polygons.geojson"
    ESFLOW_ILAMB_CACHE_SUBDIR: str = "ilamb_cache"

    model_config = SettingsConfigDict(
        env_file=str(_REPO_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ------------------------------------------------------------------ helpers

    def resolved_data_dir(self) -> Path:
        """Return the data directory as an absolute Path (default: <repo>/data)."""
        return Path(self.ESFLOW_DATA_DIR or (_REPO_ROOT / "data")).resolve()

    def resolved_output_dir(self) -> Path:
        """Return the output directory as an absolute Path (default: <repo>/output)."""
        return Path(self.ESFLOW_OUTPUT_DIR or (_REPO_ROOT / "output")).resolve()

    def resolved_catalog_file(self) -> Path:
        """Return the tool catalog path (default: <repo>/tools/tool_catalog.yaml)."""
        return Path(
            self.ESFLOW_CATALOG_FILE
            or (_REPO_ROOT / "tools" / "tool_catalog.yaml")
        ).resolve()

    def resolved_validation_cache_dir(self) -> Path:
        """Return the validation cache dir (default: <output_dir>/.validation_cache)."""
        if self.ESFLOW_VALIDATION_CACHE_DIR:
            return Path(self.ESFLOW_VALIDATION_CACHE_DIR).resolve()
        return self.resolved_output_dir() / ".validation_cache"

    def resolved_years(self) -> list[int]:
        """Parse ESFLOW_YEARS into a list of ints (empty list if not set)."""
        if not self.ESFLOW_YEARS:
            return []
        return [int(y.strip()) for y in self.ESFLOW_YEARS.split(",") if y.strip()]

    def obs_paths(self, data_dir: Optional[Path] = None) -> dict:
        """Return a dict of standard observation sub-paths under *data_dir*.

        Keys: gauge_metadata, streamflow_dir, basin_polygons, ilamb_cache
        """
        base = (data_dir or self.resolved_data_dir()) / self.ESFLOW_OBS_SUBDIR
        return {
            "gauge_metadata":  base / self.ESFLOW_GAUGE_METADATA_FILENAME,
            "streamflow_dir":  base / self.ESFLOW_STREAMFLOW_SUBDIR,
            "basin_polygons":  base / self.ESFLOW_BASIN_POLYGONS_FILENAME,
            "ilamb_cache":     base / self.ESFLOW_ILAMB_CACHE_SUBDIR,
        }


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
