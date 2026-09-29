from __future__ import annotations

from common.config import runtime_config
from common.workflow import Settings


def default_settings() -> Settings:
    """Build server-side default workflow settings from environment / .env.

    All paths are resolved through ``RuntimeConfig`` so that no directory
    names are hard-coded here.  Set ``ESFLOW_DATA_DIR``, ``ESFLOW_OUTPUT_DIR``,
    and ``ESFLOW_CASE_NAME`` in your ``.env`` (or shell environment) to point
    at your actual data and output locations.
    """
    return Settings(
        data_dir=str(runtime_config.resolved_data_dir()),
        output_dir=str(runtime_config.resolved_output_dir()),
        case_name=runtime_config.ESFLOW_CASE_NAME or None,
    )
