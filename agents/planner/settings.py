from __future__ import annotations

import os

from common.workflow import Settings


def default_settings() -> Settings:
    """Build server-side default workflow settings."""
    return Settings(
        data_dir=os.getenv("ESFLOW_DATA_DIR", "./data/e3sm"),
        output_dir=os.getenv("ESFLOW_OUTPUT_DIR", "./outputs"),
    )
