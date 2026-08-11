"""Console logging configuration for ESMFlow entrypoints."""

import logging
import sys


def configure_console_logging(level: int = logging.INFO) -> None:
    """Show ESMFlow execution logs without enabling third-party INFO logs."""
    root_logger = logging.getLogger()
    if not any(getattr(handler, "_esmflow_console", False) for handler in root_logger.handlers):
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
        handler._esmflow_console = True
        root_logger.addHandler(handler)

    logging.getLogger("common").setLevel(level)
    logging.getLogger("esmflow").setLevel(level)
