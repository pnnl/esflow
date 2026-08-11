import logging

from common.logging_setup import configure_console_logging


def test_configure_console_logging_enables_only_esmflow_namespaces(monkeypatch):
    root_logger = logging.getLogger()
    esmflow_handler = next(
        (handler for handler in root_logger.handlers if getattr(handler, "_esmflow_console", False)),
        None,
    )
    if esmflow_handler is not None:
        root_logger.removeHandler(esmflow_handler)

    monkeypatch.setattr(logging.getLogger("common"), "level", logging.NOTSET)
    monkeypatch.setattr(logging.getLogger("esmflow"), "level", logging.NOTSET)
    configure_console_logging()

    assert logging.getLogger("common.workflow_runner").isEnabledFor(logging.INFO)
    assert logging.getLogger("esmflow.tools.example").isEnabledFor(logging.INFO)
    assert not logging.getLogger("httpx").isEnabledFor(logging.INFO)

    root_logger.removeHandler(
        next(handler for handler in root_logger.handlers if getattr(handler, "_esmflow_console", False))
    )
