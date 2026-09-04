"""Deterministic tests for benchmark/run_benchmark.py's _is_transient_failure().

No Docker or live LLM calls anywhere in this module.
"""

import httpx
import pytest
from pydantic_ai.exceptions import ModelAPIError, ModelHTTPError, UsageLimitExceeded

from benchmark.run_benchmark import _is_transient_failure


# ---------------------------------------------------------------------------
# ModelHTTPError -- transient iff the status code is in _TRANSIENT_STATUS_CODES
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("status_code", [500, 502, 503, 504])
def test_model_http_error_transient_5xx_is_retried(status_code):
    exc = ModelHTTPError(status_code=status_code, model_name="fake-model")
    assert _is_transient_failure(exc) is True


@pytest.mark.parametrize("status_code", [400, 401, 403, 404, 422, 429])
def test_model_http_error_non_transient_status_is_not_retried(status_code):
    exc = ModelHTTPError(status_code=status_code, model_name="fake-model")
    assert _is_transient_failure(exc) is False


# ---------------------------------------------------------------------------
# ModelAPIError -- a bare (non-HTTP) ModelAPIError is always transient
# ---------------------------------------------------------------------------
def test_bare_model_api_error_is_always_transient():
    exc = ModelAPIError(model_name="fake-model", message="connection dropped")
    assert _is_transient_failure(exc) is True


# ---------------------------------------------------------------------------
# UsageLimitExceeded -- always transient (noisy per-attempt, not a real
# model-quality signal)
# ---------------------------------------------------------------------------
def test_usage_limit_exceeded_is_transient():
    exc = UsageLimitExceeded(message="request limit exceeded")
    assert _is_transient_failure(exc) is True


# ---------------------------------------------------------------------------
# Raw httpx transport-level exceptions
# ---------------------------------------------------------------------------
def test_httpx_read_timeout_is_transient():
    assert _is_transient_failure(httpx.ReadTimeout("timed out")) is True


def test_httpx_connect_timeout_is_transient():
    assert _is_transient_failure(httpx.ConnectTimeout("timed out")) is True


def test_httpx_connect_error_is_transient():
    assert _is_transient_failure(httpx.ConnectError("name resolution failed")) is True


def test_httpx_remote_protocol_error_is_transient():
    """Regression test: seen live as "Server disconnected without sending a
    response" during a Gemini 3.7 Flash single_agent chunk -- confirmed NOT
    retried before this exception type was added to the allowlist, forcing
    a full chunk re-run to recover one otherwise-transient failure."""
    exc = httpx.RemoteProtocolError("Server disconnected without sending a response.")
    assert _is_transient_failure(exc) is True


# ---------------------------------------------------------------------------
# Non-transient exceptions -- must NOT be retried (real model-/code-quality
# signal, or a bug in the harness itself)
# ---------------------------------------------------------------------------
def test_plain_value_error_is_not_transient():
    assert _is_transient_failure(ValueError("bad param")) is False


def test_plain_connection_error_is_not_transient():
    """A bare ConnectionError (not an httpx/pydantic-ai exception type) is
    not in the allowlist -- only the specific transport/API exception types
    above are treated as transient."""
    assert _is_transient_failure(ConnectionError("simulated")) is False
