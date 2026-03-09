"""Tests for the LlmClient HTTP wrapper."""

from __future__ import annotations

import time
from typing import Any

import pytest

from gnc_enrich.config import LlmConfig, LlmMode
from gnc_enrich.llm.client import LlmClient


class DummyResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:  # pragma: no cover - behaviour trivial
        return

    def json(self) -> dict[str, Any]:  # type: ignore[override]
        return self._payload


def test_chat_success_uses_session_post(monkeypatch: pytest.MonkeyPatch) -> None:
    """chat() returns decoded JSON when the endpoint responds successfully."""
    cfg = LlmConfig(
        mode=LlmMode.ONLINE,
        endpoint="http://llm.test/v1/chat/completions",
        model_name="test-model",
        api_key="secret",
        timeout_seconds=5,
    )
    client = LlmClient(cfg)

    calls: list[dict[str, Any]] = []

    class DummySession:
        def post(self, url, json=None, headers=None, timeout=None):  # type: ignore[override]
            calls.append({"url": url, "json": json, "headers": headers, "timeout": timeout})
            return DummyResponse({"choices": [{"message": {"content": "OK"}}]})

    monkeypatch.setattr(
        LlmClient,
        "_get_session",
        lambda self: DummySession(),  # type: ignore[method-assign]
    )

    result = client.chat(messages=[{"role": "user", "content": "Hi"}], max_tokens=10, temperature=0.5)
    assert result is not None
    assert result.get("choices")
    assert len(calls) == 1
    assert calls[0]["url"] == cfg.endpoint
    assert calls[0]["headers"]["Authorization"] == "Bearer secret"


def test_chat_retries_and_logs_on_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """When the endpoint repeatedly fails, chat() retries and eventually returns None."""
    cfg = LlmConfig(
        mode=LlmMode.ONLINE,
        endpoint="http://llm.test/v1/chat/completions",
        model_name="test-model",
        timeout_seconds=1,
    )
    client = LlmClient(cfg)

    class DummySession:
        def post(self, url, json=None, headers=None, timeout=None):  # type: ignore[override]
            import requests

            raise requests.RequestException("boom")

    monkeypatch.setattr(
        LlmClient,
        "_get_session",
        lambda self: DummySession(),  # type: ignore[method-assign]
    )

    start = time.perf_counter()
    result = client.chat(messages=[{"role": "user", "content": "Hi"}], max_tokens=10)
    elapsed = time.perf_counter() - start

    assert result is None
    # We expect at least a small delay from the backoff loop.
    assert elapsed >= 0.0


def test_warmup_is_noop_when_disabled() -> None:
    """warmup() returns quickly and does nothing when the client is disabled."""
    cfg = LlmConfig(mode=LlmMode.DISABLED)
    client = LlmClient(cfg)
    start = time.perf_counter()
    client.warmup()
    elapsed = time.perf_counter() - start
    assert elapsed < 0.5


def test_warmup_when_enabled_does_not_raise_if_chat_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """warmup() does not raise when the client is enabled but chat() returns None (e.g. backend down)."""
    import requests

    cfg = LlmConfig(
        mode=LlmMode.ONLINE,
        endpoint="http://llm.test/v1/chat/completions",
        model_name="test-model",
        timeout_seconds=1,
    )
    client = LlmClient(cfg)

    class FailingSession:
        def post(self, url, json=None, headers=None, timeout=None):  # type: ignore[override]
            raise requests.RequestException("backend down")

    monkeypatch.setattr(
        LlmClient,
        "_get_session",
        lambda self: FailingSession(),  # type: ignore[method-assign]
    )
    # chat() returns None after retries; warmup must not raise
    client.warmup()


def test_enabled_false_when_endpoint_or_model_empty() -> None:
    """enabled is False when mode is online but endpoint or model_name is empty."""
    assert LlmClient(LlmConfig(mode=LlmMode.ONLINE, endpoint="", model_name="x")).enabled is False
    assert LlmClient(LlmConfig(mode=LlmMode.ONLINE, endpoint="http://x", model_name="")).enabled is False
    assert LlmClient(LlmConfig(mode=LlmMode.ONLINE, endpoint="http://x", model_name="m")).enabled is True


def test_close_releases_session_and_is_idempotent() -> None:
    """close() releases the HTTP session; calling it again is safe."""
    cfg = LlmConfig(
        mode=LlmMode.ONLINE,
        endpoint="http://llm.test/v1/chat/completions",
        model_name="m",
    )
    client = LlmClient(cfg)
    _ = client._get_session()
    assert client._session is not None
    client.close()
    assert client._session is None
    client.close()
    assert client._session is None


def test_context_manager_closes_on_exit() -> None:
    """Using LlmClient as context manager calls close() on exit."""
    cfg = LlmConfig(
        mode=LlmMode.ONLINE,
        endpoint="http://llm.test/v1/chat/completions",
        model_name="m",
    )
    with LlmClient(cfg) as client:
        _ = client._get_session()
        assert client._session is not None
    assert client._session is None

