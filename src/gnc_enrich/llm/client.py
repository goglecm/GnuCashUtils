"""HTTP client wrapper for LLM calls with retries and optional warmup."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Iterable

from gnc_enrich.config import LlmConfig, LlmMode

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class LlmClient:
    """Small helper around an LLM HTTP endpoint.

    This wrapper centralises timeout, retries, and connection reuse so that all
    LLM call sites behave consistently. Long-running processes that create many
    clients (e.g. a review server handling many LLM requests) should call
    close() when done or use the context manager to avoid leaving connections open.
    """

    config: LlmConfig
    _session: Any | None = field(default=None, init=False, repr=False)

    def close(self) -> None:
        """Release the HTTP session. Safe to call multiple times."""
        if self._session is not None:
            try:
                self._session.close()
            except Exception:
                pass
            self._session = None

    def __enter__(self) -> LlmClient:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    @property
    def enabled(self) -> bool:
        return self.config.mode != LlmMode.DISABLED and bool(
            self.config.endpoint and self.config.model_name
        )

    def _get_session(self):
        if self._session is None:
            import requests

            self._session = requests.Session()
        return self._session

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> dict[str, Any] | None:
        """Send a minimal chat-completions style request.

        Returns the decoded JSON response dict on success or None on failure.
        """
        if not self.enabled:
            return None

        import requests

        payload: dict[str, Any] = {
            "model": self.config.model_name,
            "messages": messages,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if temperature is not None:
            payload["temperature"] = temperature

        headers: dict[str, str] = {}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        session = self._get_session()
        timeout = self.config.timeout_seconds

        backoffs: Iterable[float] = (0.0, 0.5, 1.0)
        last_error: Exception | None = None
        for delay in backoffs:
            if delay:
                time.sleep(delay)
            try:
                t0 = time.perf_counter()
                resp = session.post(
                    self.config.endpoint,
                    json=payload,
                    headers=headers,
                    timeout=timeout,
                )
                elapsed = time.perf_counter() - t0
                logger.info(
                    "LLM request: %d chars, response_time=%.2fs",
                    sum(len(str(m.get("content", ""))) for m in messages),
                    elapsed,
                )
                resp.raise_for_status()
                return resp.json()
            except requests.RequestException as exc:  # type: ignore[attr-defined]
                last_error = exc
                logger.warning("LLM request failed (will retry if attempts remain)", exc_info=True)
                continue
        if last_error is not None:
            logger.warning("LLM request failed after retries: %s", last_error)
        return None

    def warmup(self) -> None:
        """Optionally send a tiny request to warm up the model/backend.

        Failures are logged but never raised.
        """
        if not self.enabled:
            return
        try:
            _ = self.chat(
                messages=[{"role": "user", "content": "Reply with the word OK."}],
                max_tokens=10,
            )
        except Exception:
            # chat() already logs; this is extra protection against unexpected errors
            logger.warning("LLM warmup failed", exc_info=True)

