"""Minimal web app bootstrap for one-by-one transaction review."""


class ReviewWebApp:
    """Lifecycle wrapper around the chosen web framework."""

    def run(self, host: str, port: int) -> None:
        raise NotImplementedError
