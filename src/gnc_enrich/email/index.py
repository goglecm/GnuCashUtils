"""Persistent email indexing APIs."""

from pathlib import Path

from gnc_enrich.domain.models import EmailEvidence


class EmailIndexRepository:
    """Stores and queries parsed email evidence."""

    def build_or_load(self, emails_dir: Path, state_dir: Path) -> None:
        raise NotImplementedError

    def search(self, query_text: str, limit: int = 20) -> list[EmailEvidence]:
        raise NotImplementedError
