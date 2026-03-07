"""EML parsing primitives."""

from pathlib import Path

from gnc_enrich.domain.models import EmailEvidence


class EmlParser:
    """Parses .eml files into searchable evidence records."""

    def parse(self, eml_path: Path) -> EmailEvidence:
        raise NotImplementedError
