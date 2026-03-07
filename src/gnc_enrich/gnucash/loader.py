"""GnuCash loading and writing adapters for XML/SQLite files."""

from pathlib import Path

from gnc_enrich.domain.models import Transaction


class GnuCashLoader:
    """Loads transactions and account metadata from GnuCash sources."""

    def load_transactions(self, source: Path) -> list[Transaction]:
        """Return all transactions from an XML/SQLite GnuCash file."""
        raise NotImplementedError


class GnuCashWriter:
    """Applies approved mutations to GnuCash files with backup support."""

    def write_changes(self, source: Path, in_place: bool = True) -> Path:
        """Persist approved changes and return output path."""
        raise NotImplementedError

    def create_backup(self, source: Path, backup_dir: Path) -> Path:
        """Create backup snapshot and return backup path."""
        raise NotImplementedError
