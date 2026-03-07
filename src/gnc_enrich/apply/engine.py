"""Apply approved changes with backup, undo journal, and audit trail."""

from pathlib import Path


class ApplyEngine:
    def generate_dry_run_report(self, state_dir: Path) -> Path:
        raise NotImplementedError

    def apply(self, state_dir: Path) -> None:
        raise NotImplementedError

    def rollback(self, state_dir: Path, journal_id: str) -> None:
        raise NotImplementedError
