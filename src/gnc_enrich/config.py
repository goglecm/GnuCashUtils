"""Application configuration models."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class RunConfig:
    gnucash_path: Path
    emails_dir: Path
    receipts_dir: Path
    processed_receipts_dir: Path
    state_dir: Path
    date_window_days: int = 7
    amount_tolerance: float = 0.50
    include_skipped: bool = False


@dataclass(slots=True)
class ApplyConfig:
    state_dir: Path
    create_backup: bool = True
    backup_dir: Path | None = None
    in_place: bool = True


@dataclass(slots=True)
class ReviewConfig:
    state_dir: Path
    host: str = "127.0.0.1"
    port: int = 7860
