"""Application configuration models."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class LlmMode(str, Enum):
    DISABLED = "disabled"
    OFFLINE = "offline"
    ONLINE = "online"


@dataclass(slots=True)
class LlmConfig:
    mode: LlmMode = LlmMode.DISABLED
    endpoint: str = ""
    model_name: str = ""
    api_key: str = ""
    temperature: float = 0.2
    max_tokens: int = 1024


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
    llm: LlmConfig = field(default_factory=LlmConfig)


@dataclass(slots=True)
class ApplyConfig:
    state_dir: Path
    create_backup: bool = True
    backup_dir: Path | None = None
    in_place: bool = True
    dry_run: bool = False


@dataclass(slots=True)
class ReviewConfig:
    state_dir: Path
    host: str = "127.0.0.1"
    port: int = 7860
