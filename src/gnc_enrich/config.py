"""Application configuration models."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class LlmMode(str, Enum):
    """LLM integration mode selection."""

    DISABLED = "disabled"
    OFFLINE = "offline"
    ONLINE = "online"


@dataclass(slots=True)
class LlmConfig:
    """Configuration for optional LLM API integration."""

    mode: LlmMode = LlmMode.DISABLED
    endpoint: str = ""
    model_name: str = ""
    api_key: str = ""
    temperature: float = 0.2
    max_tokens: int = 1024
    use_web: bool = False


@dataclass(slots=True)
class RunConfig:
    """Configuration for the pipeline run subcommand."""

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
    """Configuration for the apply subcommand."""

    state_dir: Path
    create_backup: bool = True
    backup_dir: Path | None = None
    in_place: bool = True
    dry_run: bool = False


@dataclass(slots=True)
class ReviewConfig:
    """Configuration for the review web app subcommand."""

    state_dir: Path
    host: str = "127.0.0.1"
    port: int = 7860
