"""Persistence layer for receipt evidence and move lifecycle."""

from pathlib import Path


class ReceiptRepository:
    def list_unprocessed(self, receipts_dir: Path) -> list[Path]:
        raise NotImplementedError

    def mark_processed(self, receipt_path: Path, processed_dir: Path) -> Path:
        raise NotImplementedError
