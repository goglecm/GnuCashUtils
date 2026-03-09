"""Receipt file management: listing unprocessed and moving to processed dir."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

_RECEIPT_GLOBS = (
    "*.jpg",
    "*.jpeg",
    "*.png",
    "*.heic",
    "*.heif",
    "*.JPG",
    "*.JPEG",
    "*.PNG",
    "*.HEIC",
    "*.HEIF",
)


class ReceiptRepository:
    """Manages receipt image files: listing unprocessed and archiving processed."""

    def list_unprocessed(self, receipts_dir: Path) -> list[Path]:
        """Return all receipt image files in the given directory."""
        if not receipts_dir.is_dir():
            return []
        seen: set[Path] = set()
        files: list[Path] = []
        for pattern in _RECEIPT_GLOBS:
            for f in receipts_dir.glob(pattern):
                if f not in seen:
                    seen.add(f)
                    files.append(f)
        files.sort(key=lambda p: p.name)
        return files

    def mark_processed(self, receipt_path: Path, processed_dir: Path) -> Path:
        """Move a receipt to the processed directory with conflict-safe naming."""
        processed_dir.mkdir(parents=True, exist_ok=True)
        dest = processed_dir / receipt_path.name

        if dest.exists():
            stem = receipt_path.stem
            suffix = receipt_path.suffix
            counter = 1
            while dest.exists():
                dest = processed_dir / f"{stem}_{counter}{suffix}"
                counter += 1

        shutil.move(str(receipt_path), str(dest))
        logger.info("Moved receipt %s -> %s", receipt_path, dest)
        return dest
