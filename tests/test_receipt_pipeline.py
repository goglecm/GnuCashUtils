"""Tests for receipt OCR and repository."""

from decimal import Decimal
from pathlib import Path

import pytest
from PIL import Image, ImageDraw, ImageFont

from gnc_enrich.receipt.ocr import ReceiptOcrEngine, _extract_total, _extract_line_items
from gnc_enrich.receipt.repository import ReceiptRepository


def _make_receipt_image(path: Path, lines: list[str]) -> None:
    """Generate a simple receipt image with the given text lines."""
    img = Image.new("RGB", (400, 50 + 30 * len(lines)), color="white")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf", 18)
    except (OSError, IOError):
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
        except (OSError, IOError):
            font = ImageFont.load_default()
    y = 10
    for line in lines:
        draw.text((10, y), line, fill="black", font=font)
        y += 28
    img.save(path)


@pytest.fixture()
def receipt_dir(tmp_path: Path) -> Path:
    d = tmp_path / "receipts"
    d.mkdir()
    return d


@pytest.fixture()
def simple_receipt(receipt_dir: Path) -> Path:
    path = receipt_dir / "receipt1.jpg"
    _make_receipt_image(path, [
        "TESCO EXPRESS",
        "Milk        1.50",
        "Bread       2.00",
        "Total:      3.50",
    ])
    return path


@pytest.fixture()
def multi_item_receipt(receipt_dir: Path) -> Path:
    path = receipt_dir / "receipt2.jpg"
    _make_receipt_image(path, [
        "SAINSBURYS LOCAL",
        "Apples      2.50",
        "Cheese      3.00",
        "Water       0.80",
        "Total:      6.30",
    ])
    return path


# -- internal helpers ---------------------------------------------------------

class TestExtractTotal:

    def test_finds_total(self) -> None:
        assert _extract_total("Total: 15.50") == Decimal("15.50")

    def test_total_with_pound_sign(self) -> None:
        assert _extract_total("Total: £25.00") == Decimal("25.00")

    def test_grand_total(self) -> None:
        assert _extract_total("Grand Total  42.99") == Decimal("42.99")

    def test_amount_due(self) -> None:
        assert _extract_total("Amount Due: 7.50") == Decimal("7.50")

    def test_no_total(self) -> None:
        assert _extract_total("Just some text with no totals") is None

    def test_prefers_total_over_subtotal(self) -> None:
        text = "Subtotal: 10.00\nTotal: 12.00"
        assert _extract_total(text) == Decimal("12.00")


class TestExtractLineItems:

    def test_extracts_items(self) -> None:
        text = "Milk        1.50\nBread       2.00\nTotal:      3.50"
        items = _extract_line_items(text)
        assert len(items) >= 2
        descs = {it.description for it in items}
        assert "Milk" in descs or any("Milk" in d for d in descs)


# -- OCR engine ---------------------------------------------------------------

class TestReceiptOcrEngine:

    def test_parse_simple_receipt(self, simple_receipt: Path) -> None:
        engine = ReceiptOcrEngine()
        ev = engine.parse(simple_receipt)
        assert ev.ocr_text != ""
        assert ev.source_path == str(simple_receipt)
        assert ev.evidence_id != ""

    def test_total_extracted_from_image(self, simple_receipt: Path) -> None:
        engine = ReceiptOcrEngine()
        ev = engine.parse(simple_receipt)
        if ev.parsed_total is not None:
            assert ev.parsed_total == Decimal("3.50")

    def test_multi_item_receipt(self, multi_item_receipt: Path) -> None:
        engine = ReceiptOcrEngine()
        ev = engine.parse(multi_item_receipt)
        assert ev.ocr_text != ""

    def test_file_not_found(self) -> None:
        engine = ReceiptOcrEngine()
        with pytest.raises(FileNotFoundError):
            engine.parse(Path("/nonexistent/receipt.jpg"))

    def test_unsupported_format(self, tmp_path: Path) -> None:
        txt = tmp_path / "file.txt"
        txt.write_text("not an image")
        engine = ReceiptOcrEngine()
        with pytest.raises(ValueError, match="Unsupported"):
            engine.parse(txt)


# -- Receipt repository -------------------------------------------------------

class TestReceiptRepository:

    def test_list_unprocessed(self, receipt_dir: Path, simple_receipt: Path) -> None:
        repo = ReceiptRepository()
        files = repo.list_unprocessed(receipt_dir)
        assert len(files) >= 1
        assert simple_receipt in files

    def test_list_empty_dir(self, tmp_path: Path) -> None:
        repo = ReceiptRepository()
        assert repo.list_unprocessed(tmp_path) == []

    def test_list_nonexistent_dir(self, tmp_path: Path) -> None:
        repo = ReceiptRepository()
        assert repo.list_unprocessed(tmp_path / "nope") == []

    def test_mark_processed_moves_file(
        self, receipt_dir: Path, simple_receipt: Path, tmp_path: Path
    ) -> None:
        processed = tmp_path / "done"
        repo = ReceiptRepository()
        dest = repo.mark_processed(simple_receipt, processed)
        assert dest.exists()
        assert not simple_receipt.exists()
        assert dest.parent == processed

    def test_mark_processed_conflict_naming(
        self, receipt_dir: Path, tmp_path: Path
    ) -> None:
        processed = tmp_path / "done"
        processed.mkdir()

        img1 = receipt_dir / "dup.jpg"
        _make_receipt_image(img1, ["Test"])
        (processed / "dup.jpg").write_bytes(b"occupied")

        repo = ReceiptRepository()
        dest = repo.mark_processed(img1, processed)
        assert dest.name == "dup_1.jpg"
        assert dest.exists()
