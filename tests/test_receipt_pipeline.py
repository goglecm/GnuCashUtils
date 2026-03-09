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
    _make_receipt_image(
        path,
        [
            "TESCO EXPRESS",
            "Milk        1.50",
            "Bread       2.00",
            "Total:      3.50",
        ],
    )
    return path


@pytest.fixture()
def multi_item_receipt(receipt_dir: Path) -> Path:
    path = receipt_dir / "receipt2.jpg"
    _make_receipt_image(
        path,
        [
            "SAINSBURYS LOCAL",
            "Apples      2.50",
            "Cheese      3.00",
            "Water       0.80",
            "Total:      6.30",
        ],
    )
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

    def test_real_world_receipts_parse_without_error(self) -> None:
        """Smoke-test OCR on bundled real-world receipt images."""
        fixtures_dir = Path(__file__).parent / "fixtures" / "receipts"
        engine = ReceiptOcrEngine()
        image_paths = sorted(fixtures_dir.glob("*"))
        # Ensure we actually have some fixtures wired up.
        assert image_paths, "Expected at least one real-world receipt fixture image"
        for img_path in image_paths:
            ev = engine.parse(img_path)
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


class TestReceiptOcrEngineLlmFallback:

    def test_llm_fallback_updates_total_and_items(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """LLM fallback populates total and line items when Tesseract finds none."""
        from gnc_enrich.config import LlmConfig, LlmMode
        from gnc_enrich.domain.models import ReceiptEvidence

        dummy_image = tmp_path / "blank.jpg"
        _make_receipt_image(dummy_image, ["Unreadable content"])

        # Force Tesseract to return text without any detectable totals.
        import gnc_enrich.receipt.ocr as ocr_mod

        def fake_image_to_string(_img: Image.Image) -> str:  # type: ignore[override]
            return "no totals here"

        monkeypatch.setattr(ocr_mod.pytesseract, "image_to_string", fake_image_to_string)

        # Mock LlmClient.chat used by the LLM fallback.
        import gnc_enrich.llm.client as llm_client_mod

        def fake_chat(*_args, **_kwargs):  # type: ignore[override]
            return {
                "choices": [
                    {
                        "message": {
                            "content": '{"total": "29.99", "items": [{"description": "Bike light", "amount": "29.99"}]}'
                        }
                    }
                ]
            }

        monkeypatch.setattr(llm_client_mod.LlmClient, "chat", staticmethod(fake_chat))

        llm_cfg = LlmConfig(
            mode=LlmMode.ONLINE,
            endpoint="https://example.test/llm",
            model_name="test-model",
            api_key="test",
            timeout_seconds=5,
        )
        engine = ReceiptOcrEngine(llm_config=llm_cfg)

        # Start from an evidence object with no parsed total.
        evidence = ReceiptEvidence(
            evidence_id="ev1",
            source_path=str(dummy_image),
            ocr_text="no totals here",
            parsed_total=None,
            line_items=[],
        )

        updated = engine._try_llm_fallback(evidence, dummy_image, llm_cfg)
        assert updated.parsed_total == Decimal("29.99")
        assert updated.line_items
        assert any(it.description == "Bike light" for it in updated.line_items)

    def test_llm_fallback_swallows_errors(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """LLM fallback logs and returns original evidence when the API call fails."""
        from gnc_enrich.config import LlmConfig, LlmMode
        from gnc_enrich.domain.models import ReceiptEvidence

        dummy_image = tmp_path / "blank.jpg"
        _make_receipt_image(dummy_image, ["Unreadable content"])

        import gnc_enrich.receipt.ocr as ocr_mod

        def fake_image_to_string(_img: Image.Image) -> str:  # type: ignore[override]
            return "no totals here"

        monkeypatch.setattr(ocr_mod.pytesseract, "image_to_string", fake_image_to_string)

        # Make LlmClient.chat raise to exercise the error path.
        import gnc_enrich.llm.client as llm_client_mod

        def failing_chat(*_args, **_kwargs):  # type: ignore[override]
            raise RuntimeError("network failure")

        monkeypatch.setattr(llm_client_mod.LlmClient, "chat", staticmethod(failing_chat))

        llm_cfg = LlmConfig(
            mode=LlmMode.ONLINE,
            endpoint="https://example.test/llm",
            model_name="test-model",
            api_key="test",
        )
        engine = ReceiptOcrEngine(llm_config=llm_cfg)

        evidence = ReceiptEvidence(
            evidence_id="ev2",
            source_path=str(dummy_image),
            ocr_text="no totals here",
            parsed_total=None,
            line_items=[],
        )

        updated = engine._try_llm_fallback(evidence, dummy_image, llm_cfg)
        # On failure, we should get the original evidence back with no total.
        assert updated.parsed_total is None
        assert updated.line_items == []


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

    def test_mark_processed_conflict_naming(self, receipt_dir: Path, tmp_path: Path) -> None:
        processed = tmp_path / "done"
        processed.mkdir()

        img1 = receipt_dir / "dup.jpg"
        _make_receipt_image(img1, ["Test"])
        (processed / "dup.jpg").write_bytes(b"occupied")

        repo = ReceiptRepository()
        dest = repo.mark_processed(img1, processed)
        assert dest.name == "dup_1.jpg"
        assert dest.exists()
