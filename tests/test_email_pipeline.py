"""Tests for EML parsing and email index repository."""

from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path

from gnc_enrich.email.parser import EmlParser, _extract_amounts, _filter_body, _strip_html
from gnc_enrich.email.index import EmailIndexRepository

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "emails"


class TestEmlParser:

    def test_parse_basic_fields(self) -> None:
        parser = EmlParser()
        ev = parser.parse(FIXTURES_DIR / "order_confirm.eml")
        assert ev.sender == "orders@tesco.com"
        assert "Tesco order confirmation" in ev.subject
        assert ev.message_id == "<tesco-001@tesco.com>"

    def test_parse_date(self) -> None:
        parser = EmlParser()
        ev = parser.parse(FIXTURES_DIR / "order_confirm.eml")
        assert ev.sent_at.date() == date(2025, 1, 15)

    def test_extract_gbp_amounts_from_subject(self) -> None:
        parser = EmlParser()
        ev = parser.parse(FIXTURES_DIR / "order_confirm.eml")
        assert Decimal("25.00") in ev.parsed_amounts

    def test_extract_gbp_keyword_amounts(self) -> None:
        parser = EmlParser()
        ev = parser.parse(FIXTURES_DIR / "subscription.eml")
        assert Decimal("15.99") in ev.parsed_amounts

    def test_multiple_amounts_extracted(self) -> None:
        parser = EmlParser()
        ev = parser.parse(FIXTURES_DIR / "receipt_email.eml")
        assert Decimal("9.50") in ev.parsed_amounts
        assert Decimal("1.50") in ev.parsed_amounts

    def test_no_amounts_in_unrelated(self) -> None:
        parser = EmlParser()
        ev = parser.parse(FIXTURES_DIR / "unrelated.eml")
        assert ev.parsed_amounts == []

    def test_body_snippet_populated(self) -> None:
        parser = EmlParser()
        ev = parser.parse(FIXTURES_DIR / "order_confirm.eml")
        assert len(ev.body_snippet) > 0
        assert "Tesco" in ev.body_snippet

    def test_evidence_id_generated(self) -> None:
        parser = EmlParser()
        ev = parser.parse(FIXTURES_DIR / "order_confirm.eml")
        assert len(ev.evidence_id) == 16

    def test_full_body_preserved(self) -> None:
        parser = EmlParser()
        ev = parser.parse(FIXTURES_DIR / "order_confirm.eml")
        assert len(ev.full_body) > 0


class TestBodyFiltering:
    """Verify that HTML, signatures, and quoted replies are cleaned."""

    def test_strip_html_tags(self) -> None:
        assert _strip_html("<p>Hello</p>") == "Hello"
        assert _strip_html("<b>Bold</b> text") == "Bold text"

    def test_strip_html_collapses_whitespace(self) -> None:
        assert _strip_html("<div>  A   B  </div>") == "A B"

    def test_filter_body_removes_signature(self) -> None:
        body = "Hello\nThis is a message.\n-- \nJohn Doe\njohn@example.com"
        filtered = _filter_body(body)
        assert "John Doe" not in filtered
        assert "Hello" in filtered

    def test_filter_body_removes_sent_from(self) -> None:
        body = "Check this out.\nSent from my iPhone\nMore text"
        filtered = _filter_body(body)
        assert "iPhone" not in filtered
        assert "Check this out" in filtered

    def test_filter_body_removes_quoted_replies(self) -> None:
        body = "My reply.\n> Original message here\n> Another quoted line"
        filtered = _filter_body(body)
        assert "Original message" not in filtered
        assert "My reply" in filtered

    def test_amounts_from_signature_excluded(self) -> None:
        text = "You paid £50.00.\n-- \nCompany Ltd, registered capital £1,000,000"
        filtered = _filter_body(text)
        amounts = _extract_amounts(f"Subject {filtered}")
        assert Decimal("50.00") in amounts
        assert Decimal("1000000") not in amounts

    def test_extract_amounts_comma_formatting(self) -> None:
        assert _extract_amounts("£1,400.00") == [Decimal("1400.00")]
        assert _extract_amounts("£1,400") == [Decimal("1400")]
        assert _extract_amounts("GBP 1,234.56") == [Decimal("1234.56")]

    def test_filter_body_empty_string(self) -> None:
        assert _filter_body("") == ""

    def test_filter_body_only_signature(self) -> None:
        assert _filter_body("-- \nJohn Doe\njohn@example.com") == ""

    def test_filter_body_single_line_no_newline(self) -> None:
        assert _filter_body("Hello world") == "Hello world"

    def test_strip_html_empty_string(self) -> None:
        assert _strip_html("") == ""

    def test_extract_amounts_empty_string(self) -> None:
        assert _extract_amounts("") == []

    def test_strip_html_decodes_entities(self) -> None:
        assert "£100" in _strip_html("&pound;100")
        assert "£" in _strip_html("&#163;50")

    def test_filter_body_preserves_dashes_in_text(self) -> None:
        """'--see attached' should NOT trigger signature removal (only '-- ' does)."""
        body = "Hello\n--see attached receipt\nThanks"
        filtered = _filter_body(body)
        assert "see attached" in filtered


class TestEmailIndexRepository:

    def test_build_from_directory(self, tmp_path: Path) -> None:
        repo = EmailIndexRepository()
        repo.build_or_load(FIXTURES_DIR, tmp_path)
        assert len(repo.entries) == 13

    def test_index_persisted(self, tmp_path: Path) -> None:
        repo = EmailIndexRepository()
        repo.build_or_load(FIXTURES_DIR, tmp_path)

        index_file = tmp_path / "email_index.jsonl"
        assert index_file.exists()
        lines = [l for l in index_file.read_text().splitlines() if l.strip()]
        data_lines = [l for l in lines if '"_schema_version"' not in l]
        assert len(data_lines) == 13

    def test_incremental_update(self, tmp_path: Path) -> None:
        repo = EmailIndexRepository()
        repo.build_or_load(FIXTURES_DIR, tmp_path)
        initial_count = len(repo.entries)

        repo2 = EmailIndexRepository()
        repo2.build_or_load(FIXTURES_DIR, tmp_path)
        assert len(repo2.entries) == initial_count

    def test_search_by_amount(self, tmp_path: Path) -> None:
        repo = EmailIndexRepository()
        repo.build_or_load(FIXTURES_DIR, tmp_path)

        results = repo.search(amount=Decimal("25.00"))
        assert len(results) >= 1
        senders = {r.sender for r in results}
        assert "orders@tesco.com" in senders or "alerts@bank.co.uk" in senders

    def test_search_by_date_range(self, tmp_path: Path) -> None:
        repo = EmailIndexRepository()
        repo.build_or_load(FIXTURES_DIR, tmp_path)

        results = repo.search(
            date_from=date(2025, 1, 14),
            date_to=date(2025, 1, 16),
        )
        for r in results:
            assert date(2025, 1, 14) <= r.sent_at.date() <= date(2025, 1, 16)

    def test_build_with_min_date_excludes_older_emails(self, tmp_path: Path) -> None:
        repo = EmailIndexRepository()
        min_date = date(2025, 1, 20)
        repo.build_or_load(FIXTURES_DIR, tmp_path, min_date=min_date)
        for ev in repo.entries:
            ev_date = ev.sent_at.date() if hasattr(ev.sent_at, "date") else ev.sent_at
            assert ev_date >= min_date

    def test_search_by_text_tokens(self, tmp_path: Path) -> None:
        repo = EmailIndexRepository()
        repo.build_or_load(FIXTURES_DIR, tmp_path)

        results = repo.search("netflix")
        assert any("netflix" in r.sender.lower() for r in results)

    def test_search_by_date_amount_convenience(self, tmp_path: Path) -> None:
        repo = EmailIndexRepository()
        repo.build_or_load(FIXTURES_DIR, tmp_path)

        results = repo.search_by_date_amount(
            tx_date=date(2025, 1, 15),
            tx_amount=Decimal("25.00"),
            window_days=7,
        )
        assert len(results) >= 1

    def test_search_no_match(self, tmp_path: Path) -> None:
        repo = EmailIndexRepository()
        repo.build_or_load(FIXTURES_DIR, tmp_path)

        results = repo.search(amount=Decimal("99999.99"))
        assert len(results) == 0

    def test_manifest_tracks_files(self, tmp_path: Path) -> None:
        repo = EmailIndexRepository()
        repo.build_or_load(FIXTURES_DIR, tmp_path)

        import json
        manifest = json.loads((tmp_path / "email_index_manifest.json").read_text())
        assert "order_confirm.eml" in manifest["indexed_files"]
        assert len(manifest["indexed_files"]) == 13
