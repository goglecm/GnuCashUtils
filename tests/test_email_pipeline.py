"""Tests for EML parsing and email index repository."""

from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path

from gnc_enrich.email.parser import EmlParser
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


class TestEmailIndexRepository:

    def test_build_from_directory(self, tmp_path: Path) -> None:
        repo = EmailIndexRepository()
        repo.build_or_load(FIXTURES_DIR, tmp_path)
        assert len(repo.entries) == 5

    def test_index_persisted(self, tmp_path: Path) -> None:
        repo = EmailIndexRepository()
        repo.build_or_load(FIXTURES_DIR, tmp_path)

        index_file = tmp_path / "email_index.jsonl"
        assert index_file.exists()
        lines = [l for l in index_file.read_text().splitlines() if l.strip()]
        assert len(lines) == 5

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
        assert len(manifest["indexed_files"]) == 5
