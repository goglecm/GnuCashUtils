"""Tests for email and receipt matching logic."""

from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path

from gnc_enrich.domain.models import EmailEvidence, ReceiptEvidence, Split, Transaction
from gnc_enrich.email.index import EmailIndexRepository
from gnc_enrich.matching.email_matcher import EmailMatcher
from gnc_enrich.matching.receipt_matcher import ReceiptMatcher

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "emails"


def _make_tx(
    tx_id: str = "tx1",
    amount: str = "25.00",
    posted: date = date(2025, 1, 15),
    description: str = "Card Payment",
) -> Transaction:
    return Transaction(
        tx_id=tx_id,
        posted_date=posted,
        description=description,
        currency="GBP",
        amount=Decimal(amount),
        splits=[
            Split(account_path="Current Account", amount=Decimal(f"-{amount}")),
            Split(account_path="Unspecified", amount=Decimal(amount)),
        ],
    )


def _make_email(
    eid: str = "e1",
    sender: str = "shop@example.com",
    subject: str = "Your order",
    sent_at: datetime = datetime(2025, 1, 15, 12, 0, tzinfo=timezone.utc),
    amounts: list[str] | None = None,
) -> EmailEvidence:
    return EmailEvidence(
        evidence_id=eid,
        message_id=f"<{eid}@test>",
        sender=sender,
        subject=subject,
        sent_at=sent_at,
        parsed_amounts=[Decimal(a) for a in (amounts or [])],
    )


def _make_receipt(
    rid: str = "r1",
    total: str | None = "25.00",
    text: str = "Total: 25.00",
) -> ReceiptEvidence:
    return ReceiptEvidence(
        evidence_id=rid,
        source_path=f"/tmp/{rid}.jpg",
        ocr_text=text,
        parsed_total=Decimal(total) if total else None,
    )


# -- Email matcher ------------------------------------------------------------


class TestEmailMatcher:

    def _build_index(self, tmp_path: Path) -> EmailIndexRepository:
        repo = EmailIndexRepository()
        repo.build_or_load(FIXTURES_DIR, tmp_path)
        return repo

    def test_match_returns_ranked_results(self, tmp_path: Path) -> None:
        index = self._build_index(tmp_path)
        matcher = EmailMatcher(index, date_window_days=7, amount_tolerance=0.50)

        tx = _make_tx(amount="25.00", posted=date(2025, 1, 15), description="Card Payment")
        results = matcher.match(tx)
        assert len(results) >= 1
        assert all(r.relevance_score >= 0 for r in results)
        scores = [r.relevance_score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_amount_match_scores_high(self, tmp_path: Path) -> None:
        index = self._build_index(tmp_path)
        matcher = EmailMatcher(index, date_window_days=7, amount_tolerance=0.50)

        tx = _make_tx(amount="25.00", posted=date(2025, 1, 15))
        results = matcher.match(tx)
        if results:
            top = results[0]
            assert top.relevance_score > 1.0

    def test_no_match_outside_date_and_amount(self, tmp_path: Path) -> None:
        index = self._build_index(tmp_path)
        matcher = EmailMatcher(index, date_window_days=7, amount_tolerance=0.50)

        tx = _make_tx(amount="99999.00", posted=date(2030, 6, 15))
        results = matcher.match(tx)
        assert len(results) == 0

    def test_date_window_filtering(self, tmp_path: Path) -> None:
        index = self._build_index(tmp_path)
        matcher = EmailMatcher(index, date_window_days=1, amount_tolerance=0.50)

        tx = _make_tx(amount="25.00", posted=date(2025, 6, 15))
        results = matcher.match(tx)
        assert len(results) == 0

    def test_text_token_boost(self, tmp_path: Path) -> None:
        index = self._build_index(tmp_path)
        matcher = EmailMatcher(index, date_window_days=30, amount_tolerance=100.0)

        tx = _make_tx(description="Netflix Subscription", posted=date(2025, 1, 20))
        results = matcher.match(tx)
        if results:
            netflix_results = [r for r in results if "netflix" in r.sender.lower()]
            assert len(netflix_results) >= 1


# -- Receipt matcher ----------------------------------------------------------


class TestReceiptMatcher:

    def test_exact_amount_match(self) -> None:
        receipt = _make_receipt("r1", total="25.00")
        matcher = ReceiptMatcher([receipt])
        tx = _make_tx(amount="25.00")
        result = matcher.match(tx)
        assert result is not None
        assert result.evidence_id == "r1"
        assert result.relevance_score > 0

    def test_within_tolerance(self) -> None:
        receipt = _make_receipt("r1", total="25.30")
        matcher = ReceiptMatcher([receipt], amount_tolerance=0.50)
        tx = _make_tx(amount="25.00")
        result = matcher.match(tx)
        assert result is not None

    def test_outside_tolerance(self) -> None:
        receipt = _make_receipt("r1", total="30.00")
        matcher = ReceiptMatcher([receipt], amount_tolerance=0.50)
        tx = _make_tx(amount="25.00")
        result = matcher.match(tx)
        assert result is None

    def test_no_total_low_score(self) -> None:
        receipt = _make_receipt("r1", total=None)
        matcher = ReceiptMatcher([receipt])
        tx = _make_tx(amount="25.00")
        result = matcher.match(tx)
        assert result is not None
        assert result.relevance_score < 1.0

    def test_one_to_one_prevents_double_match(self) -> None:
        receipt = _make_receipt("r1", total="25.00")
        matcher = ReceiptMatcher([receipt])

        tx1 = _make_tx("tx1", amount="25.00")
        tx2 = _make_tx("tx2", amount="25.00")

        result1 = matcher.match(tx1)
        result2 = matcher.match(tx2)
        assert result1 is not None
        assert result2 is None

    def test_best_match_selected(self) -> None:
        r_close = _make_receipt("r1", total="25.10")
        r_exact = _make_receipt("r2", total="25.00")
        r_far = _make_receipt("r3", total="24.00")

        matcher = ReceiptMatcher([r_close, r_exact, r_far], amount_tolerance=0.50)
        tx = _make_tx(amount="25.00")
        result = matcher.match(tx)
        assert result is not None
        assert result.evidence_id == "r2"

    def test_no_receipts_returns_none(self) -> None:
        matcher = ReceiptMatcher([])
        tx = _make_tx(amount="25.00")
        assert matcher.match(tx) is None

    def test_is_amount_compatible(self) -> None:
        receipt = _make_receipt("r1", total="25.00")
        matcher = ReceiptMatcher([], amount_tolerance=0.50)
        tx = _make_tx(amount="25.30")
        assert matcher.is_amount_compatible(receipt, tx) is True
        tx_far = _make_tx(amount="30.00")
        assert matcher.is_amount_compatible(receipt, tx_far) is False

    def test_is_amount_compatible_no_total(self) -> None:
        receipt = _make_receipt("r1", total=None)
        matcher = ReceiptMatcher([])
        tx = _make_tx(amount="25.00")
        assert matcher.is_amount_compatible(receipt, tx) is False

    def test_assigned_ids_tracked(self) -> None:
        receipt = _make_receipt("r1", total="25.00")
        matcher = ReceiptMatcher([receipt])
        tx = _make_tx(amount="25.00")
        matcher.match(tx)
        assert "r1" in matcher.assigned_ids
