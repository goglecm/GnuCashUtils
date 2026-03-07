"""Tests for new features: recursive emails, new categories, evidence-driven
descriptions, terse item lookup, evidence approval, and graceful no-evidence."""

from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

from gnc_enrich.config import LlmConfig, LlmMode, RunConfig
from gnc_enrich.domain.models import (
    EmailEvidence,
    EvidencePacket,
    LineItem,
    Proposal,
    ReceiptEvidence,
    ReviewDecision,
    Split,
    Transaction,
)
from gnc_enrich.email.index import EmailIndexRepository
from gnc_enrich.email.parser import EmlParser
from gnc_enrich.gnucash.loader import GnuCashLoader, GnuCashWriter
from gnc_enrich.ml.predictor import CategoryPredictor
from gnc_enrich.review.service import ReviewQueueService
from gnc_enrich.review.webapp import create_app
from gnc_enrich.state.repository import StateRepository


FIXTURES_DIR = Path(__file__).parent / "fixtures" / "emails"


def _make_email(
    eid: str = "em1",
    sender: str = "shop@example.com",
    subject: str = "Your order",
    body: str = "Thanks for purchasing Widget Pro",
    amount: Decimal | None = Decimal("25.00"),
) -> EmailEvidence:
    return EmailEvidence(
        evidence_id=eid,
        message_id=f"<{eid}@test>",
        sender=sender,
        subject=subject,
        sent_at=datetime(2025, 1, 15, tzinfo=timezone.utc),
        body_snippet=body[:200],
        full_body=body,
        parsed_amounts=[amount] if amount else [],
    )


def _make_receipt(
    rid: str = "rc1",
    total: Decimal | None = Decimal("12.50"),
    items: list[LineItem] | None = None,
) -> ReceiptEvidence:
    return ReceiptEvidence(
        evidence_id=rid,
        source_path="/tmp/receipt.jpg",
        ocr_text="Some receipt text",
        parsed_total=total,
        line_items=items or [],
    )


def _make_tx(
    tx_id: str = "tx1",
    amount: Decimal = Decimal("25.00"),
    desc: str = "Card payment",
) -> Transaction:
    return Transaction(
        tx_id=tx_id,
        posted_date=date(2025, 1, 15),
        description=desc,
        currency="GBP",
        amount=amount,
    )


# ---------------------------------------------------------------------------
# Recursive email directory scanning
# ---------------------------------------------------------------------------


class TestRecursiveEmailScanning:

    def test_indexes_emails_in_subdirectories(self, tmp_path: Path) -> None:
        emails_dir = tmp_path / "emails"
        sub1 = emails_dir / "2024" / "jan"
        sub2 = emails_dir / "2024" / "feb"
        sub1.mkdir(parents=True)
        sub2.mkdir(parents=True)

        for d, name in [(sub1, "a.eml"), (sub2, "b.eml")]:
            (d / name).write_text(
                "From: test@example.com\n"
                "Subject: Hello\n"
                "Date: Wed, 15 Jan 2025 10:00:00 +0000\n"
                "Message-ID: <msg-{name}@test>\n"
                "\nBody text\n".format(name=name),
                encoding="utf-8",
            )

        state_dir = tmp_path / "state"
        repo = EmailIndexRepository()
        repo.build_or_load(emails_dir, state_dir)

        assert len(repo.entries) == 2

    def test_manifest_uses_relative_paths(self, tmp_path: Path) -> None:
        import json

        emails_dir = tmp_path / "emails"
        sub = emails_dir / "sub"
        sub.mkdir(parents=True)
        (sub / "test.eml").write_text(
            "From: a@b.com\nSubject: S\nDate: Wed, 15 Jan 2025 10:00:00 +0000\n"
            "Message-ID: <1@b>\n\nBody\n",
            encoding="utf-8",
        )

        state_dir = tmp_path / "state"
        repo = EmailIndexRepository()
        repo.build_or_load(emails_dir, state_dir)

        manifest = json.loads((state_dir / "email_index_manifest.json").read_text())
        assert "sub/test.eml" in manifest["indexed_files"]

    def test_incremental_with_subdirs(self, tmp_path: Path) -> None:
        emails_dir = tmp_path / "emails" / "sub"
        emails_dir.mkdir(parents=True)
        (emails_dir / "first.eml").write_text(
            "From: a@b.com\nSubject: First\nDate: Wed, 15 Jan 2025 10:00:00 +0000\n"
            "Message-ID: <first@b>\n\nBody\n",
            encoding="utf-8",
        )

        state_dir = tmp_path / "state"
        repo1 = EmailIndexRepository()
        repo1.build_or_load(tmp_path / "emails", state_dir)
        assert len(repo1.entries) == 1

        (emails_dir / "second.eml").write_text(
            "From: c@d.com\nSubject: Second\nDate: Thu, 16 Jan 2025 10:00:00 +0000\n"
            "Message-ID: <second@d>\n\nBody\n",
            encoding="utf-8",
        )

        repo2 = EmailIndexRepository()
        repo2.build_or_load(tmp_path / "emails", state_dir)
        assert len(repo2.entries) == 2


# ---------------------------------------------------------------------------
# Email parser stores full_body
# ---------------------------------------------------------------------------


class TestEmailParserFullBody:

    def test_full_body_populated(self) -> None:
        parser = EmlParser()
        ev = parser.parse(FIXTURES_DIR / "order_confirm.eml")
        assert ev.full_body
        assert len(ev.full_body) >= len(ev.body_snippet)


# ---------------------------------------------------------------------------
# Evidence-driven description enrichment
# ---------------------------------------------------------------------------


class TestEvidenceEnrichment:

    def test_enrich_from_email(self) -> None:
        predictor = CategoryPredictor()
        em = _make_email(body="You bought a Widget Pro from WidgetCo for £25.00.")
        result = predictor.enrich_description_from_evidence("Card payment", [em], None)
        assert "Widget" in result
        assert "Card payment" in result

    def test_enrich_from_receipt_with_items(self) -> None:
        predictor = CategoryPredictor()
        receipt = _make_receipt(
            items=[LineItem("Coffee", Decimal("3.50")), LineItem("Cake", Decimal("4.00"))],
            total=Decimal("7.50"),
        )
        result = predictor.enrich_description_from_evidence("Cafe purchase", None, receipt)
        assert "Coffee" in result
        assert "Cake" in result
        assert "Cafe purchase" in result

    def test_enrich_from_receipt_total_only(self) -> None:
        predictor = CategoryPredictor()
        receipt = _make_receipt(items=[], total=Decimal("42.00"))
        result = predictor.enrich_description_from_evidence("Payment", None, receipt)
        assert "42.00" in result

    def test_enrich_with_no_evidence(self) -> None:
        predictor = CategoryPredictor()
        result = predictor.enrich_description_from_evidence("Base desc", [], None)
        assert result == "Base desc"

    def test_enrich_both_email_and_receipt(self) -> None:
        predictor = CategoryPredictor()
        em = _make_email(body="Ordered a Book from Amazon")
        receipt = _make_receipt(items=[LineItem("Book", Decimal("9.99"))], total=Decimal("9.99"))
        result = predictor.enrich_description_from_evidence("Amazon", [em], receipt)
        assert "Book" in result
        assert "Amazon" in result


# ---------------------------------------------------------------------------
# LLM terse item description
# ---------------------------------------------------------------------------


class TestTerseItemLookup:

    def test_disabled_llm_returns_original_names(self) -> None:
        predictor = CategoryPredictor(llm_config=LlmConfig(mode=LlmMode.DISABLED))
        receipt = _make_receipt(items=[
            LineItem("BRD WHL", Decimal("1.20")),
            LineItem("MLK 2L", Decimal("1.50")),
        ])
        result = predictor.describe_terse_items(receipt)
        assert result == ["BRD WHL", "MLK 2L"]

    def test_empty_items_returns_empty(self) -> None:
        predictor = CategoryPredictor(llm_config=LlmConfig(mode=LlmMode.DISABLED))
        receipt = _make_receipt(items=[])
        result = predictor.describe_terse_items(receipt)
        assert result == []

    @patch("requests.post")
    def test_llm_expands_terse_items(self, mock_post) -> None:
        mock_post.return_value.status_code = 200
        mock_post.return_value.raise_for_status = lambda: None
        mock_post.return_value.json.return_value = {
            "choices": [{"message": {"content": '["Wholemeal Bread", "2 Litre Milk"]'}}]
        }

        predictor = CategoryPredictor(
            llm_config=LlmConfig(mode=LlmMode.OFFLINE, endpoint="http://llm:1234", model_name="test")
        )
        receipt = _make_receipt(items=[
            LineItem("BRD WHL", Decimal("1.20")),
            LineItem("MLK 2L", Decimal("1.50")),
        ])
        result = predictor.describe_terse_items(receipt)
        assert result == ["Wholemeal Bread", "2 Litre Milk"]
        mock_post.assert_called_once()


# ---------------------------------------------------------------------------
# New category creation in GnuCash writer
# ---------------------------------------------------------------------------


class TestNewCategoryCreation:

    def test_new_account_created_in_xml(self, sample_gnucash_path: Path, tmp_path: Path) -> None:
        loader = GnuCashLoader()
        loader.load_transactions(sample_gnucash_path)
        tree = loader.get_tree()

        writer = GnuCashWriter()
        changes = {
            "tx_unspec1": {
                "description": "Test purchase",
                "splits": [{"account_path": "Expenses:Hobbies:ModelTrains", "amount": "25.00"}],
            }
        }
        output = writer.write_changes(sample_gnucash_path, tree, changes, in_place=False)

        loader2 = GnuCashLoader()
        loader2.load_transactions(output)
        accounts = loader2.load_accounts(output)
        account_paths = {a.full_path for a in accounts}
        assert "Expenses:Hobbies:ModelTrains" in account_paths

    def test_existing_category_not_duplicated(self, sample_gnucash_path: Path, tmp_path: Path) -> None:
        loader = GnuCashLoader()
        loader.load_transactions(sample_gnucash_path)
        tree = loader.get_tree()
        accounts_before = loader.load_accounts(sample_gnucash_path)

        writer = GnuCashWriter()
        changes = {
            "tx_unspec1": {
                "description": "Test",
                "splits": [{"account_path": "Expenses:Food", "amount": "25.00"}],
            }
        }
        output = writer.write_changes(sample_gnucash_path, tree, changes, in_place=False)

        loader2 = GnuCashLoader()
        loader2.load_transactions(output)
        accounts_after = loader2.load_accounts(output)

        food_before = [a for a in accounts_before if a.name == "Food"]
        food_after = [a for a in accounts_after if a.name == "Food"]
        assert len(food_before) == len(food_after)


# ---------------------------------------------------------------------------
# Evidence approval in review decisions
# ---------------------------------------------------------------------------


class TestEvidenceApproval:

    def _seed(self, tmp_path: Path) -> tuple[StateRepository, ReviewQueueService]:
        state = StateRepository(tmp_path / "state")
        em = _make_email()
        receipt = _make_receipt()
        proposals = [
            Proposal(
                proposal_id="p1",
                tx_id="tx1",
                suggested_description="Test",
                suggested_splits=[Split("Expenses:Food", Decimal("25.00"))],
                confidence=0.8,
                rationale="test",
                evidence=EvidencePacket(tx_id="tx1", emails=[em], receipt=receipt),
            ),
        ]
        state.save_proposals(proposals)
        return state, ReviewQueueService(state)

    def test_decision_stores_approved_email_ids(self, tmp_path: Path) -> None:
        state, service = self._seed(tmp_path)
        dec = ReviewDecision(
            tx_id="tx1",
            action="approve",
            final_description="Approved",
            final_splits=[Split("Expenses:Food", Decimal("25.00"))],
            approved_email_ids=["em1"],
            approved_receipt=True,
        )
        service.submit_decision(dec)
        loaded = state.load_decisions()
        assert loaded[0].approved_email_ids == ["em1"]
        assert loaded[0].approved_receipt is True

    def test_decision_no_evidence_approved(self, tmp_path: Path) -> None:
        state, service = self._seed(tmp_path)
        dec = ReviewDecision(
            tx_id="tx1",
            action="approve",
            final_description="Approved",
            final_splits=[Split("Expenses:Food", Decimal("25.00"))],
        )
        service.submit_decision(dec)
        loaded = state.load_decisions()
        assert loaded[0].approved_email_ids == []
        assert loaded[0].approved_receipt is False

    def test_webapp_passes_approved_evidence(self, tmp_path: Path) -> None:
        state, service = self._seed(tmp_path)
        app = create_app(service)
        client = app.test_client()

        resp = client.post("/review/p1/decide", data={
            "action": "approve",
            "description": "Test desc",
            "split_path": "Expenses:Food",
            "split_amount": "25.00",
            "approved_email": "em1",
            "approved_receipt": "1",
        }, follow_redirects=False)
        assert resp.status_code in (302, 303)

        loaded = state.load_decisions()
        assert loaded[0].approved_email_ids == ["em1"]
        assert loaded[0].approved_receipt is True

    def test_webapp_no_evidence_checked(self, tmp_path: Path) -> None:
        state, service = self._seed(tmp_path)
        app = create_app(service)
        client = app.test_client()

        resp = client.post("/review/p1/decide", data={
            "action": "approve",
            "description": "Test desc",
            "split_path": "Expenses:Food",
            "split_amount": "25.00",
        }, follow_redirects=False)
        assert resp.status_code in (302, 303)

        loaded = state.load_decisions()
        assert loaded[0].approved_email_ids == []
        assert loaded[0].approved_receipt is False


# ---------------------------------------------------------------------------
# Graceful no-evidence handling
# ---------------------------------------------------------------------------


class TestGracefulNoEvidence:

    def test_pipeline_with_nonexistent_emails_dir(self, sample_gnucash_path: Path, tmp_path: Path) -> None:
        from gnc_enrich.services.pipeline import EnrichmentPipeline

        config = RunConfig(
            gnucash_path=sample_gnucash_path,
            emails_dir=tmp_path / "nonexistent_emails",
            receipts_dir=tmp_path / "nonexistent_receipts",
            processed_receipts_dir=tmp_path / "processed",
            state_dir=tmp_path / "state",
        )
        pipeline = EnrichmentPipeline()
        result = pipeline.run(config)
        assert result.proposal_count >= 0

    def test_pipeline_with_empty_emails_dir(self, sample_gnucash_path: Path, tmp_path: Path) -> None:
        from gnc_enrich.services.pipeline import EnrichmentPipeline

        emails_dir = tmp_path / "emails"
        emails_dir.mkdir()
        receipts_dir = tmp_path / "receipts"
        receipts_dir.mkdir()

        config = RunConfig(
            gnucash_path=sample_gnucash_path,
            emails_dir=emails_dir,
            receipts_dir=receipts_dir,
            processed_receipts_dir=tmp_path / "processed",
            state_dir=tmp_path / "state",
        )
        pipeline = EnrichmentPipeline()
        result = pipeline.run(config)
        assert result.proposal_count >= 1

    def test_predictor_handles_no_emails_no_receipt(self) -> None:
        tx = _make_tx()
        predictor = CategoryPredictor()
        proposal = predictor.propose(tx, [], None)
        assert proposal.suggested_description
        assert proposal.confidence > 0


class TestRefundDetection:
    """Rule 10: refunds should be categorized like the original transaction."""

    def test_refund_detected_by_positive_bank_split(self) -> None:
        """A refund has positive bank split (money flowing in)."""
        txs = []
        for i in range(1, 4):
            t = _make_tx(tx_id=f"t{i}", amount=Decimal("50.00"), desc="Tesco Groceries")
            t.splits = [Split(account_path="Expenses:Groceries", amount=t.amount)]
            t.original_category = "Expenses:Groceries"
            txs.append(t)
        for i in range(4, 7):
            t = _make_tx(tx_id=f"t{i}", amount=Decimal("20.00"), desc="Amazon Digital")
            t.splits = [Split(account_path="Expenses:Online", amount=t.amount)]
            t.original_category = "Expenses:Online"
            txs.append(t)

        predictor = CategoryPredictor(historical_transactions=txs)

        refund_tx = _make_tx(tx_id="t99", amount=Decimal("50.00"), desc="Tesco Refund")
        refund_tx.account_name = "Assets:Current Account"
        refund_tx.splits = [
            Split(account_path="Assets:Current Account", amount=Decimal("50.00")),
            Split(account_path="Imbalance-GBP", amount=Decimal("-50.00")),
        ]
        proposal = predictor.propose(refund_tx, [], None)
        assert "Refund detected" in proposal.rationale

    def test_refund_not_triggered_for_normal_expense(self) -> None:
        """Normal expenses have negative bank split (money flowing out)."""
        txs = []
        for i in range(1, 4):
            t = _make_tx(tx_id=f"t{i}", amount=Decimal("50.00"), desc="Tesco")
            t.splits = [Split(account_path="Expenses:Groceries", amount=t.amount)]
            t.original_category = "Expenses:Groceries"
            txs.append(t)
        for i in range(4, 7):
            t = _make_tx(tx_id=f"t{i}", amount=Decimal("20.00"), desc="Amazon")
            t.splits = [Split(account_path="Expenses:Online", amount=t.amount)]
            t.original_category = "Expenses:Online"
            txs.append(t)
        predictor = CategoryPredictor(historical_transactions=txs)

        normal_tx = _make_tx(tx_id="t99", amount=Decimal("25.00"), desc="Tesco")
        normal_tx.account_name = "Assets:Current Account"
        normal_tx.splits = [
            Split(account_path="Assets:Current Account", amount=Decimal("-25.00")),
            Split(account_path="Unspecified", amount=Decimal("25.00")),
        ]
        proposal = predictor.propose(normal_tx, [], None)
        assert "Refund detected" not in proposal.rationale


class TestTransferProposal:

    def test_transfer_proposal_keeps_original_splits(self) -> None:
        """Transfers get suggested_splits = original; no category change."""
        transfer_tx = Transaction(
            tx_id="tx_transfer",
            posted_date=date(2025, 1, 25),
            description="Transfer to Savings",
            currency="GBP",
            amount=Decimal("500.00"),
            splits=[
                Split(account_path="Current Account", amount=Decimal("-500.00")),
                Split(account_path="Savings Account", amount=Decimal("500.00")),
            ],
            account_name="Current Account",
            original_category="",
            is_transfer=True,
        )
        predictor = CategoryPredictor()
        proposal = predictor.propose(transfer_tx, [], None)
        assert proposal.suggested_splits == transfer_tx.splits
        assert proposal.confidence == 1.0
        assert "Transfer" in proposal.rationale
        assert "no split change" in proposal.confidence_breakdown[0].lower()


class TestSchemaVersionHeaders:
    """Spec Section 13: all state files must have schema version headers."""

    def test_decisions_jsonl_has_header(self, tmp_path: Path) -> None:
        state = StateRepository(tmp_path)
        dec = ReviewDecision(
            tx_id="tx1", action="approve", final_description="test",
            final_splits=[], decided_at=datetime.now(timezone.utc),
        )
        state.save_decision(dec)
        first_line = (tmp_path / "decisions.jsonl").read_text().splitlines()[0]
        import json
        assert "_schema_version" in json.loads(first_line)

    def test_audit_log_has_header(self, tmp_path: Path) -> None:
        from gnc_enrich.domain.models import AuditEntry
        state = StateRepository(tmp_path)
        entry = AuditEntry(
            entry_id="e1", tx_id="tx1", action="approve",
            proposed_description="p", proposed_splits=[],
            final_description="f", final_splits=[],
            confidence=0.9, evidence_ids=[],
            timestamp=datetime.now(timezone.utc),
        )
        state.append_audit(entry)
        first_line = (tmp_path / "audit_log.jsonl").read_text().splitlines()[0]
        import json
        assert "_schema_version" in json.loads(first_line)

    def test_feedback_events_has_header(self, tmp_path: Path) -> None:
        state = StateRepository(tmp_path)
        state.append_feedback({"type": "approve", "tx_id": "tx1"})
        first_line = (tmp_path / "feedback_events.jsonl").read_text().splitlines()[0]
        import json
        assert "_schema_version" in json.loads(first_line)

    def test_email_index_has_header(self, tmp_path: Path) -> None:
        emails_dir = tmp_path / "emails"
        emails_dir.mkdir()
        idx_dir = tmp_path / "state"
        idx_dir.mkdir()
        repo = EmailIndexRepository()
        repo.build_or_load(emails_dir, idx_dir)
        first_line = (idx_dir / "email_index.jsonl").read_text().splitlines()[0]
        import json
        assert "_schema_version" in json.loads(first_line)

    def test_email_manifest_has_schema_version(self, tmp_path: Path) -> None:
        emails_dir = tmp_path / "emails"
        emails_dir.mkdir()
        idx_dir = tmp_path / "state"
        idx_dir.mkdir()
        repo = EmailIndexRepository()
        repo.build_or_load(emails_dir, idx_dir)
        import json
        manifest = json.loads((idx_dir / "email_index_manifest.json").read_text())
        assert "_schema_version" in manifest

    def test_schema_header_skipped_on_load(self, tmp_path: Path) -> None:
        """Schema version lines must not appear as data when loading."""
        state = StateRepository(tmp_path)
        state.append_feedback({"type": "approve", "tx_id": "tx1"})
        state.append_feedback({"type": "skip", "tx_id": "tx2"})
        loaded = state.load_feedback()
        assert len(loaded) == 2
        assert all("_schema_version" not in d for d in loaded)


class TestEvidenceEnrichmentWiring:
    """Rule 14: evidence approval must actually enrich the description."""

    def test_submit_decision_enriches_from_approved_emails(self, tmp_path: Path) -> None:
        email_ev = EmailEvidence(
            evidence_id="em1", message_id="<m@x>", sender="shop@co.uk",
            subject="Order Confirmation",
            sent_at=datetime(2025, 2, 1, tzinfo=timezone.utc),
            body_snippet="Your order for Widget X",
            parsed_amounts=[Decimal("9.99")],
            full_body="Thank you for purchasing Widget X from our store.",
        )
        proposal = Proposal(
            proposal_id="p1", tx_id="tx1",
            suggested_description="Payment 01/02/2025", suggested_splits=[],
            confidence=0.8, rationale="ML prediction",
            evidence=EvidencePacket(
                tx_id="tx1", emails=[email_ev], receipt=None, similar_transactions=[],
            ),
        )
        state = StateRepository(tmp_path)
        state.save_proposals([proposal])

        svc = ReviewQueueService(state)

        decision = ReviewDecision(
            tx_id="tx1", action="approve",
            final_description="Payment 01/02/2025",
            final_splits=[], decided_at=None,
            approved_email_ids=["em1"], approved_receipt=False,
        )
        svc.submit_decision(decision)

        decisions = state.load_decisions()
        assert len(decisions) == 1
        assert "Widget X" in decisions[0].final_description or "Widget" in decisions[0].final_description

    def test_submit_decision_no_enrichment_without_approval(self, tmp_path: Path) -> None:
        proposal = Proposal(
            proposal_id="p2", tx_id="tx1",
            suggested_description="Payment 01/02/2025", suggested_splits=[],
            confidence=0.8, rationale="test",
            evidence=EvidencePacket(
                tx_id="tx1", emails=[], receipt=None, similar_transactions=[],
            ),
        )
        state = StateRepository(tmp_path)
        state.save_proposals([proposal])

        svc = ReviewQueueService(state)

        decision = ReviewDecision(
            tx_id="tx1", action="approve",
            final_description="Payment 01/02/2025",
            final_splits=[], decided_at=None,
            approved_email_ids=[], approved_receipt=False,
        )
        svc.submit_decision(decision)

        decisions = state.load_decisions()
        assert decisions[0].final_description == "Payment 01/02/2025"


class TestCornerCases:
    """Tests for corner cases fixed during code audit."""

    def test_parse_fraction_zero_denominator(self) -> None:
        """_parse_fraction('0/0') must not raise ZeroDivisionError."""
        from gnc_enrich.gnucash.loader import _parse_fraction
        result = _parse_fraction("0/0")
        assert result == Decimal(0)

    def test_parse_fraction_garbage(self) -> None:
        from gnc_enrich.gnucash.loader import _parse_fraction
        result = _parse_fraction("abc/def")
        assert result == Decimal(0)

    def test_parse_fraction_normal(self) -> None:
        from gnc_enrich.gnucash.loader import _parse_fraction
        assert _parse_fraction("1500/100") == Decimal("15")

    def test_jsonl_corrupt_line_skipped(self, tmp_path: Path) -> None:
        """A corrupt JSONL line must be skipped, not crash the load."""
        state = StateRepository(tmp_path)
        state.append_feedback({"type": "test", "tx_id": "tx1"})
        fb_path = tmp_path / "feedback_events.jsonl"
        with fb_path.open("a", encoding="utf-8") as f:
            f.write("{corrupt json\n")
        state.append_feedback({"type": "test", "tx_id": "tx2"})
        loaded = state.load_feedback()
        assert len(loaded) == 2
        assert loaded[0]["tx_id"] == "tx1"
        assert loaded[1]["tx_id"] == "tx2"

    def test_email_matcher_preserves_full_body(self) -> None:
        """EmailMatcher.match() must not drop full_body from evidence."""
        from gnc_enrich.matching.email_matcher import EmailMatcher
        from gnc_enrich.email.index import EmailIndexRepository

        repo = EmailIndexRepository()
        ev = EmailEvidence(
            evidence_id="e1", message_id="<m@x>", sender="shop@co.uk",
            subject="Order", sent_at=datetime(2025, 1, 15, tzinfo=timezone.utc),
            body_snippet="snippet", full_body="This is the FULL email body content.",
            parsed_amounts=[Decimal("25.00")],
        )
        repo._entries = [ev]
        matcher = EmailMatcher(repo, date_window_days=7, amount_tolerance=0.50)

        tx = _make_tx(amount=Decimal("25.00"))
        results = matcher.match(tx)
        assert len(results) >= 1
        assert results[0].full_body == "This is the FULL email body content."

    def test_review_action_validates(self) -> None:
        """ReviewAction.validate rejects invalid actions."""
        from gnc_enrich.domain.models import ReviewAction
        assert ReviewAction.validate("approve") == "approve"
        assert ReviewAction.validate("skip") == "skip"
        import pytest
        with pytest.raises(ValueError, match="Invalid review action"):
            ReviewAction.validate("invalid_action")

    def test_verbose_flag_accepted(self) -> None:
        """The CLI accepts the -v/--verbose flag."""
        from gnc_enrich.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["-v", "run", "--gnucash-path", "x", "--emails-dir", "e",
                                  "--receipts-dir", "r", "--processed-receipts-dir", "p",
                                  "--state-dir", "s"])
        assert args.verbose is True

    def test_receipt_glob_case_insensitive(self, tmp_path: Path) -> None:
        """ReceiptRepository must find uppercase-extension files."""
        from gnc_enrich.receipt.repository import ReceiptRepository
        from PIL import Image

        img = Image.new("RGB", (10, 10), "white")
        img.save(tmp_path / "receipt.JPG")
        img.save(tmp_path / "receipt2.png")

        repo = ReceiptRepository()
        files = repo.list_unprocessed(tmp_path)
        names = {f.name for f in files}
        assert "receipt.JPG" in names
        assert "receipt2.png" in names
