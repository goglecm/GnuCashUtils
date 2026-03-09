"""Tests for ReviewQueueService and Flask review web app."""

from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

import pytest

from gnc_enrich.config import ReviewConfig
from gnc_enrich.domain.models import EmailEvidence, EvidencePacket, Proposal, ReviewDecision, Split
from gnc_enrich.review.service import ReviewQueueService
from gnc_enrich.review.webapp import ReviewWebApp, create_app
from gnc_enrich.state.repository import StateRepository


def _seed_proposals(state: StateRepository) -> list[Proposal]:
    proposals = [
        Proposal(
            proposal_id="p1", tx_id="tx1",
            suggested_description="Tesco 15/01/2025",
            suggested_splits=[Split(account_path="Expenses:Food", amount=Decimal("25.00"))],
            confidence=0.85, rationale="ML match",
            tx_date=date(2025, 1, 15), tx_amount=Decimal("25.00"),
            original_description="TESCO STORES", original_splits=[
                Split(account_path="Imbalance-GBP", amount=Decimal("25.00")),
            ],
        ),
        Proposal(
            proposal_id="p2", tx_id="tx2",
            suggested_description="Netflix 20/01/2025",
            suggested_splits=[Split(account_path="Expenses:Entertainment", amount=Decimal("15.99"))],
            confidence=0.7, rationale="Keyword match",
            tx_date=date(2025, 1, 20), tx_amount=Decimal("15.99"),
            original_description="NETFLIX.COM", original_splits=[
                Split(account_path="Imbalance-GBP", amount=Decimal("15.99")),
            ],
        ),
        Proposal(
            proposal_id="p3", tx_id="tx3",
            suggested_description="Unknown 25/01/2025",
            suggested_splits=[Split(account_path="Expenses:Miscellaneous", amount=Decimal("42.00"))],
            confidence=0.3, rationale="Low confidence",
            tx_date=date(2025, 1, 25), tx_amount=Decimal("42.00"),
            original_description="UNKNOWN MERCHANT", original_splits=[
                Split(account_path="Unspecified", amount=Decimal("42.00")),
            ],
        ),
    ]
    state.save_proposals(proposals)
    return proposals


# -- ReviewQueueService -------------------------------------------------------


class TestReviewQueueService:

    def test_next_proposal_returns_first(self, tmp_path: Path) -> None:
        state = StateRepository(tmp_path)
        _seed_proposals(state)
        svc = ReviewQueueService(state)
        prop = svc.next_proposal()
        assert prop is not None
        assert prop.proposal_id == "p1"

    def test_next_proposal_skips_decided(self, tmp_path: Path) -> None:
        state = StateRepository(tmp_path)
        _seed_proposals(state)
        svc = ReviewQueueService(state)

        svc.submit_decision(ReviewDecision(
            tx_id="tx1", action="approve",
            final_description="Tesco 15/01/2025",
            final_splits=[Split(account_path="Expenses:Food", amount=Decimal("25.00"))],
        ))

        prop = svc.next_proposal()
        assert prop is not None
        assert prop.proposal_id == "p2"

    def test_next_proposal_none_when_all_decided(self, tmp_path: Path) -> None:
        state = StateRepository(tmp_path)
        _seed_proposals(state)
        svc = ReviewQueueService(state)

        for tx_id in ["tx1", "tx2", "tx3"]:
            svc.submit_decision(ReviewDecision(
                tx_id=tx_id, action="approve",
                final_description="Done", final_splits=[],
            ))

        assert svc.next_proposal() is None

    def test_counts(self, tmp_path: Path) -> None:
        state = StateRepository(tmp_path)
        _seed_proposals(state)
        svc = ReviewQueueService(state)

        assert svc.total_count == 3
        assert svc.pending_count == 3
        assert svc.decided_count == 0

        svc.submit_decision(ReviewDecision(
            tx_id="tx1", action="skip",
            final_description="", final_splits=[],
        ))
        assert svc.pending_count == 2
        assert svc.decided_count == 1

    def test_skip_creates_skip_record(self, tmp_path: Path) -> None:
        state = StateRepository(tmp_path)
        _seed_proposals(state)
        svc = ReviewQueueService(state)

        svc.submit_decision(ReviewDecision(
            tx_id="tx1", action="skip",
            final_description="", final_splits=[],
            reviewer_note="Not sure",
        ))

        skipped = state.load_skipped_ids()
        assert "tx1" in skipped

    def test_get_proposal_by_id(self, tmp_path: Path) -> None:
        state = StateRepository(tmp_path)
        _seed_proposals(state)
        svc = ReviewQueueService(state)

        prop = svc.get_proposal("p2")
        assert prop is not None
        assert prop.tx_id == "tx2"

    def test_is_decided(self, tmp_path: Path) -> None:
        state = StateRepository(tmp_path)
        _seed_proposals(state)
        svc = ReviewQueueService(state)

        assert svc.is_decided("tx1") is False
        svc.submit_decision(ReviewDecision(
            tx_id="tx1", action="approve",
            final_description="Done", final_splits=[],
        ))
        assert svc.is_decided("tx1") is True

    def test_get_account_paths_empty_without_run(self, tmp_path: Path) -> None:
        """Without a pipeline run, account_paths is empty (dropdown shows only Create new)."""
        state = StateRepository(tmp_path)
        _seed_proposals(state)
        svc = ReviewQueueService(state)
        assert svc.get_account_paths() == []

    def test_get_account_paths_after_run(self, tmp_path: Path) -> None:
        """After pipeline run, account_paths is populated from GnuCash book."""
        state = StateRepository(tmp_path)
        state.save_metadata("account_paths", {"paths": ["Expenses:Food", "Current Account", "Unspecified"]})
        svc = ReviewQueueService(state)
        paths = svc.get_account_paths()
        assert "Expenses:Food" in paths
        assert "Current Account" in paths

    def test_submit_decision_saves_unenriched_when_enrichment_raises(self, tmp_path: Path) -> None:
        """When _enrich_from_approved_evidence raises, submit_decision still saves the decision (non-fatal)."""
        state = StateRepository(tmp_path)
        email_ev = EmailEvidence(
            evidence_id="em1", message_id="<m@x>", sender="shop@co.uk",
            subject="Order", sent_at=datetime(2025, 2, 1, tzinfo=timezone.utc), body_snippet="Order for Widget",
            parsed_amounts=[Decimal("9.99")], full_body="Widget order",
        )
        prop = Proposal(
            proposal_id="p1", tx_id="tx1",
            suggested_description="Payment 01/02/2025", suggested_splits=[],
            confidence=0.8, rationale="ML",
            evidence=EvidencePacket(tx_id="tx1", emails=[email_ev], receipt=None, similar_transactions=[]),
        )
        state.save_proposals([prop])
        svc = ReviewQueueService(state)
        decision = ReviewDecision(
            tx_id="tx1", action="approve",
            final_description="User entered description",
            final_splits=[], decided_at=None,
            approved_email_ids=["em1"], approved_receipt=False,
        )
        with patch.object(svc, "_enrich_from_approved_evidence", side_effect=RuntimeError("Enrichment failed")):
            svc.submit_decision(decision)
        decisions = state.load_decisions()
        assert len(decisions) == 1
        assert decisions[0].final_description == "User entered description"
        assert svc.is_decided("tx1")


# -- Flask web app ------------------------------------------------------------


class TestReviewWebApp:

    @pytest.fixture()
    def client(self, tmp_path: Path):
        state = StateRepository(tmp_path)
        _seed_proposals(state)
        svc = ReviewQueueService(state)
        app = create_app(svc)
        app.config["TESTING"] = True
        return app.test_client()

    def test_index_redirects_to_first_proposal(self, client) -> None:
        resp = client.get("/", follow_redirects=False)
        assert resp.status_code == 302
        assert "/review/p1" in resp.headers["Location"]

    def test_review_page_renders(self, client) -> None:
        resp = client.get("/review/p1")
        assert resp.status_code == 200
        html = resp.data.decode()
        assert "tx1" in html
        assert "Tesco 15/01/2025" in html
        assert "85%" in html

    def test_review_page_shows_transaction_and_ml_llm_sections(self, client) -> None:
        """Review page shows Transaction, ML proposal, LLM section, and Use ML/Use LLM decision buttons per spec."""
        resp = client.get("/review/p1")
        html = resp.data.decode()
        assert "Transaction" in html
        assert "ML proposal" in html
        assert "TESCO STORES" in html
        assert "Tesco 15/01/2025" in html
        assert "Imbalance-GBP" in html
        assert "Expenses:Food" in html
        assert "Use ML proposal" in html
        assert "Use LLM proposal" in html

    def test_review_page_has_select_all_button(self, client) -> None:
        resp = client.get("/review/p1")
        html = resp.data.decode()
        assert "toggle-emails" in html or "Deselect All" in html or "Select All" in html

    def test_approve_decision(self, client) -> None:
        resp = client.post("/review/p1/decide", data={
            "action": "approve",
            "description": "Tesco 15/01/2025",
            "split_path": "Expenses:Food",
            "split_amount": "25.00",
        }, follow_redirects=False)
        assert resp.status_code == 302

    def test_decide_with_create_new_category(self, tmp_path: Path) -> None:
        """Submitting with split_path_new (Create new) uses that as the category."""
        state = StateRepository(tmp_path)
        _seed_proposals(state)
        svc = ReviewQueueService(state)
        app = create_app(svc)
        app.config["TESTING"] = True
        client = app.test_client()

        client.post("/review/p1/decide", data={
            "action": "approve",
            "description": "Tesco 15/01/2025",
            "split_path": "",
            "split_path_new": "Expenses:Groceries",
            "split_amount": "25.00",
        })
        decisions = state.load_decisions()
        assert len(decisions) == 1
        assert decisions[0].final_splits[0].account_path == "Expenses:Groceries"

    def test_skip_decision(self, client) -> None:
        client.post("/review/p1/decide", data={
            "action": "skip",
            "description": "",
            "note": "unclear",
        })
        resp = client.get("/", follow_redirects=False)
        assert resp.status_code == 302
        assert "/review/p2" in resp.headers["Location"]

    def test_all_decided_shows_done(self, client) -> None:
        for pid in ["p1", "p2", "p3"]:
            client.post(f"/review/{pid}/decide", data={
                "action": "approve",
                "description": "Done",
                "split_path": "Expenses:Food",
                "split_amount": "10.00",
            })
        resp = client.get("/")
        assert resp.status_code == 200
        assert b"All Proposals Reviewed" in resp.data

    def test_queue_page(self, client) -> None:
        resp = client.get("/queue")
        assert resp.status_code == 200
        html = resp.data.decode()
        assert "15/01/2025" in html
        assert "20/01/2025" in html
        assert "25/01/2025" in html
        assert "TESCO STORES" in html
        assert "NETFLIX.COM" in html

    def test_queue_sorted_by_date(self, client) -> None:
        resp = client.get("/queue")
        html = resp.data.decode()
        pos_jan15 = html.index("15/01/2025")
        pos_jan20 = html.index("20/01/2025")
        pos_jan25 = html.index("25/01/2025")
        assert pos_jan15 < pos_jan20 < pos_jan25

    def test_queue_sort_handles_none_tx_date(self, tmp_path: Path) -> None:
        state = StateRepository(tmp_path)
        proposals = [
            Proposal(
                proposal_id="pa", tx_id="txa",
                suggested_description="A", suggested_splits=[],
                confidence=0.5, rationale="r",
                tx_date=date(2025, 3, 1), tx_amount=Decimal("10"),
                original_description="A",
            ),
            Proposal(
                proposal_id="pb", tx_id="txb",
                suggested_description="B", suggested_splits=[],
                confidence=0.5, rationale="r",
                tx_date=None, tx_amount=None,
                original_description="B",
            ),
        ]
        state.save_proposals(proposals)
        svc = ReviewQueueService(state)
        app = create_app(svc)
        app.config["TESTING"] = True
        client = app.test_client()
        resp = client.get("/queue")
        assert resp.status_code == 200

    def test_review_page_has_next_prev_links(self, client) -> None:
        resp = client.get("/")
        assert resp.status_code in (200, 302)
        if resp.status_code == 302:
            location = resp.headers["Location"]
            assert "/review/" in location
            proposal_id = location.split("/review/")[-1].strip("/")
            resp2 = client.get(f"/review/{proposal_id}")
            html = resp2.data.decode()
            assert "Previous" in html or "Next" in html

    def test_queue_shows_to_categorise_and_approved_sections(self, client) -> None:
        resp = client.get("/queue")
        assert resp.status_code == 200
        html = resp.data.decode()
        # Queue shows a section for candidates ("To categorise") and one for approved transactions.
        assert "To categorise" in html or "expenses" in html.lower()
        assert "Approved transactions" in html

    def test_llm_button_hidden_when_llm_disabled(self, tmp_path: Path) -> None:
        """When LLM mode is disabled in metadata, the Check with LLM button is not shown."""
        state = StateRepository(tmp_path)
        _seed_proposals(state)
        # No run_config metadata saved → llm_mode defaults to disabled.
        svc = ReviewQueueService(state)
        app = create_app(svc)
        app.config["TESTING"] = True
        client = app.test_client()
        resp = client.get("/review/p1")
        html = resp.data.decode()
        # The LLM section's JavaScript always mentions 'btn-llm-check', so we check
        # specifically for the presence of the button element, not the string.
        assert 'id="btn-llm-check"' not in html

    def test_llm_button_hidden_when_no_description_emails_or_receipt(self, tmp_path: Path) -> None:
        """When there is no description, no emails, and no receipt, the LLM button is hidden even if LLM is enabled."""
        state = StateRepository(tmp_path)
        state.save_metadata("run_config", {"llm_mode": "online", "llm_endpoint": "http://x", "llm_model": "y"})
        prop = Proposal(
            proposal_id="p1",
            tx_id="tx1",
            suggested_description="",
            suggested_splits=[Split(account_path="Expenses:Misc", amount=Decimal("10.00"))],
            confidence=0.5,
            rationale="ML",
            tx_date=date(2025, 1, 15),
            tx_amount=Decimal("10.00"),
            original_description="",
            original_splits=[],
        )
        state.save_proposals([prop])
        svc = ReviewQueueService(state)
        app = create_app(svc)
        app.config["TESTING"] = True
        client = app.test_client()
        resp = client.get("/review/p1")
        html = resp.data.decode()
        assert 'id="btn-llm-check"' not in html

    def test_llm_button_shown_when_llm_enabled_and_description_present(self, tmp_path: Path) -> None:
        """When LLM is enabled and there is a description (or evidence), the Check with LLM button is shown."""
        state = StateRepository(tmp_path)
        state.save_metadata("run_config", {"llm_mode": "online", "llm_endpoint": "http://x", "llm_model": "y"})
        prop = Proposal(
            proposal_id="p1",
            tx_id="tx1",
            suggested_description="ML description",
            suggested_splits=[Split(account_path="Expenses:Misc", amount=Decimal("10.00"))],
            confidence=0.5,
            rationale="ML",
            tx_date=date(2025, 1, 15),
            tx_amount=Decimal("10.00"),
            original_description="Some description",
            original_splits=[],
        )
        state.save_proposals([prop])
        svc = ReviewQueueService(state)
        app = create_app(svc)
        app.config["TESTING"] = True
        client = app.test_client()
        resp = client.get("/review/p1")
        html = resp.data.decode()
        assert 'id="btn-llm-check"' in html

    def test_nonexistent_proposal_redirects(self, client) -> None:
        resp = client.get("/review/nonexistent", follow_redirects=False)
        assert resp.status_code == 302

    def test_review_page_200_with_empty_emails_when_display_build_raises(self, tmp_path: Path) -> None:
        """When get_emails_for_display or category hint raises, review page returns 200 with empty emails (non-fatal)."""
        state = StateRepository(tmp_path)
        email_ev = EmailEvidence(
            evidence_id="em1", message_id="<m@x>", sender="a@b.com",
            subject="Test", sent_at=datetime(2025, 2, 1, tzinfo=timezone.utc), body_snippet="Body",
            parsed_amounts=[Decimal("10.00")], full_body="Full",
        )
        prop = Proposal(
            proposal_id="p1", tx_id="tx1",
            suggested_description="Pay", suggested_splits=[],
            confidence=0.8, rationale="r",
            tx_date=date(2025, 1, 15), tx_amount=Decimal("25.00"),
            original_description="TESCO", original_splits=[],
            evidence=EvidencePacket(tx_id="tx1", emails=[email_ev]),
        )
        state.save_proposals([prop])
        svc = ReviewQueueService(state)
        app = create_app(svc)
        app.config["TESTING"] = True
        client = app.test_client()
        with patch("gnc_enrich.review.webapp.CategoryPredictor.get_emails_for_display", side_effect=ValueError("Display error")):
            resp = client.get("/review/p1")
        assert resp.status_code == 200
        assert b"tx1" in resp.data

    def test_webapp_class_creates_app(self, tmp_path: Path) -> None:
        state = StateRepository(tmp_path)
        _seed_proposals(state)
        svc = ReviewQueueService(state)
        config = ReviewConfig(state_dir=tmp_path)
        webapp = ReviewWebApp(svc, config)
        app = webapp.get_app()
        assert app is not None

    def test_llm_check_endpoint_404_for_missing_proposal(self, client) -> None:
        resp = client.post("/review/nonexistent-id/llm-check")
        assert resp.status_code == 404
        data = resp.get_json()
        assert data is not None and data.get("ok") is False

    def test_llm_check_endpoint_400_when_llm_fails(self, client) -> None:
        """When run_llm_check returns None (LLM disabled or flow failed), API returns 400."""
        resp = client.post("/review/p1/llm-check")
        assert resp.status_code == 400
        data = resp.get_json()
        assert data is not None and data.get("ok") is False

    def test_llm_check_returns_200_with_null_extraction_when_serialize_raises(self, tmp_path: Path) -> None:
        """When _serialize_extraction_for_json raises, llm-check returns 200 with extraction: null (non-fatal)."""
        state = StateRepository(tmp_path)
        state.save_metadata("account_paths", {"paths": ["Expenses:Food"]})
        state.save_metadata("run_config", {"llm_mode": "online", "llm_endpoint": "http://x", "llm_model": "y"})
        prop = Proposal(
            proposal_id="p1", tx_id="tx1",
            suggested_description="ML", suggested_splits=[Split(account_path="Expenses:Food", amount=Decimal("25.00"))],
            confidence=0.8, rationale="ML",
            tx_date=date(2025, 1, 15), tx_amount=Decimal("25.00"),
            original_description="TESCO", original_splits=[],
            evidence=EvidencePacket(tx_id="tx1", emails=[]),
        )
        state.save_proposals([prop])
        svc = ReviewQueueService(state)
        with patch("gnc_enrich.review.service.CategoryPredictor") as MockPredictor:
            mock_instance = MockPredictor.return_value
            mock_instance.run_llm_check.return_value = {
                "extraction": {"seller_name": "Tesco", "items": []},
                "category": "Expenses:Food",
                "description": "Tesco",
                "confidence": 0.9,
            }
            app = create_app(svc)
            app.config["TESTING"] = True
            client = app.test_client()
            with patch("gnc_enrich.review.webapp._serialize_extraction_for_json", side_effect=TypeError("Bad type")):
                resp = client.post("/review/p1/llm-check")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data is not None and data.get("ok") is True
        assert data.get("extraction") is None
        assert data.get("category") == "Expenses:Food"

    def test_run_llm_check_updates_proposal_llm_fields_preserves_ml(self, tmp_path: Path) -> None:
        """run_llm_check updates proposal with llm_* and extraction_result; does not overwrite suggested_description/suggested_splits."""
        state = StateRepository(tmp_path)
        state.save_metadata("account_paths", {"paths": ["Expenses:Food", "Expenses:Miscellaneous"]})
        state.save_metadata("run_config", {"llm_mode": "online", "llm_endpoint": "http://x", "llm_model": "y"})
        prop = Proposal(
            proposal_id="p1", tx_id="tx1",
            suggested_description="ML description",
            suggested_splits=[Split(account_path="Expenses:Miscellaneous", amount=Decimal("25.00"))],
            confidence=0.5, rationale="ML",
            tx_date=date(2025, 1, 15), tx_amount=Decimal("25.00"),
            original_description="TESCO", original_splits=[],
            evidence=EvidencePacket(tx_id="tx1", emails=[]),
        )
        state.save_proposals([prop])
        svc = ReviewQueueService(state)
        with patch("gnc_enrich.review.service.CategoryPredictor") as MockPredictor:
            mock_instance = MockPredictor.return_value
            mock_instance.run_llm_check.return_value = {
                "extraction": {"seller_name": "Tesco", "items": []},
                "category": "Expenses:Food",
                "description": "LLM description",
                "confidence": 0.8,
            }
            result = svc.run_llm_check("p1")
        assert result is not None
        assert result["category"] == "Expenses:Food"
        updated = svc.get_proposal("p1")
        assert updated is not None
        assert updated.llm_category == "Expenses:Food"
        assert updated.llm_description == "LLM description"
        assert updated.llm_confidence == 0.8
        assert updated.extraction_result is not None
        assert updated.suggested_description == "ML description"
        assert updated.suggested_splits[0].account_path == "Expenses:Miscellaneous"

    def test_run_config_loads_llm_warmup_on_start(self, tmp_path: Path) -> None:
        """When run_config has llm_warmup_on_start True, ReviewQueueService builds LlmConfig with warmup_on_start True."""
        state = StateRepository(tmp_path)
        state.save_metadata("account_paths", {"paths": ["Expenses:Food"]})
        state.save_metadata("run_config", {
            "llm_mode": "online",
            "llm_endpoint": "http://x",
            "llm_model": "y",
            "llm_warmup_on_start": True,
        })
        prop = Proposal(
            proposal_id="p1", tx_id="tx1",
            suggested_description="ML", suggested_splits=[Split(account_path="Expenses:Food", amount=Decimal("25.00"))],
            confidence=0.5, rationale="ML",
            tx_date=date(2025, 1, 15), tx_amount=Decimal("25.00"),
            original_description="TESCO", original_splits=[],
            evidence=EvidencePacket(tx_id="tx1", emails=[]),
        )
        state.save_proposals([prop])
        svc = ReviewQueueService(state)
        with patch("gnc_enrich.review.service.CategoryPredictor") as MockPredictor:
            mock_instance = MockPredictor.return_value
            mock_instance.run_llm_check.return_value = {
                "extraction": {}, "category": "Expenses:Food", "description": "OK", "confidence": 0.8,
            }
            svc.run_llm_check("p1")
        call_kwargs = MockPredictor.call_args[1]
        assert call_kwargs["llm_config"].warmup_on_start is True

    def test_run_llm_check_returns_none_does_not_update_proposal(self, tmp_path: Path) -> None:
        """When run_llm_check returns None (flow failed or disabled), proposal is not updated and service returns None."""
        state = StateRepository(tmp_path)
        state.save_metadata("account_paths", {"paths": ["Expenses:Food"]})
        state.save_metadata("run_config", {"llm_mode": "online", "llm_endpoint": "http://x", "llm_model": "y"})
        prop = Proposal(
            proposal_id="p1", tx_id="tx1",
            suggested_description="ML only",
            suggested_splits=[Split(account_path="Expenses:Miscellaneous", amount=Decimal("25.00"))],
            confidence=0.5, rationale="ML",
            tx_date=date(2025, 1, 15), tx_amount=Decimal("25.00"),
            original_description="TESCO", original_splits=[],
            evidence=EvidencePacket(tx_id="tx1", emails=[]),
        )
        state.save_proposals([prop])
        svc = ReviewQueueService(state)
        with patch("gnc_enrich.review.service.CategoryPredictor") as MockPredictor:
            mock_instance = MockPredictor.return_value
            mock_instance.run_llm_check.return_value = None
            result = svc.run_llm_check("p1")
        assert result is None
        updated = svc.get_proposal("p1")
        assert updated is not None
        assert updated.llm_category is None
        assert updated.llm_description is None
        assert updated.extraction_result is None
        assert updated.suggested_description == "ML only"

    def test_llm_check_api_returns_200_with_serialized_extraction(self, tmp_path: Path) -> None:
        """POST llm-check returns 200 with extraction serialized for JSON (Decimal -> str in items)."""
        state = StateRepository(tmp_path)
        state.save_metadata("account_paths", {"paths": ["Expenses:Food"]})
        state.save_metadata("run_config", {"llm_mode": "online", "llm_endpoint": "http://x", "llm_model": "y"})
        prop = Proposal(
            proposal_id="p1", tx_id="tx1",
            suggested_description="ML",
            suggested_splits=[Split(account_path="Expenses:Food", amount=Decimal("25.00"))],
            confidence=0.8, rationale="ML",
            tx_date=date(2025, 1, 15), tx_amount=Decimal("25.00"),
            original_description="TESCO", original_splits=[],
            evidence=EvidencePacket(tx_id="tx1", emails=[]),
        )
        state.save_proposals([prop])
        svc = ReviewQueueService(state)
        with patch("gnc_enrich.review.service.CategoryPredictor") as MockPredictor:
            mock_instance = MockPredictor.return_value
            mock_instance.run_llm_check.return_value = {
                "extraction": {"seller_name": "Tesco", "items": [{"description": "Bread", "amount": Decimal("3.50")}]},
                "category": "Expenses:Food",
                "description": "Tesco Bread",
                "confidence": 0.9,
            }
            app = create_app(svc)
            app.config["TESTING"] = True
            client = app.test_client()
            resp = client.post("/review/p1/llm-check")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data is not None and data.get("ok") is True
        assert data.get("category") == "Expenses:Food"
        ext = data.get("extraction")
        assert ext is not None
        assert ext.get("seller_name") == "Tesco"
        items = ext.get("items") or []
        assert len(items) == 1
        assert items[0].get("amount") == "3.50"

    def test_run_llm_check_exception_returns_none(self, tmp_path: Path) -> None:
        """When run_llm_check raises (e.g. network error), service returns None and proposal is not updated."""
        state = StateRepository(tmp_path)
        state.save_metadata("account_paths", {"paths": ["Expenses:Food"]})
        state.save_metadata("run_config", {"llm_mode": "online", "llm_endpoint": "http://x", "llm_model": "y"})
        prop = Proposal(
            proposal_id="p1", tx_id="tx1",
            suggested_description="ML",
            suggested_splits=[Split(account_path="Expenses:Food", amount=Decimal("25.00"))],
            confidence=0.8, rationale="ML",
            tx_date=date(2025, 1, 15), tx_amount=Decimal("25.00"),
            original_description="TESCO", original_splits=[],
            evidence=EvidencePacket(tx_id="tx1", emails=[]),
        )
        state.save_proposals([prop])
        svc = ReviewQueueService(state)
        with patch("gnc_enrich.review.service.CategoryPredictor") as MockPredictor:
            mock_instance = MockPredictor.return_value
            mock_instance.run_llm_check.side_effect = RuntimeError("Network error")
            result = svc.run_llm_check("p1")
        assert result is None
        updated = svc.get_proposal("p1")
        assert updated is not None and updated.llm_category is None

    def test_decide_returns_503_when_save_fails(self, tmp_path: Path) -> None:
        """When submit_decision raises, decide route returns 503 with error message."""
        state = StateRepository(tmp_path)
        _seed_proposals(state)
        svc = ReviewQueueService(state)
        app = create_app(svc)
        app.config["TESTING"] = True
        client = app.test_client()
        with patch.object(svc, "submit_decision", side_effect=OSError("Disk full")):
            resp = client.post("/review/p1/decide", data={
                "action": "approve",
                "description": "Tesco 15/01/2025",
                "split_path": "Expenses:Food",
                "split_amount": "25.00",
            })
        assert resp.status_code == 503
        assert b"Failed to save" in resp.data

    def test_decide_invalid_action_returns_400(self, client) -> None:
        """When POST action is invalid, decide returns 400 with error message (non-fatal)."""
        resp = client.post("/review/p1/decide", data={
            "action": "invalid_action",
            "description": "Tesco",
            "split_path": "Expenses:Food",
            "split_amount": "25.00",
        })
        assert resp.status_code == 400
        assert b"Invalid review action" in resp.data or b"invalid_action" in resp.data
