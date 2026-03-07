"""Tests for ReviewQueueService and Flask review web app."""

from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from gnc_enrich.config import ReviewConfig
from gnc_enrich.domain.models import Proposal, ReviewDecision, Split
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

    def test_review_page_shows_before_after(self, client) -> None:
        resp = client.get("/review/p1")
        html = resp.data.decode()
        assert "ORIGINAL" in html
        assert "PROPOSED" in html
        assert "TESCO STORES" in html
        assert "Tesco 15/01/2025" in html
        assert "Imbalance-GBP" in html
        assert "Expenses:Food" in html

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

    def test_queue_shows_expense_transfer_approved_sections(self, client) -> None:
        resp = client.get("/queue")
        assert resp.status_code == 200
        html = resp.data.decode()
        assert "To categorise" in html or "expenses" in html.lower()
        assert "transfer" in html.lower()
        assert "Approved" in html

    def test_nonexistent_proposal_redirects(self, client) -> None:
        resp = client.get("/review/nonexistent", follow_redirects=False)
        assert resp.status_code == 302

    def test_webapp_class_creates_app(self, tmp_path: Path) -> None:
        state = StateRepository(tmp_path)
        _seed_proposals(state)
        svc = ReviewQueueService(state)
        config = ReviewConfig(state_dir=tmp_path)
        webapp = ReviewWebApp(svc, config)
        app = webapp.get_app()
        assert app is not None
