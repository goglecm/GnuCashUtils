"""System-level tests exercising the full pipeline with realistic synthetic data.

These tests model a real-world scenario: Proton Mail-style email directory with
multiple subdirectories, realistic GnuCash XML, programmatic receipt images,
and the complete run -> review -> apply cycle.
"""

from __future__ import annotations

import gzip
from decimal import Decimal
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from gnc_enrich.config import RunConfig
from gnc_enrich.domain.models import ReviewDecision, Split
from gnc_enrich.gnucash.loader import GnuCashLoader
from gnc_enrich.review.service import ReviewQueueService
from gnc_enrich.services.pipeline import EnrichmentPipeline
from gnc_enrich.state.repository import StateRepository
from gnc_enrich.apply.engine import ApplyEngine

EMAILS_FIXTURE_DIR = Path(__file__).parent / "fixtures" / "emails"


SYSTEM_GNUCASH_XML = """\
<?xml version="1.0" encoding="utf-8" ?>
<gnc-v2
     xmlns:gnc="http://www.gnucash.org/XML/gnc"
     xmlns:act="http://www.gnucash.org/XML/act"
     xmlns:book="http://www.gnucash.org/XML/book"
     xmlns:cd="http://www.gnucash.org/XML/cd"
     xmlns:cmdty="http://www.gnucash.org/XML/cmdty"
     xmlns:slot="http://www.gnucash.org/XML/slot"
     xmlns:split="http://www.gnucash.org/XML/split"
     xmlns:trn="http://www.gnucash.org/XML/trn"
     xmlns:ts="http://www.gnucash.org/XML/ts">
<gnc:count-data cd:type="book">1</gnc:count-data>
<gnc:book version="2.0.0">
<book:id type="guid">sysbook01</book:id>

<gnc:commodity version="2.0.0">
  <cmdty:space>CURRENCY</cmdty:space>
  <cmdty:id>GBP</cmdty:id>
</gnc:commodity>

<gnc:account version="2.0.0">
  <act:name>Root Account</act:name>
  <act:id type="guid">root01</act:id>
  <act:type>ROOT</act:type>
</gnc:account>

<gnc:account version="2.0.0">
  <act:name>Current Account</act:name>
  <act:id type="guid">acct_current</act:id>
  <act:type>BANK</act:type>
  <act:commodity><cmdty:space>CURRENCY</cmdty:space><cmdty:id>GBP</cmdty:id></act:commodity>
  <act:parent type="guid">root01</act:parent>
</gnc:account>

<gnc:account version="2.0.0">
  <act:name>Expenses</act:name>
  <act:id type="guid">acct_expenses</act:id>
  <act:type>EXPENSE</act:type>
  <act:commodity><cmdty:space>CURRENCY</cmdty:space><cmdty:id>GBP</cmdty:id></act:commodity>
  <act:parent type="guid">root01</act:parent>
</gnc:account>

<gnc:account version="2.0.0">
  <act:name>Groceries</act:name>
  <act:id type="guid">acct_groceries</act:id>
  <act:type>EXPENSE</act:type>
  <act:commodity><cmdty:space>CURRENCY</cmdty:space><cmdty:id>GBP</cmdty:id></act:commodity>
  <act:parent type="guid">acct_expenses</act:parent>
</gnc:account>

<gnc:account version="2.0.0">
  <act:name>Entertainment</act:name>
  <act:id type="guid">acct_entertainment</act:id>
  <act:type>EXPENSE</act:type>
  <act:commodity><cmdty:space>CURRENCY</cmdty:space><cmdty:id>GBP</cmdty:id></act:commodity>
  <act:parent type="guid">acct_expenses</act:parent>
</gnc:account>

<gnc:account version="2.0.0">
  <act:name>Utilities</act:name>
  <act:id type="guid">acct_utilities</act:id>
  <act:type>EXPENSE</act:type>
  <act:commodity><cmdty:space>CURRENCY</cmdty:space><cmdty:id>GBP</cmdty:id></act:commodity>
  <act:parent type="guid">acct_expenses</act:parent>
</gnc:account>

<gnc:account version="2.0.0">
  <act:name>Unspecified</act:name>
  <act:id type="guid">acct_unspec</act:id>
  <act:type>EXPENSE</act:type>
  <act:commodity><cmdty:space>CURRENCY</cmdty:space><cmdty:id>GBP</cmdty:id></act:commodity>
  <act:parent type="guid">root01</act:parent>
</gnc:account>

<gnc:account version="2.0.0">
  <act:name>Imbalance-GBP</act:name>
  <act:id type="guid">acct_imbalance</act:id>
  <act:type>EXPENSE</act:type>
  <act:commodity><cmdty:space>CURRENCY</cmdty:space><cmdty:id>GBP</cmdty:id></act:commodity>
  <act:parent type="guid">root01</act:parent>
</gnc:account>

<!-- Historical: Tesco groceries (training data) -->
<gnc:transaction version="2.0.0">
  <trn:id type="guid">tx_hist_groc1</trn:id>
  <trn:currency><cmdty:space>CURRENCY</cmdty:space><cmdty:id>GBP</cmdty:id></trn:currency>
  <trn:date-posted><ts:date>2023-02-10 00:00:00 +0000</ts:date></trn:date-posted>
  <trn:description>Tesco Weekly Shop</trn:description>
  <trn:splits>
    <trn:split><split:id type="guid">sp_hg1a</split:id><split:value>-4790/100</split:value><split:account type="guid">acct_current</split:account></trn:split>
    <trn:split><split:id type="guid">sp_hg1b</split:id><split:value>4790/100</split:value><split:account type="guid">acct_groceries</split:account></trn:split>
  </trn:splits>
</gnc:transaction>

<!-- Historical: Netflix -->
<gnc:transaction version="2.0.0">
  <trn:id type="guid">tx_hist_ent1</trn:id>
  <trn:currency><cmdty:space>CURRENCY</cmdty:space><cmdty:id>GBP</cmdty:id></trn:currency>
  <trn:date-posted><ts:date>2023-02-01 00:00:00 +0000</ts:date></trn:date-posted>
  <trn:description>Netflix Subscription</trn:description>
  <trn:splits>
    <trn:split><split:id type="guid">sp_he1a</split:id><split:value>-1099/100</split:value><split:account type="guid">acct_current</split:account></trn:split>
    <trn:split><split:id type="guid">sp_he1b</split:id><split:value>1099/100</split:value><split:account type="guid">acct_entertainment</split:account></trn:split>
  </trn:splits>
</gnc:transaction>

<!-- Historical: BT broadband -->
<gnc:transaction version="2.0.0">
  <trn:id type="guid">tx_hist_util1</trn:id>
  <trn:currency><cmdty:space>CURRENCY</cmdty:space><cmdty:id>GBP</cmdty:id></trn:currency>
  <trn:date-posted><ts:date>2023-02-11 00:00:00 +0000</ts:date></trn:date-posted>
  <trn:description>BT Broadband Direct Debit</trn:description>
  <trn:splits>
    <trn:split><split:id type="guid">sp_hu1a</split:id><split:value>-5400/100</split:value><split:account type="guid">acct_current</split:account></trn:split>
    <trn:split><split:id type="guid">sp_hu1b</split:id><split:value>5400/100</split:value><split:account type="guid">acct_utilities</split:account></trn:split>
  </trn:splits>
</gnc:transaction>

<!-- CANDIDATE 1: Tesco card payment (matches email) -->
<gnc:transaction version="2.0.0">
  <trn:id type="guid">tx_cand_tesco</trn:id>
  <trn:currency><cmdty:space>CURRENCY</cmdty:space><cmdty:id>GBP</cmdty:id></trn:currency>
  <trn:date-posted><ts:date>2023-03-15 00:00:00 +0000</ts:date></trn:date-posted>
  <trn:description>Card Payment</trn:description>
  <trn:splits>
    <trn:split><split:id type="guid">sp_ct1a</split:id><split:value>-4790/100</split:value><split:account type="guid">acct_current</split:account></trn:split>
    <trn:split><split:id type="guid">sp_ct1b</split:id><split:value>4790/100</split:value><split:account type="guid">acct_unspec</split:account></trn:split>
  </trn:splits>
</gnc:transaction>

<!-- CANDIDATE 2: Netflix subscription (matches email) -->
<gnc:transaction version="2.0.0">
  <trn:id type="guid">tx_cand_netflix</trn:id>
  <trn:currency><cmdty:space>CURRENCY</cmdty:space><cmdty:id>GBP</cmdty:id></trn:currency>
  <trn:date-posted><ts:date>2023-03-01 00:00:00 +0000</ts:date></trn:date-posted>
  <trn:description>Direct Debit</trn:description>
  <trn:splits>
    <trn:split><split:id type="guid">sp_cn1a</split:id><split:value>-1099/100</split:value><split:account type="guid">acct_current</split:account></trn:split>
    <trn:split><split:id type="guid">sp_cn1b</split:id><split:value>1099/100</split:value><split:account type="guid">acct_imbalance</split:account></trn:split>
  </trn:splits>
</gnc:transaction>

<!-- CANDIDATE 3: BT broadband (matches email) -->
<gnc:transaction version="2.0.0">
  <trn:id type="guid">tx_cand_bt</trn:id>
  <trn:currency><cmdty:space>CURRENCY</cmdty:space><cmdty:id>GBP</cmdty:id></trn:currency>
  <trn:date-posted><ts:date>2023-03-11 00:00:00 +0000</ts:date></trn:date-posted>
  <trn:description>Direct Debit BT</trn:description>
  <trn:splits>
    <trn:split><split:id type="guid">sp_cb1a</split:id><split:value>-5400/100</split:value><split:account type="guid">acct_current</split:account></trn:split>
    <trn:split><split:id type="guid">sp_cb1b</split:id><split:value>5400/100</split:value><split:account type="guid">acct_unspec</split:account></trn:split>
  </trn:splits>
</gnc:transaction>

<!-- CANDIDATE 4: Unknown (no matching email - tests graceful handling) -->
<gnc:transaction version="2.0.0">
  <trn:id type="guid">tx_cand_unknown</trn:id>
  <trn:currency><cmdty:space>CURRENCY</cmdty:space><cmdty:id>GBP</cmdty:id></trn:currency>
  <trn:date-posted><ts:date>2023-03-20 00:00:00 +0000</ts:date></trn:date-posted>
  <trn:description>POS Purchase</trn:description>
  <trn:splits>
    <trn:split><split:id type="guid">sp_cu1a</split:id><split:value>-1599/100</split:value><split:account type="guid">acct_current</split:account></trn:split>
    <trn:split><split:id type="guid">sp_cu1b</split:id><split:value>1599/100</split:value><split:account type="guid">acct_unspec</split:account></trn:split>
  </trn:splits>
</gnc:transaction>

</gnc:book>
</gnc-v2>
"""


def _create_receipt_image(path: Path, lines: list[str]) -> None:
    """Generate a synthetic receipt image with text lines."""
    img = Image.new("RGB", (400, 50 * len(lines) + 60), "white")
    draw = ImageDraw.Draw(img)
    y = 20
    for line in lines:
        draw.text((20, y), line, fill="black")
        y += 40
    img.save(path)


@pytest.fixture()
def system_env(tmp_path: Path) -> dict[str, Path]:
    """Set up a complete environment with GnuCash file, emails, receipts, and state."""
    gnucash_path = tmp_path / "books.gnucash"
    with gzip.open(gnucash_path, "wb") as f:
        f.write(SYSTEM_GNUCASH_XML.encode("utf-8"))

    state_dir = tmp_path / "state"
    state_dir.mkdir()

    receipts_dir = tmp_path / "receipts"
    receipts_dir.mkdir()
    _create_receipt_image(
        receipts_dir / "tesco_receipt.jpg",
        ["TESCO STORES LTD", "Bread     1.20", "Milk      1.50", "Chicken  5.99", "Total: 47.90"],
    )

    processed_dir = tmp_path / "processed_receipts"
    processed_dir.mkdir()

    return {
        "gnucash_path": gnucash_path,
        "emails_dir": EMAILS_FIXTURE_DIR,
        "receipts_dir": receipts_dir,
        "processed_dir": processed_dir,
        "state_dir": state_dir,
    }


class TestSystemPipeline:
    """Full pipeline run with realistic data from multiple email subdirectories."""

    def test_pipeline_indexes_emails_from_subdirectories(self, system_env: dict[str, Path]) -> None:
        """Emails spread across bank/, amazon/, subscriptions/, utilities/ subdirs are all found."""
        config = RunConfig(
            gnucash_path=system_env["gnucash_path"],
            emails_dir=system_env["emails_dir"],
            receipts_dir=system_env["receipts_dir"],
            processed_receipts_dir=system_env["processed_dir"],
            state_dir=system_env["state_dir"],
        )
        result = EnrichmentPipeline().run(config)
        assert result.proposal_count == 4

        index_path = system_env["state_dir"] / "email_index.jsonl"
        assert index_path.exists()
        lines = [
            line
            for line in index_path.read_text().splitlines()
            if line.strip() and '"_schema_version"' not in line
        ]
        assert len(lines) >= 8

    def test_pipeline_produces_proposals_for_all_candidates(
        self, system_env: dict[str, Path]
    ) -> None:
        config = RunConfig(
            gnucash_path=system_env["gnucash_path"],
            emails_dir=system_env["emails_dir"],
            receipts_dir=system_env["receipts_dir"],
            processed_receipts_dir=system_env["processed_dir"],
            state_dir=system_env["state_dir"],
        )
        EnrichmentPipeline().run(config)

        state = StateRepository(system_env["state_dir"])
        proposals = state.load_proposals()
        assert len(proposals) == 4
        tx_ids = {p.tx_id for p in proposals}
        assert tx_ids == {"tx_cand_tesco", "tx_cand_netflix", "tx_cand_bt", "tx_cand_unknown"}

    def test_pipeline_matches_emails_to_transactions(self, system_env: dict[str, Path]) -> None:
        """Email evidence should be attached to proposals whose amounts/dates match."""
        config = RunConfig(
            gnucash_path=system_env["gnucash_path"],
            emails_dir=system_env["emails_dir"],
            receipts_dir=system_env["receipts_dir"],
            processed_receipts_dir=system_env["processed_dir"],
            state_dir=system_env["state_dir"],
        )
        EnrichmentPipeline().run(config)

        state = StateRepository(system_env["state_dir"])
        proposals = state.load_proposals()
        tesco_prop = next(p for p in proposals if p.tx_id == "tx_cand_tesco")
        assert tesco_prop.evidence is not None
        assert len(tesco_prop.evidence.emails) > 0

    def test_pipeline_persists_run_metadata(self, system_env: dict[str, Path]) -> None:
        config = RunConfig(
            gnucash_path=system_env["gnucash_path"],
            emails_dir=system_env["emails_dir"],
            receipts_dir=system_env["receipts_dir"],
            processed_receipts_dir=system_env["processed_dir"],
            state_dir=system_env["state_dir"],
        )
        EnrichmentPipeline().run(config)

        state = StateRepository(system_env["state_dir"])
        meta = state.load_metadata("run_config")
        assert meta is not None
        assert "gnucash_path" in meta

    def test_pipeline_with_no_receipts(self, system_env: dict[str, Path]) -> None:
        """Pipeline works when receipts directory is empty."""
        for f in system_env["receipts_dir"].iterdir():
            f.unlink()

        config = RunConfig(
            gnucash_path=system_env["gnucash_path"],
            emails_dir=system_env["emails_dir"],
            receipts_dir=system_env["receipts_dir"],
            processed_receipts_dir=system_env["processed_dir"],
            state_dir=system_env["state_dir"],
        )
        result = EnrichmentPipeline().run(config)
        assert result.proposal_count == 4

    def test_pipeline_with_nonexistent_emails_dir(self, system_env: dict[str, Path]) -> None:
        """Pipeline works when emails directory doesn't exist."""
        config = RunConfig(
            gnucash_path=system_env["gnucash_path"],
            emails_dir=Path("/nonexistent/emails"),
            receipts_dir=system_env["receipts_dir"],
            processed_receipts_dir=system_env["processed_dir"],
            state_dir=system_env["state_dir"],
        )
        result = EnrichmentPipeline().run(config)
        assert result.proposal_count == 4


class TestSystemReviewCycle:
    """Review queue operations with realistic proposal data."""

    def _run_pipeline(self, env: dict[str, Path]) -> None:
        config = RunConfig(
            gnucash_path=env["gnucash_path"],
            emails_dir=env["emails_dir"],
            receipts_dir=env["receipts_dir"],
            processed_receipts_dir=env["processed_dir"],
            state_dir=env["state_dir"],
        )
        EnrichmentPipeline().run(config)

    def test_review_queue_has_all_proposals(self, system_env: dict[str, Path]) -> None:
        self._run_pipeline(system_env)
        state = StateRepository(system_env["state_dir"])
        svc = ReviewQueueService(state)

        assert svc.total_count == 4
        assert svc.pending_count == 4
        assert svc.decided_count == 0

    def test_approve_skip_and_edit_cycle(self, system_env: dict[str, Path]) -> None:
        """Approve one, skip one, edit one, leave one pending."""
        self._run_pipeline(system_env)
        state = StateRepository(system_env["state_dir"])
        svc = ReviewQueueService(state)

        proposals = svc.all_proposals()

        svc.submit_decision(
            ReviewDecision(
                tx_id=proposals[0].tx_id,
                action="approve",
                final_description=proposals[0].suggested_description,
                final_splits=proposals[0].suggested_splits,
            )
        )
        svc.submit_decision(
            ReviewDecision(
                tx_id=proposals[1].tx_id,
                action="skip",
                final_description="",
                final_splits=[],
                reviewer_note="Not sure about this one",
            )
        )
        svc.submit_decision(
            ReviewDecision(
                tx_id=proposals[2].tx_id,
                action="edit",
                final_description="BT Broadband March 2023",
                final_splits=[Split(account_path="Expenses:Utilities", amount=Decimal("54.00"))],
            )
        )

        assert svc.decided_count == 3
        assert svc.pending_count == 1

        decisions = state.load_decisions()
        assert len(decisions) == 3
        actions = {d.action for d in decisions}
        assert actions == {"approve", "skip", "edit"}

    def test_feedback_recorded_on_decision(self, system_env: dict[str, Path]) -> None:
        """FeedbackTrainer must record feedback when decisions are submitted."""
        self._run_pipeline(system_env)
        state = StateRepository(system_env["state_dir"])
        svc = ReviewQueueService(state)
        p = svc.all_proposals()[0]

        svc.submit_decision(
            ReviewDecision(
                tx_id=p.tx_id,
                action="approve",
                final_description=p.suggested_description,
                final_splits=p.suggested_splits,
            )
        )

        feedback = state.load_feedback()
        assert len(feedback) >= 1
        assert feedback[0]["tx_id"] == p.tx_id
        assert feedback[0]["accepted"] is True

    def test_skipped_persists_across_runs(self, system_env: dict[str, Path]) -> None:
        """Skipped transactions should be excluded on the next pipeline run."""
        self._run_pipeline(system_env)
        state = StateRepository(system_env["state_dir"])
        svc = ReviewQueueService(state)
        p = svc.all_proposals()[0]

        svc.submit_decision(
            ReviewDecision(
                tx_id=p.tx_id,
                action="skip",
                final_description="",
                final_splits=[],
            )
        )

        config = RunConfig(
            gnucash_path=system_env["gnucash_path"],
            emails_dir=system_env["emails_dir"],
            receipts_dir=system_env["receipts_dir"],
            processed_receipts_dir=system_env["processed_dir"],
            state_dir=system_env["state_dir"],
            include_skipped=False,
        )
        result = EnrichmentPipeline().run(config)
        assert result.proposal_count == 3


class TestSystemApply:
    """Apply engine with realistic data."""

    def _run_and_approve_all(self, env: dict[str, Path]) -> None:
        config = RunConfig(
            gnucash_path=env["gnucash_path"],
            emails_dir=env["emails_dir"],
            receipts_dir=env["receipts_dir"],
            processed_receipts_dir=env["processed_dir"],
            state_dir=env["state_dir"],
        )
        EnrichmentPipeline().run(config)

        state = StateRepository(env["state_dir"])
        svc = ReviewQueueService(state)
        for p in svc.all_proposals():
            svc.submit_decision(
                ReviewDecision(
                    tx_id=p.tx_id,
                    action="approve",
                    final_description=p.suggested_description,
                    final_splits=p.suggested_splits,
                )
            )

    def test_dry_run_report(self, system_env: dict[str, Path]) -> None:
        self._run_and_approve_all(system_env)
        engine = ApplyEngine()
        report_path = engine.generate_dry_run_report(system_env["state_dir"])

        assert report_path.exists()
        content = report_path.read_text()
        assert "DRY-RUN REPORT" in content
        assert "4 approved" in content

    def test_apply_modifies_gnucash_file(self, system_env: dict[str, Path]) -> None:
        self._run_and_approve_all(system_env)
        engine = ApplyEngine()
        engine.apply(system_env["state_dir"])

        backup_dir = system_env["state_dir"] / "backups"
        assert backup_dir.exists()
        assert len(list(backup_dir.iterdir())) == 1

        journal_path = system_env["state_dir"] / "apply_journal.jsonl"
        assert journal_path.exists()

        state = StateRepository(system_env["state_dir"])
        audit = state.load_audit_log()
        assert len(audit) == 4

    def test_apply_creates_backup_then_rollback(self, system_env: dict[str, Path]) -> None:
        self._run_and_approve_all(system_env)
        engine = ApplyEngine()
        engine.apply(system_env["state_dir"])

        loader = GnuCashLoader()
        txs_after = loader.load_transactions(system_env["gnucash_path"])
        assert any(t.tx_id == "tx_cand_tesco" for t in txs_after)

        engine.rollback(system_env["state_dir"])

        loader2 = GnuCashLoader()
        txs_restored = loader2.load_transactions(system_env["gnucash_path"])
        restored_tx = next(t for t in txs_restored if t.tx_id == "tx_cand_tesco")
        assert restored_tx.description == "Card Payment"


class TestSystemWebApp:
    """Flask webapp integration with realistic pipeline data."""

    def _run_pipeline(self, env: dict[str, Path]) -> None:
        config = RunConfig(
            gnucash_path=env["gnucash_path"],
            emails_dir=env["emails_dir"],
            receipts_dir=env["receipts_dir"],
            processed_receipts_dir=env["processed_dir"],
            state_dir=env["state_dir"],
        )
        EnrichmentPipeline().run(config)

    def test_webapp_review_page_renders(self, system_env: dict[str, Path]) -> None:
        self._run_pipeline(system_env)
        state = StateRepository(system_env["state_dir"])
        svc = ReviewQueueService(state)

        from gnc_enrich.review.webapp import create_app

        app = create_app(svc)
        client = app.test_client()

        resp = client.get("/")
        assert resp.status_code == 302

        resp = client.get("/", follow_redirects=True)
        assert resp.status_code == 200
        assert b"proposal" in resp.data.lower() or b"review" in resp.data.lower()

    def test_webapp_queue_page_shows_all_proposals(self, system_env: dict[str, Path]) -> None:
        self._run_pipeline(system_env)
        state = StateRepository(system_env["state_dir"])
        svc = ReviewQueueService(state)

        from gnc_enrich.review.webapp import create_app

        app = create_app(svc)
        client = app.test_client()

        resp = client.get("/queue")
        assert resp.status_code == 200
        html = resp.data.decode()
        assert "15/03/2023" in html
        assert "01/03/2023" in html
        assert "Card Payment" in html
        assert "Direct Debit" in html

    def test_webapp_decide_approve(self, system_env: dict[str, Path]) -> None:
        self._run_pipeline(system_env)
        state = StateRepository(system_env["state_dir"])
        svc = ReviewQueueService(state)

        from gnc_enrich.review.webapp import create_app

        app = create_app(svc)
        client = app.test_client()

        proposal = svc.next_proposal()
        assert proposal is not None

        resp = client.post(
            f"/review/{proposal.proposal_id}/decide",
            data={
                "action": "approve",
                "description": proposal.suggested_description,
                "split_path": [proposal.suggested_splits[0].account_path],
                "split_amount": [str(proposal.suggested_splits[0].amount)],
            },
            follow_redirects=True,
        )
        assert resp.status_code == 200
        assert svc.decided_count == 1

    def test_webapp_done_page_when_all_decided(self, system_env: dict[str, Path]) -> None:
        self._run_pipeline(system_env)
        state = StateRepository(system_env["state_dir"])
        svc = ReviewQueueService(state)

        for p in svc.all_proposals():
            svc.submit_decision(
                ReviewDecision(
                    tx_id=p.tx_id,
                    action="approve",
                    final_description=p.suggested_description,
                    final_splits=p.suggested_splits,
                )
            )

        from gnc_enrich.review.webapp import create_app

        app = create_app(svc)
        client = app.test_client()

        resp = client.get("/", follow_redirects=True)
        assert resp.status_code == 200
        assert b"done" in resp.data.lower() or b"complete" in resp.data.lower()
