"""End-to-end integration tests: run -> review -> apply."""

import gzip
import json
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from gnc_enrich.apply.engine import ApplyEngine
from gnc_enrich.config import ApplyConfig, ReviewConfig, RunConfig
from gnc_enrich.domain.models import ReviewDecision, Split
from gnc_enrich.gnucash.loader import GnuCashLoader
from gnc_enrich.review.service import ReviewQueueService
from gnc_enrich.review.webapp import create_app
from gnc_enrich.services.pipeline import EnrichmentPipeline
from gnc_enrich.state.repository import StateRepository
from tests.conftest import SAMPLE_GNUCASH_XML

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _setup_full(tmp_path: Path) -> dict[str, Path]:
    gnucash = tmp_path / "book.gnucash"
    with gzip.open(gnucash, "wb") as f:
        f.write(SAMPLE_GNUCASH_XML.encode("utf-8"))

    return {
        "gnucash": gnucash,
        "emails": FIXTURES_DIR / "emails",
        "receipts": tmp_path / "receipts",
        "processed": tmp_path / "processed",
        "state": tmp_path / "state",
    }


def test_full_run_review_apply_cycle(tmp_path: Path) -> None:
    """Test the complete pipeline: run -> review decisions -> apply."""
    dirs = _setup_full(tmp_path)
    (dirs["receipts"]).mkdir(exist_ok=True)

    # --- STEP 1: Run pipeline ---
    config = RunConfig(
        gnucash_path=dirs["gnucash"],
        emails_dir=dirs["emails"],
        receipts_dir=dirs["receipts"],
        processed_receipts_dir=dirs["processed"],
        state_dir=dirs["state"],
    )
    result = EnrichmentPipeline().run(config)
    assert result.proposal_count == 4

    # --- STEP 2: Review via service ---
    state = StateRepository(dirs["state"])
    svc = ReviewQueueService(state)
    assert svc.total_count == 4
    assert svc.pending_count == 4

    prop1 = svc.next_proposal()
    assert prop1 is not None
    svc.submit_decision(ReviewDecision(
        tx_id=prop1.tx_id,
        action="approve",
        final_description=prop1.suggested_description,
        final_splits=prop1.suggested_splits,
    ))

    prop2 = svc.next_proposal()
    assert prop2 is not None
    svc.submit_decision(ReviewDecision(
        tx_id=prop2.tx_id,
        action="edit",
        final_description="Edited Description 01/02/2025",
        final_splits=[Split(account_path="Expenses:Utilities", amount=Decimal("9.50"))],
    ))

    prop3 = svc.next_proposal()
    assert prop3 is not None
    svc.submit_decision(ReviewDecision(
        tx_id=prop3.tx_id,
        action="skip",
        final_description="",
        final_splits=[],
        reviewer_note="Low confidence",
    ))

    assert svc.pending_count == 1

    # --- STEP 3: Dry-run report ---
    engine = ApplyEngine()
    report = engine.generate_dry_run_report(dirs["state"])
    assert report.exists()
    report_text = report.read_text()
    assert "approve" in report_text.lower()
    assert "skip" in report_text.lower()

    # --- STEP 4: Apply ---
    engine.apply(dirs["state"])

    loader = GnuCashLoader()
    txs = loader.load_transactions(dirs["gnucash"])
    tx_map = {t.tx_id: t for t in txs}

    assert tx_map[prop1.tx_id].description == prop1.suggested_description
    assert tx_map[prop2.tx_id].description == "Edited Description 01/02/2025"

    # --- STEP 5: Verify audit log ---
    audit = state.load_audit_log()
    assert len(audit) == 2
    audit_tx_ids = {a.tx_id for a in audit}
    assert prop1.tx_id in audit_tx_ids
    assert prop2.tx_id in audit_tx_ids
    assert prop3.tx_id not in audit_tx_ids

    # --- STEP 6: Verify backup ---
    backup_dir = dirs["state"] / "backups"
    assert backup_dir.exists()
    backups = list(backup_dir.glob("*.gnucash"))
    assert len(backups) >= 1

    # --- STEP 7: Verify skipped ---
    skipped = state.load_skipped_ids()
    assert prop3.tx_id in skipped

    # --- STEP 8: Rollback ---
    original_desc = tx_map[prop1.tx_id].description
    engine.rollback(dirs["state"])

    restored = GnuCashLoader()
    rtxs = restored.load_transactions(dirs["gnucash"])
    rtx_map = {t.tx_id: t for t in rtxs}
    assert rtx_map[prop1.tx_id].description != original_desc


def test_full_cycle_via_flask_client(tmp_path: Path) -> None:
    """Test the review web app with Flask test client in the full flow."""
    dirs = _setup_full(tmp_path)
    (dirs["receipts"]).mkdir(exist_ok=True)

    config = RunConfig(
        gnucash_path=dirs["gnucash"],
        emails_dir=dirs["emails"],
        receipts_dir=dirs["receipts"],
        processed_receipts_dir=dirs["processed"],
        state_dir=dirs["state"],
    )
    EnrichmentPipeline().run(config)

    state = StateRepository(dirs["state"])
    svc = ReviewQueueService(state)
    app = create_app(svc)
    app.config["TESTING"] = True
    client = app.test_client()

    resp = client.get("/")
    assert resp.status_code == 302

    proposals = svc.all_proposals()
    for p in proposals:
        client.post(f"/review/{p.proposal_id}/decide", data={
            "action": "approve",
            "description": p.suggested_description,
            "split_path": p.suggested_splits[0].account_path,
            "split_amount": str(p.suggested_splits[0].amount),
        })

    resp = client.get("/")
    assert resp.status_code == 200
    assert b"All Proposals Reviewed" in resp.data

    decisions = state.load_decisions()
    assert len(decisions) == 4
    assert all(d.action == "approve" for d in decisions)


def test_cli_run_command_dispatches(tmp_path: Path) -> None:
    """Verify CLI run command builds config and dispatches to pipeline."""
    dirs = _setup_full(tmp_path)
    (dirs["receipts"]).mkdir(exist_ok=True)

    from gnc_enrich.cli import main

    rc = main([
        "run",
        "--gnucash-path", str(dirs["gnucash"]),
        "--emails-dir", str(dirs["emails"]),
        "--receipts-dir", str(dirs["receipts"]),
        "--processed-receipts-dir", str(dirs["processed"]),
        "--state-dir", str(dirs["state"]),
    ])
    assert rc == 0

    state = StateRepository(dirs["state"])
    proposals = state.load_proposals()
    assert len(proposals) == 4
