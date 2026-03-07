from pathlib import Path

import pytest

from gnc_enrich.apply.engine import ApplyEngine
from gnc_enrich.email.index import EmailIndexRepository
from gnc_enrich.email.parser import EmlParser
from gnc_enrich.gnucash.loader import GnuCashLoader, GnuCashWriter
from gnc_enrich.matching.email_matcher import EmailMatcher
from gnc_enrich.matching.receipt_matcher import ReceiptMatcher
from gnc_enrich.ml.predictor import CategoryPredictor, FeedbackTrainer
from gnc_enrich.receipt.ocr import ReceiptOcrEngine
from gnc_enrich.receipt.repository import ReceiptRepository
from gnc_enrich.review.service import ReviewQueueService
from gnc_enrich.review.webapp import ReviewWebApp
from gnc_enrich.services.pipeline import EnrichmentPipeline
from gnc_enrich.state.repository import StateRepository


def test_gnucash_loader_writer_contracts_raise_not_implemented() -> None:
    loader = GnuCashLoader()
    writer = GnuCashWriter()
    with pytest.raises(NotImplementedError):
        loader.load_transactions(Path("book.gnucash"))
    with pytest.raises(NotImplementedError):
        writer.create_backup(Path("book.gnucash"), Path("backups"))
    with pytest.raises(NotImplementedError):
        writer.write_changes(Path("book.gnucash"))


def test_email_contracts_raise_not_implemented() -> None:
    parser = EmlParser()
    repo = EmailIndexRepository()
    with pytest.raises(NotImplementedError):
        parser.parse(Path("mail.eml"))
    with pytest.raises(NotImplementedError):
        repo.build_or_load(Path("emails"), Path("state"))
    with pytest.raises(NotImplementedError):
        repo.search("merchant")


def test_receipt_contracts_raise_not_implemented() -> None:
    ocr = ReceiptOcrEngine()
    repo = ReceiptRepository()
    with pytest.raises(NotImplementedError):
        ocr.parse(Path("receipt.jpg"))
    with pytest.raises(NotImplementedError):
        repo.list_unprocessed(Path("receipts"))
    with pytest.raises(NotImplementedError):
        repo.mark_processed(Path("receipts/a.jpg"), Path("processed"))


def test_matching_contracts_raise_not_implemented() -> None:
    email_matcher = EmailMatcher()
    receipt_matcher = ReceiptMatcher()
    with pytest.raises(NotImplementedError):
        email_matcher.match(None)  # type: ignore[arg-type]
    with pytest.raises(NotImplementedError):
        receipt_matcher.match(None)  # type: ignore[arg-type]


def test_ml_contracts_raise_not_implemented() -> None:
    predictor = CategoryPredictor()
    trainer = FeedbackTrainer()
    with pytest.raises(NotImplementedError):
        predictor.propose(None, [], None)  # type: ignore[arg-type]
    with pytest.raises(NotImplementedError):
        trainer.record_feedback(None, accepted=True)  # type: ignore[arg-type]


def test_review_contracts_raise_not_implemented() -> None:
    service = ReviewQueueService()
    app = ReviewWebApp()
    with pytest.raises(NotImplementedError):
        service.next_proposal()
    with pytest.raises(NotImplementedError):
        service.submit_decision(None)  # type: ignore[arg-type]
    with pytest.raises(NotImplementedError):
        app.run("127.0.0.1", 7860)


def test_state_apply_pipeline_contracts_raise_not_implemented() -> None:
    state = StateRepository()
    apply_engine = ApplyEngine()
    pipeline = EnrichmentPipeline()
    with pytest.raises(NotImplementedError):
        state.save_proposals([])
    with pytest.raises(NotImplementedError):
        state.load_proposals()
    with pytest.raises(NotImplementedError):
        state.save_decision(None)  # type: ignore[arg-type]
    with pytest.raises(NotImplementedError):
        state.load_skipped_ids()
    with pytest.raises(NotImplementedError):
        apply_engine.generate_dry_run_report(Path("state"))
    with pytest.raises(NotImplementedError):
        apply_engine.apply(Path("state"))
    with pytest.raises(NotImplementedError):
        apply_engine.rollback(Path("state"), "journal-1")
    with pytest.raises(NotImplementedError):
        pipeline.run(None)  # type: ignore[arg-type]
    with pytest.raises(NotImplementedError):
        pipeline.build_proposals(None)  # type: ignore[arg-type]
