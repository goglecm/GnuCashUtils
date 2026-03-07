"""High-level orchestration for candidate selection and proposal generation."""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass
from datetime import timedelta

from gnc_enrich.config import LlmMode, RunConfig
from gnc_enrich.domain.models import Proposal, ReceiptEvidence
from gnc_enrich.email.index import EmailIndexRepository
from gnc_enrich.gnucash.loader import GnuCashLoader
from gnc_enrich.matching.email_matcher import EmailMatcher
from gnc_enrich.matching.receipt_matcher import ReceiptMatcher
from gnc_enrich.ml.predictor import CategoryPredictor
from gnc_enrich.receipt.ocr import ReceiptOcrEngine
from gnc_enrich.receipt.repository import ReceiptRepository
from gnc_enrich.state.repository import StateRepository

logger = logging.getLogger(__name__)


def _test_llm_connection(endpoint: str, model_name: str, api_key: str) -> bool:
    """Send a minimal chat completion request to verify the LLM endpoint works. Returns True if OK."""
    if not endpoint or not model_name:
        logger.warning("LLM endpoint or model missing; skipping connection test")
        return False
    try:
        import requests
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": "Reply with the word OK."}],
            "max_tokens": 10,
        }
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        resp = requests.post(endpoint, json=payload, headers=headers, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        if "choices" in data and len(data["choices"]) > 0:
            logger.info("LLM connection OK (endpoint responded successfully)")
            return True
        logger.warning("LLM connection test: unexpected response shape")
        return False
    except Exception as e:
        logger.warning("LLM connection failed: %s", e, exc_info=True)
        return False


@dataclass(slots=True)
class PipelineResult:
    """Summary of a pipeline run."""
    proposal_count: int
    skipped_count: int


class EnrichmentPipeline:
    """Coordinates loading, matching, inference, and proposal persistence."""

    def run(self, config: RunConfig) -> PipelineResult:
        if config.llm.mode != LlmMode.DISABLED:
            logger.info(
                "LLM enabled: mode=%s endpoint=%s model=%s",
                config.llm.mode.value,
                config.llm.endpoint or "(none)",
                config.llm.model_name or "(none)",
            )
            if not _test_llm_connection(
                config.llm.endpoint, config.llm.model_name, config.llm.api_key
            ):
                logger.warning("LLM connection test failed; pipeline will continue but LLM calls may fail")
        else:
            logger.info("LLM disabled; using ML/heuristics and OCR only")
        proposals = self.build_proposals(config)

        state = StateRepository(config.state_dir)
        state.save_proposals(proposals)
        state.save_metadata("run_config", {
            "gnucash_path": str(config.gnucash_path),
            "emails_dir": str(config.emails_dir),
            "receipts_dir": str(config.receipts_dir),
            "processed_receipts_dir": str(config.processed_receipts_dir),
            "llm_mode": config.llm.mode.value,
            "llm_endpoint": config.llm.endpoint,
            "llm_model": config.llm.model_name,
        })

        skipped = state.load_skipped_ids()
        return PipelineResult(
            proposal_count=len(proposals),
            skipped_count=len(skipped),
        )

    def build_proposals(self, config: RunConfig) -> list[Proposal]:
        """Load data, match evidence, and generate proposals for all candidates."""
        loader = GnuCashLoader()
        all_txs = loader.load_transactions(config.gnucash_path)
        logger.info("Loaded %d transactions from %s", len(all_txs), config.gnucash_path)

        state = StateRepository(config.state_dir)
        skipped_ids = state.load_skipped_ids()

        candidates = loader.filter_candidates(
            all_txs,
            include_skipped=config.include_skipped,
            skipped_ids=skipped_ids,
        )
        logger.info("Filtered to %d candidate transactions (Unspecified/Imbalance-GBP only)", len(candidates))

        min_email_date = None
        if candidates and config.emails_dir.exists():
            earliest_candidate_date = min(tx.posted_date for tx in candidates)
            latest_candidate_date = max(tx.posted_date for tx in candidates)
            min_email_date = earliest_candidate_date - timedelta(days=config.date_window_days)
            logger.info(
                "Indexing emails from %s onwards (Unspecified/Imbalance-GBP tx range %s to %s, window %d days before earliest)",
                min_email_date,
                earliest_candidate_date,
                latest_candidate_date,
                config.date_window_days,
            )

        email_index = EmailIndexRepository()
        if config.emails_dir.exists():
            email_index.build_or_load(
                config.emails_dir,
                config.state_dir,
                min_date=min_email_date,
            )
        else:
            logger.info("Emails directory does not exist: %s — skipping email indexing", config.emails_dir)

        receipt_repo = ReceiptRepository()
        receipt_files = receipt_repo.list_unprocessed(config.receipts_dir) if config.receipts_dir.exists() else []
        ocr_engine = ReceiptOcrEngine(llm_config=config.llm)

        receipt_evidences: list[ReceiptEvidence] = []
        for rpath in receipt_files:
            try:
                ev = ocr_engine.parse(rpath)
                receipt_evidences.append(ev)
            except Exception:
                logger.warning("Failed to OCR %s, skipping", rpath, exc_info=True)

        logger.info("OCR processed %d receipts", len(receipt_evidences))

        email_matcher = EmailMatcher(
            email_index,
            date_window_days=config.date_window_days,
            amount_tolerance=config.amount_tolerance,
        )
        receipt_matcher = ReceiptMatcher(
            receipt_evidences,
            amount_tolerance=config.amount_tolerance,
        )

        candidate_ids = {tx.tx_id for tx in candidates}
        historical = [tx for tx in all_txs if tx.tx_id not in candidate_ids]
        predictor = CategoryPredictor(
            historical_transactions=historical,
            llm_config=config.llm,
        )

        proposals: list[Proposal] = []
        for i, tx in enumerate(candidates, 1):
            logger.debug(
                "Processing candidate %d/%d: tx=%s amount=£%s",
                i, len(candidates), tx.tx_id, tx.amount,
            )
            matched_emails = email_matcher.match(tx)
            matched_receipt = receipt_matcher.match(tx)
            logger.debug(
                "  Matched %d emails, receipt=%s",
                len(matched_emails),
                matched_receipt.evidence_id if matched_receipt else "none",
            )
            proposal = predictor.propose(tx, matched_emails, matched_receipt)
            proposal = dataclasses.replace(proposal, is_transfer=getattr(tx, "is_unsettled_transfer", False))
            logger.debug(
                "  Proposal: category=%s confidence=%.2f transfer=%s",
                proposal.suggested_splits[0].account_path if proposal.suggested_splits else "?",
                proposal.confidence,
                proposal.is_transfer,
            )
            proposals.append(proposal)

        logger.info("Generated %d proposals", len(proposals))
        return proposals
