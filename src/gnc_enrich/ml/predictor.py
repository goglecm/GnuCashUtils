"""ML category prediction using TF-IDF + classifier with optional LLM rationale."""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

import requests

from gnc_enrich.config import LlmConfig, LlmMode
from gnc_enrich.domain.models import (
    EmailEvidence,
    EvidencePacket,
    Proposal,
    ReceiptEvidence,
    Split,
    Transaction,
)

logger = logging.getLogger(__name__)


class CategoryPredictor:
    """Predicts category and description using TF-IDF + classifier trained on history."""

    def __init__(
        self,
        historical_transactions: list[Transaction] | None = None,
        llm_config: LlmConfig | None = None,
    ) -> None:
        self._llm_config = llm_config or LlmConfig()
        self._classes: list[str] = []
        self._trained = False
        self._vectorizer: Any = None
        self._classifier: Any = None

        if historical_transactions:
            self._train(historical_transactions)

    def _train(self, transactions: list[Transaction]) -> None:
        categorized = [
            tx for tx in transactions
            if tx.original_category
            and tx.original_category not in ("Unspecified", "Imbalance-GBP")
        ]
        if len(categorized) < 2:
            logger.warning("Not enough categorised history to train (%d)", len(categorized))
            return

        categories = list({tx.original_category for tx in categorized})
        if len(categories) < 2:
            logger.warning("Only one category in history; skipping training")
            return

        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import SGDClassifier
        from sklearn.pipeline import make_pipeline

        texts = [self._featurize_text(tx) for tx in categorized]
        labels = [tx.original_category for tx in categorized]

        self._vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        self._classifier = SGDClassifier(loss="modified_huber", random_state=42, max_iter=1000)

        pipeline = make_pipeline(self._vectorizer, self._classifier)
        pipeline.fit(texts, labels)

        self._classes = list(self._classifier.classes_)
        self._trained = True
        logger.info("Trained on %d transactions across %d categories", len(categorized), len(self._classes))

    def _featurize_text(
        self,
        tx: Transaction,
        emails: list[EmailEvidence] | None = None,
        receipt: ReceiptEvidence | None = None,
    ) -> str:
        parts = [tx.description, tx.account_name]

        if emails:
            for em in emails[:3]:
                parts.extend([em.sender, em.subject, em.body_snippet[:200]])

        if receipt:
            parts.append(receipt.ocr_text[:300])

        return " ".join(p for p in parts if p)

    def propose(
        self,
        tx: Transaction,
        emails: list[EmailEvidence],
        receipt: ReceiptEvidence | None,
    ) -> Proposal:
        text = self._featurize_text(tx, emails, receipt)
        suggested_category = ""
        confidence = 0.0
        rationale_parts: list[str] = []

        if self._trained and self._vectorizer and self._classifier:
            X = self._vectorizer.transform([text])
            proba = self._classifier.predict_proba(X)[0]
            best_idx = proba.argmax()
            suggested_category = self._classes[best_idx]
            confidence = float(proba[best_idx])
            rationale_parts.append(
                f"ML classifier predicted '{suggested_category}' with {confidence:.0%} confidence"
            )
        else:
            suggested_category = self._fallback_category(tx, emails)
            confidence = 0.3
            rationale_parts.append("No trained model; using heuristic fallback")

        if emails:
            top = emails[0]
            rationale_parts.append(f"Top email match: {top.sender} - {top.subject}")

        if receipt and receipt.parsed_total is not None:
            rationale_parts.append(f"Receipt total: £{receipt.parsed_total}")

        if (
            self._llm_config.mode != LlmMode.DISABLED
            and confidence < 0.6
        ):
            llm_suggestion = self._query_llm(tx, emails, receipt, suggested_category)
            if llm_suggestion:
                rationale_parts.append(f"LLM suggestion: {llm_suggestion}")

        suggested_desc = self._build_description(tx, emails, receipt)
        suggested_splits = [
            Split(account_path=suggested_category, amount=tx.amount),
        ]

        return Proposal(
            proposal_id=uuid.uuid4().hex[:12],
            tx_id=tx.tx_id,
            suggested_description=suggested_desc,
            suggested_splits=suggested_splits,
            confidence=round(confidence, 4),
            rationale="; ".join(rationale_parts),
            evidence=EvidencePacket(
                tx_id=tx.tx_id,
                emails=emails,
                receipt=receipt,
            ),
        )

    def _fallback_category(
        self, tx: Transaction, emails: list[EmailEvidence]
    ) -> str:
        text = tx.description.lower()
        if emails:
            text += " " + " ".join(e.sender.lower() + " " + e.subject.lower() for e in emails[:3])

        keywords = {
            "Expenses:Food": ["grocery", "tesco", "sainsbury", "asda", "lidl", "aldi", "food", "restaurant", "cafe"],
            "Expenses:Transport": ["fuel", "petrol", "uber", "train", "bus", "parking", "transport"],
            "Expenses:Entertainment": ["netflix", "spotify", "cinema", "theatre", "amazon prime"],
            "Expenses:Utilities": ["electric", "gas", "water", "broadband", "internet", "phone", "mobile"],
        }
        for category, kws in keywords.items():
            if any(kw in text for kw in kws):
                return category
        return "Expenses:Miscellaneous"

    def _build_description(
        self,
        tx: Transaction,
        emails: list[EmailEvidence],
        receipt: ReceiptEvidence | None,
    ) -> str:
        date_str = tx.posted_date.strftime("%d/%m/%Y")
        base = tx.description.strip()

        merchant = ""
        if emails:
            sender = emails[0].sender
            if "@" in sender:
                domain = sender.split("@")[1].split(".")[0]
                merchant = domain.capitalize()

        if merchant and merchant.lower() not in base.lower():
            return f"{base} - {merchant} {date_str}" if base else f"{merchant} {date_str}"
        return f"{base} {date_str}" if base else date_str

    def _query_llm(
        self,
        tx: Transaction,
        emails: list[EmailEvidence],
        receipt: ReceiptEvidence | None,
        current_suggestion: str,
    ) -> str | None:
        try:
            evidence_text = f"Transaction: {tx.description}, £{tx.amount}, {tx.posted_date}"
            if emails:
                evidence_text += f"\nTop email from: {emails[0].sender}, subject: {emails[0].subject}"
            if receipt:
                evidence_text += f"\nReceipt text excerpt: {receipt.ocr_text[:200]}"

            payload = {
                "model": self._llm_config.model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            f"Given this transaction evidence, suggest the most appropriate expense "
                            f"category. Current ML suggestion: {current_suggestion}\n\n{evidence_text}\n\n"
                            f"Return only the category path (e.g. Expenses:Food)."
                        ),
                    }
                ],
                "temperature": self._llm_config.temperature,
                "max_tokens": 100,
            }
            resp = requests.post(self._llm_config.endpoint, json=payload, timeout=30)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception:
            logger.warning("LLM query failed", exc_info=True)
            return None


class FeedbackTrainer:
    """Captures user-approved outcomes for future model improvement."""

    def __init__(self, state_dir: Path | None = None) -> None:
        self._state_dir = state_dir

    def record_feedback(
        self,
        proposal: Proposal,
        accepted: bool,
        note: str = "",
    ) -> None:
        feedback = {
            "proposal_id": proposal.proposal_id,
            "tx_id": proposal.tx_id,
            "suggested_category": proposal.suggested_splits[0].account_path if proposal.suggested_splits else "",
            "confidence": proposal.confidence,
            "accepted": accepted,
            "note": note,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        if self._state_dir:
            from gnc_enrich.state.repository import StateRepository
            repo = StateRepository(self._state_dir)
            repo.append_feedback(feedback)

        logger.info(
            "Feedback recorded: proposal=%s accepted=%s", proposal.proposal_id, accepted
        )
