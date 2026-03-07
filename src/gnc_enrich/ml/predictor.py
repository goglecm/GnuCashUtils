"""ML category prediction using TF-IDF + classifier with optional LLM rationale."""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

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
        """Generate a Proposal with category, description, and evidence for a transaction."""
        if getattr(tx, "is_transfer", False):
            return self._propose_transfer(tx, emails, receipt)

        text = self._featurize_text(tx, emails, receipt)
        suggested_category = ""
        confidence = 0.0
        rationale_parts: list[str] = []

        breakdown: list[str] = []
        refund_category = self._check_refund_match(tx)
        if refund_category:
            suggested_category = refund_category
            confidence = 0.65
            rationale_parts.append(
                f"Refund detected (negative amount); matched to '{refund_category}'"
            )
            breakdown.append("Refund detection: matched to original category (confidence 65%)")
        elif self._trained and self._vectorizer and self._classifier:
            X = self._vectorizer.transform([text])
            proba = self._classifier.predict_proba(X)[0]
            best_idx = proba.argmax()
            suggested_category = self._classes[best_idx]
            confidence = float(proba[best_idx])
            rationale_parts.append(
                f"ML classifier predicted '{suggested_category}' with {confidence:.0%} confidence"
            )
            breakdown.append(f"ML classifier: {suggested_category} ({confidence:.0%})")
        else:
            suggested_category = self._fallback_category(tx, emails)
            confidence = 0.3
            rationale_parts.append("No trained model; using heuristic fallback")
            breakdown.append("Keyword heuristic fallback (confidence 30%)")

        if emails:
            top = emails[0]
            rationale_parts.append(f"Top email match: {(top.sender or '')[:50]} - {(top.subject or '')[:50]}")
            breakdown.append(f"Email evidence: {(top.sender or '')[:50]} — {(top.subject or '')[:50]}")

        if receipt and receipt.parsed_total is not None:
            rationale_parts.append(f"Receipt total: £{receipt.parsed_total}")
            breakdown.append(f"Receipt total: £{receipt.parsed_total}")

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
            tx_date=tx.posted_date,
            tx_amount=tx.amount,
            original_description=tx.description,
            original_splits=list(tx.splits),
            confidence_breakdown=breakdown,
        )

    def _propose_transfer(
        self,
        tx: Transaction,
        emails: list[EmailEvidence],
        receipt: ReceiptEvidence | None,
    ) -> Proposal:
        """Proposal for a transfer: keep original splits; only description may be enriched.

        Transfers between own accounts are not recategorised; the apply engine
        will only update the description when the user approves.
        """
        suggested_desc = self._build_description(tx, emails, receipt)
        return Proposal(
            proposal_id=uuid.uuid4().hex[:12],
            tx_id=tx.tx_id,
            suggested_description=suggested_desc,
            suggested_splits=list(tx.splits),
            confidence=1.0,
            rationale="Transfer between own accounts; no recategorisation needed.",
            evidence=EvidencePacket(
                tx_id=tx.tx_id,
                emails=emails,
                receipt=receipt,
            ),
            tx_date=tx.posted_date,
            tx_amount=tx.amount,
            original_description=tx.description,
            original_splits=list(tx.splits),
            confidence_breakdown=["Transfer: no split change (keep original)"],
        )

    def _check_refund_match(self, tx: Transaction) -> str | None:
        """If tx looks like a refund (incoming bank transfer), find the original category.

        Since ``tx.amount`` is always the absolute sum of positive splits,
        we detect refunds by checking whether the bank-side split
        (``tx.account_name``) carries a positive amount (money flowing in).
        """
        if not self._trained:
            return None
        if not tx.account_name:
            return None
        is_refund = any(
            sp.amount > 0
            for sp in tx.splits
            if sp.account_path == tx.account_name
        )
        if not is_refund:
            return None
        text = self._featurize_text(tx)
        X = self._vectorizer.transform([text])
        proba = self._classifier.predict_proba(X)[0]
        best_idx = proba.argmax()
        if proba[best_idx] >= 0.4:
            return self._classes[best_idx]
        return None

    def _fallback_category(
        self, tx: Transaction, emails: list[EmailEvidence]
    ) -> str:
        text = tx.description.lower()
        if emails:
            text += " " + " ".join(e.sender.lower() + " " + e.subject.lower() for e in emails[:3])
        return self._category_from_text(text)

    def _category_from_text(self, text: str) -> str:
        """Suggest a category from free text using keyword heuristics."""
        if not text:
            return ""
        text_lower = text.lower()
        keywords = {
            "Expenses:Food": ["grocery", "tesco", "sainsbury", "asda", "lidl", "aldi", "food", "restaurant", "cafe"],
            "Expenses:Transport": ["fuel", "petrol", "uber", "train", "bus", "parking", "transport"],
            "Expenses:Entertainment": ["netflix", "spotify", "cinema", "theatre", "amazon prime"],
            "Expenses:Utilities": ["electric", "gas", "water", "broadband", "internet", "phone", "mobile"],
        }
        for category, kws in keywords.items():
            if any(kw in text_lower for kw in kws):
                return category
        return "Expenses:Miscellaneous"

    def suggest_category_from_email(self, sender: str, subject: str, body: str) -> str:
        """Suggest a category from email sender, subject, and body (e.g. for UI hint)."""
        text = f"{(sender or '')} {(subject or '')} {(body or '')}"
        return self._category_from_text(text)

    def _build_description(
        self,
        tx: Transaction,
        emails: list[EmailEvidence],
        receipt: ReceiptEvidence | None,
    ) -> str:
        """Build the suggested description from transaction data and evidence."""
        date_str = tx.posted_date.strftime("%d/%m/%Y")
        base = tx.description.strip()

        merchant = ""
        if emails:
            sender = emails[0].sender
            if "@" in sender:
                domain = sender.split("@")[1].split(".")[0]
                merchant = domain.capitalize()

        if merchant and merchant.lower() not in base.lower():
            desc = f"{base} - {merchant} {date_str}" if base else f"{merchant} {date_str}"
        else:
            desc = f"{base} {date_str}" if base else date_str

        return desc

    def enrich_description_from_evidence(
        self,
        base_description: str,
        approved_emails: list[EmailEvidence],
        approved_receipt: ReceiptEvidence | None,
    ) -> str:
        """Enrich description with content from user-approved evidence."""
        parts = [base_description]

        if approved_emails:
            email_detail = self._extract_purchase_detail(approved_emails)
            if email_detail:
                parts.append(email_detail)

        if approved_receipt:
            receipt_detail = self._extract_receipt_detail(approved_receipt)
            if receipt_detail:
                parts.append(receipt_detail)

        return " | ".join(p for p in parts if p)

    def _extract_purchase_detail(self, emails: list[EmailEvidence]) -> str:
        """Pull purchase details from email body content."""
        details: list[str] = []
        for em in emails[:2]:
            body = em.full_body or em.body_snippet
            if not body:
                continue
            snippet = body[:300].strip().replace("\n", " ").replace("\r", "")
            if em.subject:
                details.append(f"{em.subject}: {snippet}")
            else:
                details.append(snippet)
        return "; ".join(details)[:200] if details else ""

    def _extract_receipt_detail(self, receipt: ReceiptEvidence) -> str:
        """Summarise receipt line items for the description."""
        if not receipt.line_items:
            if receipt.parsed_total is not None:
                return f"Receipt total £{receipt.parsed_total}"
            return ""
        item_strs = [f"{it.description} £{it.amount}" for it in receipt.line_items[:5]]
        summary = ", ".join(item_strs)
        if receipt.parsed_total is not None:
            summary += f" (total £{receipt.parsed_total})"
        return summary

    def describe_terse_items(
        self, receipt: ReceiptEvidence
    ) -> list[str]:
        """Use LLM to expand terse/abbreviated receipt item names."""
        if (
            not self._llm_config
            or self._llm_config.mode == LlmMode.DISABLED
            or not receipt.line_items
        ):
            return [it.description for it in receipt.line_items]

        terse_names = [it.description for it in receipt.line_items]
        try:
            import requests

            payload = {
                "model": self._llm_config.model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "These are abbreviated item names from a receipt. "
                            "For each one, provide a clear, expanded description of what the item likely is. "
                            "Return a JSON array of strings, one per input item.\n\n"
                            + json.dumps(terse_names)
                        ),
                    }
                ],
                "temperature": self._llm_config.temperature,
                "max_tokens": self._llm_config.max_tokens,
            }
            headers = {}
            if self._llm_config.api_key:
                headers["Authorization"] = f"Bearer {self._llm_config.api_key}"
            resp = requests.post(self._llm_config.endpoint, json=payload, headers=headers, timeout=30)
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"].strip()
            expanded = json.loads(content)
            if isinstance(expanded, list) and len(expanded) == len(terse_names):
                return expanded
        except Exception:
            logger.warning("LLM item description failed", exc_info=True)

        return terse_names

    def _query_llm(
        self,
        tx: Transaction,
        emails: list[EmailEvidence],
        receipt: ReceiptEvidence | None,
        current_suggestion: str,
    ) -> str | None:
        """Query the LLM for a category suggestion when ML confidence is low."""
        try:
            import requests

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
            headers = {}
            if self._llm_config.api_key:
                headers["Authorization"] = f"Bearer {self._llm_config.api_key}"
            resp = requests.post(self._llm_config.endpoint, json=payload, headers=headers, timeout=30)
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
        """Record a user feedback event to the state store."""
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
