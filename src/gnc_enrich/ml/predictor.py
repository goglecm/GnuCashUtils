"""ML category prediction using TF-IDF + classifier with optional LLM rationale."""

from __future__ import annotations

import difflib
import json
import logging
import uuid
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

from gnc_enrich.config import LlmConfig, LlmMode
from gnc_enrich.llm.client import LlmClient
from gnc_enrich.prompts import get_prompts_dir, load_template, render
from gnc_enrich.domain.models import (
    EmailEvidence,
    EvidencePacket,
    Proposal,
    ReceiptEvidence,
    Split,
    Transaction,
)

logger = logging.getLogger(__name__)

# Limit snippet length in DEBUG logs to reduce PII exposure (transaction text, emails, LLM I/O).
_LOG_SNIPPET_MAX = 200


def _truncate_for_log(s: str, max_len: int = _LOG_SNIPPET_MAX) -> str:
    """Return a short snippet for logging; avoids logging full PII."""
    if not s or len(s) <= max_len:
        return s
    return s[:max_len] + "..."


def _has_name_like_content(s: str) -> bool:
    """True if string contains at least one letter (i.e. looks like a name/description, not just amount/digits)."""
    return bool(s and any(c.isalpha() for c in s))


@dataclass(frozen=True)
class LlmSuggestion:
    """Structured result from LLM when suggesting category and description."""

    category: str
    reason: str
    description: str
    confidence: float = 0.0  # 0-1, from LLM step1 score (1-10) / 10


class CategoryPredictor:
    """Predicts category and description using TF-IDF + classifier trained on history."""

    def __init__(
        self,
        historical_transactions: list[Transaction] | None = None,
        llm_config: LlmConfig | None = None,
        prompts_dir: Path | None = None,
    ) -> None:
        self._llm_config = llm_config or LlmConfig()
        self._prompts_dir = get_prompts_dir(prompts_dir)
        self._classes: list[str] = []
        self._trained = False
        self._vectorizer: Any = None
        self._classifier: Any = None

        if historical_transactions:
            self._train(historical_transactions)

    def _train(self, transactions: list[Transaction]) -> None:
        categorized = [
            tx
            for tx in transactions
            if tx.original_category and tx.original_category not in ("Unspecified", "Imbalance-GBP")
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
        logger.info(
            "Trained on %d transactions across %d categories", len(categorized), len(self._classes)
        )

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
        account_paths: list[str] | None = None,
        skip_llm: bool = False,
    ) -> Proposal:
        """Generate a Proposal with category, description, and evidence for a transaction."""
        text = self._featurize_text(tx, emails, receipt)
        suggested_category = ""
        confidence = 0.0
        rationale_parts: list[str] = []
        llm_description: str | None = None

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
            rationale_parts.append(
                f"Top email match: {(top.sender or '')[:50]} - {(top.subject or '')[:50]}"
            )
            breakdown.append(
                f"Email evidence: {(top.sender or '')[:50]} — {(top.subject or '')[:50]}"
            )

        if receipt and receipt.parsed_total is not None:
            rationale_parts.append(f"Receipt total: £{receipt.parsed_total}")
            breakdown.append(f"Receipt total: £{receipt.parsed_total}")

        if not skip_llm and self._llm_config.mode != LlmMode.DISABLED and confidence < 0.6:
            llm_suggestion = self._query_llm(tx, emails, account_paths or [])
            if llm_suggestion:
                rationale_parts.append(
                    f"LLM suggestion: {llm_suggestion.category}"
                    + (f" — {llm_suggestion.reason}" if llm_suggestion.reason else "")
                )
                breakdown.append(
                    f"LLM picked: {llm_suggestion.category}"
                    + (f". Reason: {llm_suggestion.reason}" if llm_suggestion.reason else "")
                )
                if llm_suggestion.category and (
                    not account_paths or llm_suggestion.category in account_paths
                ):
                    suggested_category = llm_suggestion.category
                if llm_suggestion.description:
                    llm_description = llm_suggestion.description.strip()

        if not suggested_category or suggested_category in ("Unspecified", "Imbalance-GBP"):
            suggested_category = "Expenses:Miscellaneous"
            if not breakdown or "fallback" not in breakdown[-1].lower():
                breakdown.append("Default category (no Unspecified/Imbalance suggested)")
        suggested_desc = (
            llm_description if llm_description else self._build_description(tx, emails, receipt)
        )
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
        is_refund = any(sp.amount > 0 for sp in tx.splits if sp.account_path == tx.account_name)
        if not is_refund:
            return None
        text = self._featurize_text(tx)
        X = self._vectorizer.transform([text])
        proba = self._classifier.predict_proba(X)[0]
        best_idx = proba.argmax()
        if proba[best_idx] >= 0.4:
            return self._classes[best_idx]
        return None

    def _fallback_category(self, tx: Transaction, emails: list[EmailEvidence]) -> str:
        text = tx.description.lower()
        if emails:
            text += " " + " ".join(e.sender.lower() + " " + e.subject.lower() for e in emails[:3])
        return self._category_from_text(text)

    def _category_from_text(self, text: str, account_paths: list[str] | None = None) -> str:
        """Suggest a category from free text using keyword heuristics.
        If account_paths is provided, returns a leaf path from that list under the matched
        heuristic category when possible; otherwise returns the top-level heuristic (e.g. Expenses:Food).
        """
        if not text:
            return ""
        text_lower = text.lower()
        keywords = {
            "Expenses:Food": [
                "grocery",
                "tesco",
                "sainsbury",
                "asda",
                "lidl",
                "aldi",
                "food",
                "restaurant",
                "cafe",
            ],
            "Expenses:Transport": [
                "fuel",
                "petrol",
                "uber",
                "train",
                "bus",
                "parking",
                "transport",
            ],
            "Expenses:Entertainment": ["netflix", "spotify", "cinema", "theatre", "amazon prime"],
            "Expenses:Utilities": [
                "electric",
                "gas",
                "water",
                "broadband",
                "internet",
                "phone",
                "mobile",
            ],
        }
        heuristic = "Expenses:Miscellaneous"
        for category, kws in keywords.items():
            if any(kw in text_lower for kw in kws):
                heuristic = category
                break
        if not account_paths:
            return heuristic
        # Prefer a leaf under the heuristic: path P is a leaf if no other path has P as strict prefix
        under = [p for p in account_paths if p == heuristic or p.startswith(heuristic + ":")]
        if not under:
            return heuristic
        path_set = set(account_paths)
        leaves = [p for p in under if not any(p != q and q.startswith(p + ":") for q in path_set)]
        return leaves[0] if leaves else heuristic

    def suggest_category_from_email(
        self, sender: str, subject: str, body: str, account_paths: list[str] | None = None
    ) -> str:
        """Suggest a category from email sender, subject, and body (e.g. for UI hint).
        If account_paths is provided, may return a leaf from that list under the heuristic category.
        """
        text = f"{(sender or '')} {(subject or '')} {(body or '')}"
        return self._category_from_text(text, account_paths)

    def _build_description(
        self,
        tx: Transaction,
        emails: list[EmailEvidence],
        receipt: ReceiptEvidence | None,
    ) -> str:
        """Build the suggested description from transaction data and evidence (no date appended)."""
        base = tx.description.strip()

        merchant = ""
        if emails:
            sender = emails[0].sender
            if "@" in sender:
                domain = sender.split("@")[1].split(".")[0]
                merchant = domain.capitalize()

        if merchant and merchant.lower() not in base.lower():
            desc = f"{base} - {merchant}" if base else merchant
        else:
            desc = base if base else ""

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

    def describe_terse_items(self, receipt: ReceiptEvidence) -> list[str]:
        """Use LLM to expand terse/abbreviated receipt item names."""
        if (
            not self._llm_config
            or self._llm_config.mode == LlmMode.DISABLED
            or not receipt.line_items
        ):
            return [it.description for it in receipt.line_items]

        terse_names = [it.description for it in receipt.line_items]
        try:
            client = LlmClient(self._llm_config)
            if not client.enabled:
                return terse_names
            user_content = (
                "Expand each receipt item name to a clear description. "
                "Return a JSON array of strings, one per input item.\n\n" + json.dumps(terse_names)
            )
            logger.debug(
                "LLM (receipt items) full request (%d chars):\n%s\n<<< END LLM REQUEST >>>",
                len(user_content),
                _truncate_for_log(user_content),
            )
            data = client.chat(
                messages=[{"role": "user", "content": user_content}],
                max_tokens=self._llm_config.max_tokens,
                temperature=self._llm_config.temperature,
            )
            if not data:
                return terse_names
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            logger.debug(
                "LLM (receipt items) full response:\n%s\n<<< END LLM RESPONSE >>>",
                _truncate_for_log(content),
            )
            expanded = json.loads(content)
            if isinstance(expanded, list) and len(expanded) == len(terse_names):
                return expanded
        except Exception:
            logger.warning("LLM item description failed", exc_info=True)

        return terse_names

    def _web_search_short(self, query: str, min_chars: int = 50, max_chars: int = 200) -> str:
        """Return a single 50-200 char description from the web for the query. Empty if use_web disabled or failed."""
        if not getattr(self._llm_config, "use_web", False):
            return ""
        try:
            try:
                from ddgs import DDGS
            except ImportError:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    from duckduckgo_search import DDGS
            # Prefer English descriptions (e.g. for seller/item context)
            search_query = f"{query} English" if query.strip() else query
            results = list(DDGS().text(search_query, max_results=1))
            if not results:
                return ""
            body = (results[0].get("body") or "").strip()
            if len(body) <= max_chars:
                return body if len(body) >= min_chars else body[:max_chars]
            return body[:max_chars].rsplit(" ", 1)[0] if max_chars > min_chars else body[:max_chars]
        except Exception as e:
            logger.debug("Web search short failed for %r: %s", query[:50], e)
            return ""

    # Top-level account types to include in LLM category list (excludes e.g. Assets, Equity).
    _LLM_CATEGORY_PREFIXES = ("Expenses:", "Income:")
    _EXPENSE_NOTE = "Expense transaction; amount is positive (what merchant received)."
    _CONFIDENCE_THRESHOLD = 6
    # Non-GBP currency codes: paths containing any of these (as a segment or in parentheses in a segment) are excluded.
    _NON_GBP_CURRENCY_CODES = frozenset(
        {
            "USD",
            "EUR",
            "JPY",
            "CHF",
            "AUD",
            "CAD",
            "CNY",
            "INR",
            "NZD",
            "ZAR",
            "SEK",
            "NOK",
            "DKK",
            "CZK",
            "THB",
            "RON",
        }
    )

    @staticmethod
    def _filter_gbp_paths_only(paths: list[str]) -> list[str]:
        """Exclude paths that contain a non-GBP currency (e.g. segment EUR or segment 'Eat in (CZK)')."""

        def has_non_gbp(path: str) -> bool:
            for seg in path.split(":"):
                if seg.upper() in CategoryPredictor._NON_GBP_CURRENCY_CODES:
                    return True
                for code in CategoryPredictor._NON_GBP_CURRENCY_CODES:
                    if f"({code})" in seg.upper():
                        return True
            return False

        return [p for p in paths if not has_non_gbp(p)]

    @staticmethod
    def _get_expenses_first_level(allowed_paths: list[str]) -> list[str]:
        """Return full paths for first-level subcategories of Expenses only (e.g. Expenses:Food, Expenses:Household). Excludes the top-level 'Expenses' itself."""
        result: list[str] = []
        seen: set[str] = set()
        for p in sorted(allowed_paths):
            if p == "Expenses":
                continue
            if p.startswith("Expenses:"):
                parts = p.split(":")
                if len(parts) >= 2:
                    top = "Expenses:" + parts[1]
                    if top not in seen:
                        seen.add(top)
                        result.append(top)
        return sorted(result)

    @staticmethod
    def _format_expenses_first_level_for_prompt(expenses_first: list[str]) -> str:
        """Short names (last segment) semicolon-separated (e.g. Food; Household). Full path required in response."""
        names = [p.split(":")[-1] if ":" in p else p for p in expenses_first]
        return "; ".join(sorted(set(names)))

    @staticmethod
    def _format_step2_subcategories(chosen: str, paths_under_chosen: list[str]) -> str:
        """Format subcategories under chosen only (no Expenses:Food prefix): subcat1:{a; b}; subcat2; ...
        Semicolon-delimited; colon from upper to lower; children in {}."""
        if not paths_under_chosen:
            return ""
        prefix = chosen + ":"
        suffixes: list[str] = []
        for p in paths_under_chosen:
            if p == chosen:
                continue
            if p.startswith(prefix):
                suffixes.append(p[len(prefix) :])
            else:
                suffixes.append(p)
        if not suffixes:
            return ""

        def _tree_from_suffixes(sufs: list[str]) -> dict:
            root: dict = {}
            for s in sufs:
                parts = s.split(":")
                node = root
                for i, seg in enumerate(parts):
                    if i == len(parts) - 1:
                        node[seg] = True
                    else:
                        if seg not in node:
                            node[seg] = {}
                        elif node[seg] is True:
                            node[seg] = {}
                        node = node[seg]
            return root

        def _format_node(node: dict | bool) -> str:
            if node is True:
                return ""
            assert isinstance(node, dict)
            parts = []
            for k in sorted(node.keys()):
                v = node[k]
                if v is True:
                    parts.append(k)
                else:
                    inner = _format_node(v)
                    parts.append(f"{k}:{{{inner}}}")
            return "; ".join(parts)

        tree = _tree_from_suffixes(suffixes)
        return _format_node(tree)

    @staticmethod
    def _get_top_level_categories(allowed_paths: list[str]) -> list[str]:
        """Return all unique first-level subcategories (Expenses:X or Income:Y). No cap."""
        seen: set[str] = set()
        result: list[str] = []
        for p in sorted(allowed_paths):
            if p not in ("Expenses", "Income") and not any(
                p.startswith(prefix) for prefix in CategoryPredictor._LLM_CATEGORY_PREFIXES
            ):
                continue
            if p in ("Expenses", "Income"):
                top = p
            else:
                parts = p.split(":")
                top = parts[0] + ":" + parts[1] if len(parts) >= 2 else p
            if top not in seen:
                seen.add(top)
                result.append(top)
        return sorted(result)

    @staticmethod
    def _get_subcategories(chosen: str, allowed_paths: list[str]) -> list[str]:
        """Return paths that are the chosen category or a subpath of it (chosen, chosen:X, chosen:X:Y, ...)."""
        sub = [p for p in allowed_paths if p == chosen or p.startswith(chosen + ":")]
        return sorted(sub) if sub else [chosen]

    _EMAIL_CONTEXT_CHARS_BEFORE_AMOUNT = 500
    _EMAIL_DEDUPE_SIMILARITY_THRESHOLD = 0.95

    @staticmethod
    def _extract_body_context_around_amount(body: str, amount: Decimal) -> str:
        """Return up to 500 chars before the earliest amount, everything up to and including the last amount; strip everything after the last amount. When amount is 0, return first 500 chars only (avoid matching '0' everywhere)."""
        if not body:
            return body
        body = body.strip()
        if amount is not None and amount == 0:
            return (
                body[: CategoryPredictor._EMAIL_CONTEXT_CHARS_BEFORE_AMOUNT]
                if len(body) > CategoryPredictor._EMAIL_CONTEXT_CHARS_BEFORE_AMOUNT
                else body
            )
        patterns = [
            f"£{amount:.2f}",
            f"£{amount}",
            f"{amount:.2f}",
            str(amount),
        ]
        if amount == amount.to_integral_value():
            patterns.append(f"{int(amount)}")
        first_pos = -1
        last_end = -1
        for p in patterns:
            if not p:
                continue
            pos_first = body.find(p)
            if pos_first != -1:
                if first_pos == -1 or pos_first < first_pos:
                    first_pos = pos_first
            pos_last = body.rfind(p)
            if pos_last != -1:
                end = pos_last + len(p)
                if end > last_end:
                    last_end = end
        if first_pos == -1:
            return (
                body[: CategoryPredictor._EMAIL_CONTEXT_CHARS_BEFORE_AMOUNT]
                if len(body) > CategoryPredictor._EMAIL_CONTEXT_CHARS_BEFORE_AMOUNT
                else body
            )
        start = max(0, first_pos - CategoryPredictor._EMAIL_CONTEXT_CHARS_BEFORE_AMOUNT)
        return body[start:last_end] if last_end != -1 else body[start:]

    @staticmethod
    def _email_contexts_for_llm(emails_sorted: list[EmailEvidence], amount: Decimal) -> list[str]:
        """Per-email context strings (subject + body context), deduplicated. Same logic as _emails_for_llm but returns a list."""
        if not emails_sorted:
            return []
        contexts: list[tuple[EmailEvidence, str]] = []
        for em in emails_sorted[:3]:
            subject = (em.subject or "").strip()
            body = (
                getattr(em, "filtered_body", None) or em.body_snippet or em.full_body or ""
            ).strip()
            body_ctx = CategoryPredictor._extract_body_context_around_amount(body, amount)
            ctx_str = f"Subject: {subject}\nBody: {body_ctx}"
            contexts.append((em, ctx_str))
        kept: list[tuple[EmailEvidence, str]] = []
        for em, ctx_str in contexts:
            is_dup = False
            for _, kept_ctx in kept:
                if (
                    difflib.SequenceMatcher(None, ctx_str, kept_ctx).ratio()
                    >= CategoryPredictor._EMAIL_DEDUPE_SIMILARITY_THRESHOLD
                ):
                    is_dup = True
                    break
            if not is_dup:
                kept.append((em, ctx_str))
        return [ctx for _, ctx in kept]

    @staticmethod
    def get_emails_for_display(
        emails_sorted: list[EmailEvidence], amount: Decimal
    ) -> list[tuple[EmailEvidence, str]]:
        """Deduplicated (email, context) for web display. Context = 500 chars before first amount, between earliest and last amount, nothing after. Dedupe by 95%% context similarity."""
        if not emails_sorted:
            return []
        contexts: list[tuple[EmailEvidence, str]] = []
        for em in emails_sorted[:3]:
            subject = (em.subject or "").strip()
            body = (
                getattr(em, "filtered_body", None) or em.body_snippet or em.full_body or ""
            ).strip()
            body_ctx = CategoryPredictor._extract_body_context_around_amount(body, amount)
            ctx_str = f"Subject: {subject}\nBody: {body_ctx}"
            contexts.append((em, ctx_str))
        kept: list[tuple[EmailEvidence, str]] = []
        for em, ctx_str in contexts:
            is_dup = False
            for _, kept_ctx in kept:
                if (
                    difflib.SequenceMatcher(None, ctx_str, kept_ctx).ratio()
                    >= CategoryPredictor._EMAIL_DEDUPE_SIMILARITY_THRESHOLD
                ):
                    is_dup = True
                    break
            if not is_dup:
                kept.append((em, ctx_str))
        return kept

    @staticmethod
    def _emails_for_llm(emails_sorted: list[EmailEvidence], amount: Decimal) -> str:
        """Format up to 3 emails: clean subject + body context (500 chars before earliest amount, up to last amount; nothing after). Deduplicate by 95%% context similarity."""
        contexts = CategoryPredictor._email_contexts_for_llm(emails_sorted, amount)
        if not contexts:
            return ""
        blocks = [f"--- Email {i} ---\n{ctx}" for i, ctx in enumerate(contexts, 1)]
        return "\n\n".join(blocks)

    @staticmethod
    def _format_categories_compact(paths: list[str]) -> tuple[list[str], list[str]]:
        """Filter to relevant paths (Expenses/Income only; excludes Assets, Equity, etc.), group by prefix.
        Returns (formatted_lines, allowed_paths). Formatted list is compact: group header (e.g. 'Expenses:')
        then lines like '  Food:Groceries' meaning full path Expenses:Food:Groceries. Saves tokens.
        """
        allowed = [
            p
            for p in paths
            if p in ("Expenses", "Income")
            or any(p.startswith(prefix) for prefix in CategoryPredictor._LLM_CATEGORY_PREFIXES)
        ]
        if not allowed:
            return [], list(paths)
        by_prefix: dict[str, list[str]] = {}
        for p in allowed:
            if p == "Expenses":
                by_prefix.setdefault("Expenses:", []).append("Expenses")
            elif p == "Income":
                by_prefix.setdefault("Income:", []).append("Income")
            else:
                for prefix in CategoryPredictor._LLM_CATEGORY_PREFIXES:
                    if p.startswith(prefix):
                        rest = p[len(prefix) :].strip()
                        if rest:
                            by_prefix.setdefault(prefix, []).append(rest)
                        break
        lines: list[str] = []
        for prefix in CategoryPredictor._LLM_CATEGORY_PREFIXES:
            if prefix not in by_prefix:
                continue
            subs = sorted(set(by_prefix[prefix]))
            lines.append(prefix)
            for s in subs:
                lines.append("  " + s)
            lines.append("")
        if lines and lines[-1] == "":
            lines.pop()
        return lines, allowed

    @staticmethod
    def _parse_llm_json(raw: str) -> dict | None:
        """Parse JSON from LLM response; tolerate markdown or extra text by extracting first {...} block."""
        if not raw or not raw.strip():
            return None
        s = raw.strip()
        try:
            data = json.loads(s)
            return data if isinstance(data, dict) else None
        except json.JSONDecodeError:
            pass
        start = s.find("{")
        end = s.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                data = json.loads(s[start : end + 1])
                return data if isinstance(data, dict) else None
            except json.JSONDecodeError:
                pass
        logger.debug("LLM response is not valid JSON: %s", _truncate_for_log(s))
        return None

    def _llm_post(self, user_content: str, label: str, max_tokens: int = 300) -> str | None:
        """Send one user message to the LLM; return raw content or None. Logs size and response time."""
        try:
            client = LlmClient(self._llm_config)
            if not client.enabled:
                return None
            logger.debug(
                "LLM (%s) full request (%d chars):\n%s\n<<< END LLM REQUEST >>>",
                label,
                len(user_content),
                _truncate_for_log(user_content),
            )
            data = client.chat(
                messages=[{"role": "user", "content": user_content}],
                max_tokens=max_tokens,
                temperature=self._llm_config.temperature,
            )
            if not data:
                return None
            raw = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            logger.debug(
                "LLM (%s) full response:\n%s\n<<< END LLM RESPONSE >>>",
                label,
                _truncate_for_log(raw),
            )
            return raw
        except Exception:
            logger.warning("LLM query failed (%s)", label, exc_info=True)
            return None

    def _llm_post_extraction_with_messages(
        self, messages: list[dict], max_tokens: int = 500
    ) -> str | None:
        """Send messages to the extraction LLM; return last assistant content or None."""
        endpoint = getattr(self._llm_config, "extraction_endpoint", "") or ""
        model = getattr(self._llm_config, "extraction_model", "") or ""
        if not endpoint or not model:
            return None
        try:
            llm_cfg = LlmConfig(
                mode=self._llm_config.mode,
                endpoint=endpoint,
                model_name=model,
                api_key=getattr(self._llm_config, "extraction_api_key", "") or "",
                temperature=getattr(self._llm_config, "temperature", 0.2),
                max_tokens=max_tokens,
                timeout_seconds=getattr(self._llm_config, "timeout_seconds", 180),
            )
            client = LlmClient(llm_cfg)
            if not client.enabled:
                return None
            total_chars = sum(len(str(m.get("content", ""))) for m in messages)
            request_body = "\n---\n".join(
                f"[{m.get('role', '')}]: {str(m.get('content', ''))}" for m in messages
            )
            logger.debug(
                "LLM (extraction) full request (%d chars):\n%s\n<<< END LLM REQUEST >>>",
                total_chars,
                _truncate_for_log(request_body),
            )
            data = client.chat(messages=messages, max_tokens=max_tokens)
            if not data:
                return None
            raw = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            logger.debug(
                "LLM (extraction) full response:\n%s\n<<< END LLM RESPONSE >>>",
                _truncate_for_log(raw),
            )
            return raw
        except Exception:
            logger.warning("Extraction LLM query failed", exc_info=True)
            return None

    @staticmethod
    def _format_extraction_for_prompt(extraction: dict) -> str:
        """Format extracted email data as a short block for the category LLM. Includes web descriptions when present."""
        parts = []
        seller = extraction.get("seller_name")
        if seller:
            line = f"Seller: {seller}"
            web = extraction.get("seller_web_description")
            if web:
                line += f" (web: {web})"
            parts.append(line)
        order_ids = extraction.get("order_ids") or []
        if not order_ids and extraction.get("order_id"):
            order_ids = [extraction["order_id"]]
        if order_ids:
            parts.append("Order/transaction IDs: " + ", ".join(str(x) for x in order_ids))
        txn_ids = extraction.get("transaction_ids") or []
        if txn_ids and txn_ids != order_ids:
            parts.append("Transaction IDs: " + ", ".join(str(x) for x in txn_ids))
        items = extraction.get("items") or []
        if items:
            item_strs = []
            for it in items[:20]:
                if not isinstance(it, dict):
                    continue
                desc = it.get("description") or it.get("name") or ""
                amt = it.get("amount") or it.get("price") or ""
                web_desc = it.get("web_description")
                if desc and amt:
                    item_strs.append(f"{desc} £{amt}" + (f" (web: {web_desc})" if web_desc else ""))
                elif desc:
                    item_strs.append(desc + (f" (web: {web_desc})" if web_desc else ""))
            if item_strs:
                parts.append("Items: " + "; ".join(item_strs))
        return "\n".join(parts) if parts else ""

    _EXTRACT_JSON_INSTRUCTION_FALLBACK = (
        "Extract from the following email and return only valid JSON. "
        "Fields: seller_name (string), items (array of {description, amount} per line item), order_ids (array of strings), transaction_ids (array of strings). "
        "Use empty array/string if not found. Amounts as numbers or strings.\n\n--- Email ---\n"
    )

    def _enrich_extraction_with_web(self, data: dict) -> None:
        """Optionally add web descriptions for seller/items when use_web is enabled.

        Mutates *data* in place. Safe to call with any mapping; no-op when use_web
        is False or when seller/items keys are missing.
        """
        if not getattr(self._llm_config, "use_web", False):
            return
        seller = (data.get("seller_name") or "").strip()
        if seller and _has_name_like_content(seller):
            logger.debug("Extraction LLM (use_web): web search for seller %r", seller[:50])
            web_seller = self._web_search_short(seller, min_chars=50, max_chars=200)
            if web_seller:
                data["seller_web_description"] = web_seller
                logger.debug(
                    "Extraction LLM (use_web): seller web description %d chars", len(web_seller)
                )
        for idx, it in enumerate(data.get("items") or []):
            if not isinstance(it, dict):
                logger.debug(
                    "Extraction LLM (use_web): skip item %d (not a dict: %s)",
                    idx + 1,
                    type(it).__name__,
                )
                continue
            desc = (it.get("description") or it.get("name") or "").strip()
            # Only add web description when it looks like a name/description, not just amount digits
            if not desc or not _has_name_like_content(desc):
                continue
            logger.debug("Extraction LLM (use_web): web search for item %d %r", idx + 1, desc[:50])
            web_item = self._web_search_short(desc, min_chars=50, max_chars=200)
            if web_item:
                it["web_description"] = web_item
                logger.debug(
                    "Extraction LLM (use_web): item %d web description %d chars",
                    idx + 1,
                    len(web_item),
                )

    @staticmethod
    def _sanitize_extraction_items(data: dict) -> dict:
        """Ensure extraction['items'] contains only dicts, logging drop count via caller."""
        raw_items = data.get("items") or []
        items = [it for it in raw_items if isinstance(it, dict)]
        data["items"] = items
        return data

    def _query_llm_extract(
        self, emails_sorted: list[EmailEvidence], amount: Decimal
    ) -> dict | None:
        """Per-email extraction in same session, then merge. When use_web, add 50-200 char web descriptions for seller and each item. Returns parsed dict or None."""
        email_contexts = self._email_contexts_for_llm(emails_sorted, amount)
        if not email_contexts:
            logger.debug("Extraction LLM: no email contexts")
            return None
        logger.info(
            "Extraction LLM: %d email(s), per-email extraction then merge", len(email_contexts)
        )
        extract_tpl = load_template(self._prompts_dir, "extract_email")
        messages: list[dict] = []
        extraction_raws: list[str] = []
        for i, ctx in enumerate(email_contexts):
            user_content = (
                render(extract_tpl, email_context=ctx)
                if extract_tpl
                else self._EXTRACT_JSON_INSTRUCTION_FALLBACK + ctx
            )
            messages.append({"role": "user", "content": user_content})
            logger.debug(
                "Extraction LLM: query %d/%d for email %d (%d chars)",
                i + 1,
                len(email_contexts),
                i + 1,
                len(ctx),
            )
            raw = self._llm_post_extraction_with_messages(messages, max_tokens=500)
            if not raw:
                logger.debug("Extraction LLM: no response for email %d", i + 1)
                break
            extraction_raws.append(raw)
            messages.append({"role": "assistant", "content": raw})
        if not extraction_raws:
            return None
        if len(extraction_raws) == 1:
            data = self._parse_llm_json(extraction_raws[0])
        else:
            merge_tpl = load_template(self._prompts_dir, "extract_merge")
            extractions_blob = "\n\n".join(
                f"--- Extraction {j + 1} ---\n{r}" for j, r in enumerate(extraction_raws)
            )
            merge_prompt = (
                render(merge_tpl, extractions=extractions_blob)
                if merge_tpl
                else "Merge the following extraction results into one JSON. "
                "Use seller_name from the first non-empty; combine items (deduplicate by description); "
                "merge order_ids and transaction_ids arrays. Return only valid JSON with keys: "
                "seller_name, items, order_ids, transaction_ids.\n\n" + extractions_blob
            )
            messages.append({"role": "user", "content": merge_prompt})
            logger.debug(
                "Extraction LLM: merge query with %d per-email results", len(extraction_raws)
            )
            raw_merge = self._llm_post_extraction_with_messages(messages, max_tokens=600)
            if not raw_merge:
                logger.debug("Extraction LLM: no response for merge")
                return None
            data = self._parse_llm_json(raw_merge)
        if not isinstance(data, dict):
            logger.debug("Extraction LLM: merge result was not valid JSON")
            return None
        logger.debug("Extraction LLM: merge parsed with keys %s", list(data.keys()))
        self._enrich_extraction_with_web(data)
        raw_items = data.get("items") or []
        data = self._sanitize_extraction_items(data)
        if len(data["items"]) != len(raw_items):
            logger.debug(
                "Extraction LLM: filtered items to %d dicts (dropped %d non-dict entries)",
                len(data["items"]),
                len(raw_items) - len(data["items"]),
            )
        return data

    _EXTRACT_FROM_DESCRIPTION_FALLBACK = (
        "From the following transaction description and amount only (no email), extract if present: "
        "supplier/seller name, and item name(s). Return only valid JSON with keys: seller_name (string), "
        "items (array of {description, amount} per item; if the description suggests a single purchase use the "
        "transaction amount for that item, else empty array if unclear). Use empty string or empty array when not found.\n\n"
        "Transaction description: {{description}}\nAmount: £{{amount}}"
    )

    def _query_llm_extract_from_description(self, tx: Transaction) -> dict | None:
        """Extract supplier and item(s) from transaction description when there are no emails.
        Uses extraction LLM if configured, otherwise main LLM. When use_web is set, adds web descriptions for seller and items.
        """
        desc = (tx.description or "").strip()
        if not desc:
            return None
        tpl = load_template(self._prompts_dir, "extract_from_description")
        prompt = (
            render(tpl, description=desc, amount=str(tx.amount))
            if tpl
            else self._EXTRACT_FROM_DESCRIPTION_FALLBACK.replace("{{description}}", desc).replace(
                "{{amount}}", str(tx.amount)
            )
        )
        raw: str | None = None
        if getattr(self._llm_config, "extraction_endpoint", "") and getattr(
            self._llm_config, "extraction_model", ""
        ):
            raw = self._llm_post_extraction_with_messages(
                [{"role": "user", "content": prompt}], max_tokens=400
            )
        if not raw and self._llm_config.endpoint and self._llm_config.model_name:
            raw = self._llm_post(prompt, "extract from description", max_tokens=400)
        if not raw:
            return None
        data = self._parse_llm_json(raw)
        if not isinstance(data, dict):
            return None
        self._enrich_extraction_with_web(data)
        raw_items = data.get("items") or []
        data = self._sanitize_extraction_items(data)
        if len(data["items"]) != len(raw_items):
            logger.debug(
                "Extraction LLM: filtered items to %d dicts (dropped %d non-dict entries)",
                len(data["items"]),
                len(raw_items) - len(data["items"]),
            )
        return data

    def _query_llm_step1(
        self,
        tx: Transaction,
        expenses_first: list[str],
        emails_sorted: list[EmailEvidence] | None = None,
        extracted_from_emails: str | None = None,
    ) -> dict | None:
        """Step 1 (or 2 when extraction ran): improve description, confidence, pick category. When extracted_from_emails is set, that is shown instead of raw emails."""
        cats_line = self._format_expenses_first_level_for_prompt(expenses_first)
        has_emails = bool(extracted_from_emails or emails_sorted)
        # Build input block first so model sees data before tasks
        input_parts = [
            f"Description: {tx.description}",
            f"Amount: £{tx.amount}",
        ]
        if extracted_from_emails:
            input_parts.append(f"Extracted from emails:\n{extracted_from_emails}")
        elif emails_sorted:
            emails_block = self._emails_for_llm(emails_sorted, tx.amount)
            input_parts.append(f"Emails:\n{emails_block}")
        input_block = "\n".join(input_parts)
        # Tasks: same structure for both; only description task differs
        if has_emails:
            desc_task = (
                "Using the details provided in the Input above, improve the transaction description (do not include the total amount). "
                "Ensure the improved description includes the order/transaction ID or number when it appears in the Input."
            )
        else:
            desc_task = "Improve the transaction description (do not include the total amount)."
        step1_tpl = load_template(self._prompts_dir, "category_step1")
        example = '{"improved_description": "PayPal payment to Seller for item X", "confidence": 7, "category": "Food"}'
        if step1_tpl:
            user_content = render(
                step1_tpl,
                expense_note=CategoryPredictor._EXPENSE_NOTE,
                input_block=input_block,
                desc_task=desc_task,
                categories_line=cats_line,
                example=example,
            )
        else:
            user_content = (
                f"{CategoryPredictor._EXPENSE_NOTE}\n\n"
                "--- Input ---\n"
                f"{input_block}\n\n"
                "--- Tasks ---\n"
                f"1) {desc_task}\n\n"
                "2) Confidence 1-10 that the transaction can be categorised: 1 = not at all confident, 10 = very confident. "
                "You can only be confident if: the description or email contains an item description that fits the chosen category, OR the seller is known to belong to that category. "
                "Do not inflate the score.\n\n"
                f"3) Categories (pick exactly one from this list): {cats_line}\n"
                "You must pick one category from the list above; use only one of these names. Do not invent a category.\n\n"
                "--- Output ---\n"
                "Reply with only valid JSON, no other text. Example:\n"
                f"{example}"
            )
        logger.info("LLM step 1: emails=%s", has_emails)
        raw = self._llm_post(user_content, "category step 1", max_tokens=350)
        if not raw:
            return None
        data = self._parse_llm_json(raw)
        if not data:
            return None
        improved_description = (
            data.get("improved_description") or data.get("description") or tx.description or ""
        ).strip()
        if not improved_description:
            improved_description = tx.description or ""
        try:
            confidence = int(data.get("confidence", 0))
        except (TypeError, ValueError):
            confidence = 0
        confident = confidence > CategoryPredictor._CONFIDENCE_THRESHOLD
        category_raw = (data.get("category") or "").strip()
        if category_raw in expenses_first:
            category = category_raw
        else:
            name_match = next(
                (p for p in expenses_first if p == category_raw or p.endswith(":" + category_raw)),
                None,
            )
            category = name_match if name_match else (expenses_first[0] if expenses_first else "")
        if category and not category.startswith("Expenses:"):
            category = "Expenses:" + category
        return {
            "improved_description": improved_description,
            "confidence": confidence,
            "confident": confident,
            "category": category,
        }

    def _query_llm_step2(
        self,
        improved_description: str,
        amount: Decimal,
        chosen: str,
        subcategories: list[str],
    ) -> dict | None:
        """Step 2: present improved description and amount; subcategories as subcat1:{a;b};subcat2; Pick one. Full path = Expenses:chosen:picked."""
        if not subcategories:
            return None
        desc_block = f"Description: {improved_description}\nAmount: £{amount}"
        cats_block = CategoryPredictor._format_step2_subcategories(chosen, subcategories)
        if not cats_block:
            return {"category": chosen, "description": improved_description}
        step2_tpl = load_template(self._prompts_dir, "category_step2")
        if step2_tpl:
            user_content = render(
                step2_tpl,
                description_block=desc_block,
                categories_block=cats_block,
            )
        else:
            instruction = (
                "Pick one category from the list below that best matches the description. "
                "Reply with only that category exactly as shown. One line, no JSON, no explanation.\n"
                "Example reply: Household:Kitchen equipment & expendables"
            )
            user_content = (
                f"{desc_block}\n\n--- Categories ---\n{cats_block}\n\n--- Task ---\n{instruction}"
            )
        raw = self._llm_post(user_content, "category step 2", max_tokens=150)
        if not raw:
            return None
        raw = raw.strip()
        picked = raw
        try:
            data = json.loads(raw)
            if isinstance(data, dict) and data.get("category"):
                picked = (data.get("category") or "").strip()
        except json.JSONDecodeError:
            start, end = raw.find("{"), raw.rfind("}")
            if start != -1 and end > start:
                try:
                    data = json.loads(raw[start : end + 1])
                    if isinstance(data, dict) and data.get("category"):
                        picked = (data.get("category") or "").strip()
                except json.JSONDecodeError:
                    pass
        if not picked:
            return {"category": chosen, "description": improved_description}
        prefix = chosen + ":"
        # Reconstruct full path as chosen + ":" + picked (e.g. Expenses:Accommodation:Kitchen equipment & expendables)
        candidate = picked if picked.startswith(prefix) else (prefix + picked)
        logger.debug("LLM step 2 picked full path: %s", candidate)
        return {"category": candidate, "description": improved_description}

    def _run_llm_flow(
        self,
        tx: Transaction,
        emails: list[EmailEvidence],
        account_paths: list[str],
    ) -> dict | None:
        """Single path for extraction + category LLM. Returns dict with extraction, category, description, confidence (0-1), or None."""
        try:

            def _email_sort_key(e: EmailEvidence) -> datetime:
                if isinstance(e.sent_at, datetime):
                    return e.sent_at
                return datetime(9999, 12, 31, tzinfo=timezone.utc)

            emails_sorted = sorted(emails[:3], key=_email_sort_key)
            allowed_paths = [
                p
                for p in account_paths
                if p in ("Expenses", "Income")
                or any(p.startswith(prefix) for prefix in self._LLM_CATEGORY_PREFIXES)
            ]
            allowed_paths = self._filter_gbp_paths_only(allowed_paths)
            if not allowed_paths:
                allowed_paths = self._filter_gbp_paths_only(list(account_paths))
            if not allowed_paths:
                return None
            expenses_first = self._get_expenses_first_level(allowed_paths)
            if not expenses_first:
                return None

            extraction: dict | None = None
            extracted_from_emails: str | None = None
            if emails_sorted and self._llm_config.extraction_endpoint:
                extraction = self._query_llm_extract(emails_sorted, tx.amount)
                if extraction:
                    extracted_from_emails = self._format_extraction_for_prompt(extraction)
                    if extracted_from_emails:
                        logger.debug("Using extracted email data for category step (3-step flow)")
            elif not emails_sorted:
                extraction = self._query_llm_extract_from_description(tx)
                if extraction:
                    extracted_from_emails = self._format_extraction_for_prompt(extraction)
                    if extracted_from_emails:
                        logger.debug(
                            "Using extraction from transaction description for category step (no-email flow)"
                        )
            step1 = self._query_llm_step1(
                tx,
                expenses_first,
                emails_sorted=emails_sorted if extracted_from_emails is None else None,
                extracted_from_emails=extracted_from_emails,
            )
            if not step1:
                return None
            chosen = step1.get("category") or (allowed_paths[0] if allowed_paths else "Expense")
            if "category" not in step1:
                logger.warning("LLM step1 response missing 'category'; using fallback %s", chosen)
            improved_description = step1.get("improved_description") or tx.description or ""
            step1_confidence_raw = step1.get("confidence", 0)
            try:
                llm_confidence = min(1.0, max(0.0, int(step1_confidence_raw) / 10.0))
            except (TypeError, ValueError):
                llm_confidence = 0.0

            if not step1.get("confident", True):
                final_category = (
                    chosen
                    if chosen in allowed_paths
                    else (allowed_paths[0] if allowed_paths else chosen)
                )
                return {
                    "extraction": extraction,
                    "category": final_category,
                    "description": improved_description,
                    "confidence": llm_confidence,
                }
            subcategories = self._get_subcategories(chosen, allowed_paths)
            step2 = self._query_llm_step2(improved_description, tx.amount, chosen, subcategories)
            if not step2:
                return {
                    "extraction": extraction,
                    "category": chosen,
                    "description": improved_description,
                    "confidence": llm_confidence,
                }
            final_category = step2.get("category") or chosen
            if "category" not in step2:
                logger.warning(
                    "LLM step2 response missing 'category'; using step1 choice %s", final_category
                )
            if final_category not in allowed_paths:
                final_category = (
                    chosen
                    if chosen in allowed_paths
                    else (allowed_paths[0] if allowed_paths else chosen)
                )
            logger.debug("LLM category full path after step 2: %s", final_category)
            return {
                "extraction": extraction,
                "category": final_category,
                "description": step2.get("description") or improved_description,
                "confidence": llm_confidence,
            }
        except Exception:
            logger.warning("LLM flow failed", exc_info=True)
            return None

    def _query_llm(
        self,
        tx: Transaction,
        emails: list[EmailEvidence],
        account_paths: list[str],
    ) -> LlmSuggestion | None:
        """LLM for propose(): extraction (if configured) + category step 1 + step 2. Returns LlmSuggestion or None."""
        if self._llm_config.mode == LlmMode.DISABLED:
            return None
        result = self._run_llm_flow(tx, emails, account_paths)
        if not result:
            return None
        return LlmSuggestion(
            category=result["category"],
            reason="",
            description=result["description"],
            confidence=result["confidence"],
        )

    def run_llm_check(
        self,
        tx: Transaction,
        emails: list[EmailEvidence],
        receipt: ReceiptEvidence | None,
        account_paths: list[str],
    ) -> dict | None:
        """Run extraction + category LLM for review UI. Returns dict with extraction, category, description, confidence (0-1), or None if LLM disabled/failed."""
        if self._llm_config.mode == LlmMode.DISABLED:
            return None
        result = self._run_llm_flow(tx, emails, account_paths)
        if not result:
            return None
        return result


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
            "suggested_category": (
                proposal.suggested_splits[0].account_path if proposal.suggested_splits else ""
            ),
            "confidence": proposal.confidence,
            "accepted": accepted,
            "note": note,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        if self._state_dir:
            from gnc_enrich.state.repository import StateRepository

            repo = StateRepository(self._state_dir)
            repo.append_feedback(feedback)

        logger.info("Feedback recorded: proposal=%s accepted=%s", proposal.proposal_id, accepted)
