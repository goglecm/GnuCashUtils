"""Tests for ML category predictor and feedback trainer."""

import requests
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch, MagicMock

from gnc_enrich.config import LlmConfig, LlmMode
from gnc_enrich.domain.models import (
    EmailEvidence,
    Proposal,
    ReceiptEvidence,
    Split,
    Transaction,
)
from gnc_enrich.ml.predictor import CategoryPredictor, FeedbackTrainer


def _make_history() -> list[Transaction]:
    """Generate synthetic historical transactions for training."""
    txs = []
    food_descs = [
        "Tesco Weekly Shop", "Sainsburys Groceries", "Asda Online",
        "Lidl Store", "Aldi Supermarket", "M&S Food Hall",
        "Waitrose Order", "Co-op Food", "Tesco Express", "Sainsburys Local",
    ]
    transport_descs = [
        "Uber Ride", "Trainline Ticket", "TfL Bus", "Shell Petrol",
        "BP Fuel", "Parking Meter", "National Rail", "First Bus",
        "Uber Trip", "EasyJet Flight",
    ]
    for i, desc in enumerate(food_descs):
        txs.append(Transaction(
            tx_id=f"hist_food_{i}",
            posted_date=date(2024, 6, 1 + i),
            description=desc,
            currency="GBP",
            amount=Decimal("25.00"),
            splits=[
                Split(account_path="Current Account", amount=Decimal("-25.00")),
                Split(account_path="Expenses:Food", amount=Decimal("25.00")),
            ],
            account_name="Current Account",
            original_category="Expenses:Food",
        ))
    for i, desc in enumerate(transport_descs):
        txs.append(Transaction(
            tx_id=f"hist_transport_{i}",
            posted_date=date(2024, 6, 1 + i),
            description=desc,
            currency="GBP",
            amount=Decimal("15.00"),
            splits=[
                Split(account_path="Current Account", amount=Decimal("-15.00")),
                Split(account_path="Expenses:Transport", amount=Decimal("15.00")),
            ],
            account_name="Current Account",
            original_category="Expenses:Transport",
        ))
    return txs


def _make_target_tx(desc: str = "Card Payment", amount: str = "25.00") -> Transaction:
    return Transaction(
        tx_id="target_1",
        posted_date=date(2025, 1, 15),
        description=desc,
        currency="GBP",
        amount=Decimal(amount),
        splits=[
            Split(account_path="Current Account", amount=Decimal(f"-{amount}")),
            Split(account_path="Unspecified", amount=Decimal(amount)),
        ],
        account_name="Current Account",
        original_category="Unspecified",
    )


class TestCategoryPredictor:

    def test_trained_predictor_returns_valid_proposal(self) -> None:
        history = _make_history()
        predictor = CategoryPredictor(historical_transactions=history)

        tx = _make_target_tx(desc="Tesco Store Purchase")
        emails = [
            EmailEvidence(
                evidence_id="e1", message_id="<m1>", sender="orders@tesco.com",
                subject="Your Tesco receipt", sent_at=datetime(2025, 1, 15, 12, 0, tzinfo=timezone.utc),
                parsed_amounts=[Decimal("25.00")],
            )
        ]
        proposal = predictor.propose(tx, emails, None)

        assert isinstance(proposal, Proposal)
        assert proposal.tx_id == "target_1"
        assert 0.0 <= proposal.confidence <= 1.0
        assert proposal.suggested_splits
        assert proposal.rationale

    def test_food_transaction_classified_as_food(self) -> None:
        history = _make_history()
        predictor = CategoryPredictor(historical_transactions=history)

        tx = _make_target_tx(desc="Tesco Groceries Weekly")
        proposal = predictor.propose(tx, [], None)

        assert proposal.suggested_splits[0].account_path == "Expenses:Food"

    def test_transport_transaction_classified(self) -> None:
        history = _make_history()
        predictor = CategoryPredictor(historical_transactions=history)

        tx = _make_target_tx(desc="Uber Ride to Airport")
        proposal = predictor.propose(tx, [], None)

        assert proposal.suggested_splits[0].account_path == "Expenses:Transport"

    def test_confidence_between_0_and_1(self) -> None:
        history = _make_history()
        predictor = CategoryPredictor(historical_transactions=history)

        tx = _make_target_tx(desc="Tesco Express")
        proposal = predictor.propose(tx, [], None)

        assert 0.0 <= proposal.confidence <= 1.0

    def test_untrained_predictor_uses_fallback(self) -> None:
        predictor = CategoryPredictor()
        tx = _make_target_tx(desc="Netflix Subscription")
        proposal = predictor.propose(tx, [], None)

        assert proposal.confidence == 0.3
        assert "heuristic" in proposal.rationale.lower() or "fallback" in proposal.rationale.lower()

    def test_fallback_keyword_matching(self) -> None:
        predictor = CategoryPredictor()

        tx = _make_target_tx(desc="Tesco Supermarket")
        proposal = predictor.propose(tx, [], None)
        assert "Food" in proposal.suggested_splits[0].account_path

        tx2 = _make_target_tx(desc="Shell Petrol Station")
        proposal2 = predictor.propose(tx2, [], None)
        assert "Transport" in proposal2.suggested_splits[0].account_path

    def test_description_no_date_appended(self) -> None:
        """ML recommended description does not append the transaction date (per spec)."""
        predictor = CategoryPredictor()
        tx = _make_target_tx(desc="Card Payment")
        proposal = predictor.propose(tx, [], None)
        assert "Card Payment" in proposal.suggested_description
        assert "15/01/2025" not in proposal.suggested_description

    def test_description_includes_merchant_from_email(self) -> None:
        predictor = CategoryPredictor()
        tx = _make_target_tx(desc="Card Payment")
        emails = [
            EmailEvidence(
                evidence_id="e1", message_id="<m1>", sender="orders@tesco.com",
                subject="Receipt", sent_at=datetime(2025, 1, 15, 12, 0, tzinfo=timezone.utc),
            )
        ]
        proposal = predictor.propose(tx, emails, None)
        assert "Tesco" in proposal.suggested_description

    def test_evidence_packet_attached(self) -> None:
        predictor = CategoryPredictor()
        tx = _make_target_tx()
        proposal = predictor.propose(tx, [], None)
        assert proposal.evidence is not None
        assert proposal.evidence.tx_id == tx.tx_id

    def test_llm_query_attempted_on_low_confidence(self) -> None:
        """With no emails: extract-from-description then step1 then step2."""
        llm_cfg = LlmConfig(mode=LlmMode.ONLINE, endpoint="http://fake:1234/v1/chat/completions", model_name="test")
        predictor = CategoryPredictor(llm_config=llm_cfg)
        tx = _make_target_tx(desc="Unknown purchase")
        account_paths = ["Expenses:Shopping", "Expenses:Shopping:Online", "Expenses:Food"]

        with patch("gnc_enrich.llm.client.LlmClient.chat") as mock_chat:
            mock_chat.side_effect = [
                {"choices": [{"message": {"content": '{"seller_name": "", "items": []}'}}]},
                {
                    "choices": [
                        {
                            "message": {
                                "content": '{"improved_description": "Unknown purchase 15/01/2025", "confidence": 8, "category": "Shopping"}'
                            }
                        }
                    ]
                },
                {"choices": [{"message": {"content": '{"category": "Online"}'}}]},
            ]

            proposal = predictor.propose(tx, [], None, account_paths=account_paths)
            assert mock_chat.call_count == 3
            assert "LLM suggestion" in proposal.rationale
            assert proposal.suggested_splits[0].account_path == "Expenses:Shopping:Online"
            assert proposal.suggested_description == "Unknown purchase 15/01/2025"
            assert any("LLM picked" in b for b in proposal.confidence_breakdown)

    def test_llm_non_dict_response_ignored(self) -> None:
        """When extraction or step1 returns non-dict JSON (e.g. list), we ignore it and use ML fallback."""
        llm_cfg = LlmConfig(mode=LlmMode.ONLINE, endpoint="http://fake:1234/v1/chat/completions", model_name="test")
        predictor = CategoryPredictor(llm_config=llm_cfg)
        tx = _make_target_tx(desc="Unknown purchase")
        with patch("gnc_enrich.llm.client.LlmClient.chat") as mock_chat:
            mock_chat.return_value = {"choices": [{"message": {"content": "[]"}}]}
            proposal = predictor.propose(tx, [], None, account_paths=["Expenses:Food"])
            # No emails: extract-from-description then step1; both get "[]" and are ignored
            assert mock_chat.call_count == 2
            assert "LLM suggestion" not in proposal.rationale
            assert proposal.suggested_splits[0].account_path == "Expenses:Miscellaneous"

    def test_get_top_level_and_subcategories(self) -> None:
        """Top-level = first two segments, cap 20; subcategories = chosen or children."""
        paths = ["Expenses:Food", "Expenses:Food:Groceries", "Expenses:Transport", "Income:Salary"]
        top = CategoryPredictor._get_top_level_categories(paths)
        assert top == ["Expenses:Food", "Expenses:Transport", "Income:Salary"]
        sub = CategoryPredictor._get_subcategories("Expenses:Food", paths)
        assert sub == ["Expenses:Food", "Expenses:Food:Groceries"]
        sub2 = CategoryPredictor._get_subcategories("Expenses:Transport", paths)
        assert sub2 == ["Expenses:Transport"]

    def test_expenses_first_level_and_format(self) -> None:
        """Step 1: only Expenses first-level; format is short names (Food; Household)."""
        paths = ["Expenses:Food", "Expenses:Household", "Expenses:Food:Groceries", "Income:Salary"]
        first = CategoryPredictor._get_expenses_first_level(paths)
        assert first == ["Expenses:Food", "Expenses:Household"]
        out = CategoryPredictor._format_expenses_first_level_for_prompt(first)
        assert out == "Food; Household"

    def test_get_subcategories_and_format_step2(self) -> None:
        """Step 2: _get_subcategories returns chosen + paths under it; _format_step2_subcategories formats for prompt."""
        paths = ["Expenses:Food", "Expenses:Food:Groceries", "Expenses:Food:Takeaway", "Expenses:Transport"]
        sub = CategoryPredictor._get_subcategories("Expenses:Food", paths)
        assert "Expenses:Food" in sub and "Expenses:Food:Groceries" in sub and "Expenses:Food:Takeaway" in sub
        assert "Expenses:Transport" not in sub
        out = CategoryPredictor._format_step2_subcategories("Expenses:Food", sub)
        assert "Groceries" in out and "Takeaway" in out
        assert "{" in out or ";" in out
        single = ["Expenses:Transport"]
        single_sub = CategoryPredictor._get_subcategories("Expenses:Transport", single)
        assert single_sub == ["Expenses:Transport"]
        both_leaf_and_branch = ["Expenses:Accommodation", "Expenses:Accommodation:Hotel"]
        out2 = CategoryPredictor._format_step2_subcategories("Expenses", both_leaf_and_branch)
        assert "Hotel" in out2 and "Accommodation" in out2

    def test_filter_gbp_paths_excludes_non_gbp_currency(self) -> None:
        """Paths with non-GBP currency (segment or in parentheses in segment) are excluded."""
        paths = [
            "Expenses:Food",
            "Expenses:USD",
            "Income:Salary",
            "Assets:Bank:EUR",
            "Expenses:Food:Eat in & takeaway (CZK)",
            "Expenses:Food:Groceries (GBP)",
        ]
        out = CategoryPredictor._filter_gbp_paths_only(paths)
        assert "Expenses:Food" in out and "Income:Salary" in out
        assert "Expenses:Food:Groceries (GBP)" in out
        assert "Expenses:USD" not in out and "Assets:Bank:EUR" not in out
        assert "Expenses:Food:Eat in & takeaway (CZK)" not in out

    def test_extract_body_context_around_amount(self) -> None:
        """Up to 500 chars before earliest amount; include up to last amount; strip after last."""
        body = "Some preamble. " * 10 + "Total £25.50 paid. Thanks."
        ctx = CategoryPredictor._extract_body_context_around_amount(body, Decimal("25.50"))
        assert "25.50" in ctx
        assert "Thanks" not in ctx
        assert ctx.endswith("25.50")
        # When amount appears twice: include up to 500 chars before first, everything through last amount
        body2 = "First £25.50 here. More text. Second £25.50 at end."
        ctx2 = CategoryPredictor._extract_body_context_around_amount(body2, Decimal("25.50"))
        assert ctx2.endswith("25.50")
        assert "First" in ctx2 and "Second" in ctx2 and "More text" in ctx2
        assert "at end" not in ctx2
        # When amount is 0, return first 500 chars only (avoid matching "0" everywhere)
        long_body = "A" * 400 + " 0 " + "B" * 400
        ctx0 = CategoryPredictor._extract_body_context_around_amount(long_body, Decimal("0"))
        assert len(ctx0) == 500
        assert ctx0 == long_body[:500]

    def test_format_categories_compact_filters_and_groups(self) -> None:
        """Compact formatter keeps only Expenses/Income and groups by prefix to save tokens."""
        paths = [
            "Assets:Current Account",
            "Expenses:Food",
            "Expenses:Food:Groceries",
            "Expenses:Transport",
            "Income:Salary",
            "Liabilities:Credit Card",
        ]
        lines, allowed = CategoryPredictor._format_categories_compact(paths)
        assert "Assets" not in allowed
        assert "Liabilities" not in allowed
        assert "Expenses:Food" in allowed
        assert "Expenses:Food:Groceries" in allowed
        assert "Income:Salary" in allowed
        assert "Expenses:" in "\n".join(lines)
        assert "Income:" in "\n".join(lines)
        assert "  Food" in "\n".join(lines)
        assert "  Food:Groceries" in "\n".join(lines)
        assert "  Salary" in "\n".join(lines)

    def test_llm_timeout_uses_ml_fallback(self) -> None:
        """When the LLM request times out (e.g. step1), we log and return ML fallback instead of raising."""
        llm_cfg = LlmConfig(
            mode=LlmMode.ONLINE,
            endpoint="http://fake:1234/v1/chat/completions",
            model_name="test",
            timeout_seconds=90,
        )
        predictor = CategoryPredictor(llm_config=llm_cfg)
        tx = _make_target_tx(desc="Unknown purchase")
        with patch("gnc_enrich.llm.client.LlmClient.chat") as mock_chat:
            mock_chat.side_effect = requests.exceptions.ReadTimeout("Read timed out")
            proposal = predictor.propose(tx, [], None, account_paths=["Expenses:Food"])
            # No emails: extract-from-description then step1; both time out → ML fallback
            assert mock_chat.call_count == 2
            assert proposal is not None
            assert "LLM suggestion" not in proposal.rationale
            assert proposal.suggested_splits[0].account_path == "Expenses:Miscellaneous"

    def test_single_transaction_does_not_crash(self) -> None:
        """< 2 categorized transactions → falls back to heuristic."""
        txs = [
            Transaction(
                tx_id="t1", posted_date=date(2025, 1, 1),
                description="Tesco", currency="GBP", amount=Decimal("10"),
                splits=[Split(account_path="Expenses:Food", amount=Decimal("10"))],
                original_category="Expenses:Food",
            ),
        ]
        predictor = CategoryPredictor(historical_transactions=txs)
        tx = _make_target_tx(desc="Aldi Store")
        proposal = predictor.propose(tx, [], None)
        assert proposal.confidence > 0

    def test_single_category_does_not_crash(self) -> None:
        """All history in one category → falls back, doesn't crash SGDClassifier."""
        txs = []
        for i in range(5):
            txs.append(Transaction(
                tx_id=f"t{i}", posted_date=date(2025, 1, i + 1),
                description=f"Tesco {i}", currency="GBP", amount=Decimal("10"),
                splits=[Split(account_path="Expenses:Food", amount=Decimal("10"))],
                original_category="Expenses:Food",
            ))
        predictor = CategoryPredictor(historical_transactions=txs)
        tx = _make_target_tx(desc="Sainsburys")
        proposal = predictor.propose(tx, [], None)
        assert proposal.confidence > 0

    def test_refund_detection_none_account_name(self) -> None:
        """Transactions with empty account_name should skip refund detection."""
        predictor = CategoryPredictor(historical_transactions=_make_history())
        tx = _make_target_tx(desc="Refund from Store")
        tx.account_name = ""
        proposal = predictor.propose(tx, [], None)
        assert "Refund detected" not in proposal.rationale

    def test_run_llm_check_disabled_returns_none(self) -> None:
        """run_llm_check returns None when LLM mode is DISABLED."""
        predictor = CategoryPredictor(llm_config=LlmConfig(mode=LlmMode.DISABLED))
        tx = _make_target_tx()
        emails: list[EmailEvidence] = []
        result = predictor.run_llm_check(tx, emails, None, ["Expenses:Food"])
        assert result is None

    def test_run_llm_check_returns_dict_when_flow_succeeds(self) -> None:
        """run_llm_check returns dict with extraction, category, description, confidence when _run_llm_flow returns."""
        predictor = CategoryPredictor(llm_config=LlmConfig(mode=LlmMode.ONLINE, endpoint="http://x", model_name="y"))
        tx = _make_target_tx()
        with patch.object(predictor, "_run_llm_flow", return_value={
            "extraction": {"seller_name": "Shop", "items": [{"description": "Item", "amount": "25"}]},
            "category": "Expenses:Food",
            "description": "Shop order 123",
            "confidence": 0.8,
        }):
            result = predictor.run_llm_check(tx, [], None, ["Expenses:Food"])
        assert result is not None
        assert result["category"] == "Expenses:Food"
        assert result["description"] == "Shop order 123"
        assert result["confidence"] == 0.8
        assert result["extraction"]["seller_name"] == "Shop"

    def test_get_emails_for_display_empty_list_returns_empty(self) -> None:
        """get_emails_for_display with no emails returns [] (robustness)."""
        assert CategoryPredictor.get_emails_for_display([], Decimal("10.00")) == []

    def test_get_emails_for_display_deduplicates_by_context(self) -> None:
        """get_emails_for_display returns (email, context) list with 95% similar contexts deduplicated."""
        from gnc_enrich.domain.models import EmailEvidence
        sent = datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        em1 = EmailEvidence(
            evidence_id="e1", message_id="m1", sender="a@b.com", subject="Order",
            sent_at=sent, body_snippet="Your order total £25.00 thank you.",
        )
        em2 = EmailEvidence(
            evidence_id="e2", message_id="m2", sender="a@b.com", subject="Order",
            sent_at=sent, body_snippet="Your order total £25.00 thank you.",
        )
        pairs = CategoryPredictor.get_emails_for_display([em1, em2], Decimal("25.00"))
        assert len(pairs) == 1
        assert pairs[0][0].evidence_id in ("e1", "e2")
        assert "25.00" in pairs[0][1]

    def test_format_extraction_skips_non_dict_items(self) -> None:
        """_format_extraction_for_prompt only formats dict-shaped items; list items are skipped."""
        extraction = {
            "seller_name": "Store",
            "items": [
                {"description": "A", "amount": "10"},
                ["list", "not", "dict"],
                {"description": "B", "amount": "5"},
            ],
        }
        out = CategoryPredictor._format_extraction_for_prompt(extraction)
        assert "Store" in out
        assert "A" in out and "B" in out
        assert "list" not in out

    def test_run_llm_flow_step1_missing_category_uses_fallback(self) -> None:
        """When step1 response omits 'category', fallback is first allowed path (or Expense)."""
        predictor = CategoryPredictor(llm_config=LlmConfig(mode=LlmMode.ONLINE, endpoint="http://x", model_name="y"))
        tx = _make_target_tx()
        with patch.object(predictor, "_query_llm_extract", return_value=None):
            with patch.object(predictor, "_query_llm_step1", return_value={
                "improved_description": "Improved",
                "confidence": 7,
                "confident": False,
                # no "category" key
            }):
                result = predictor._run_llm_flow(tx, [], ["Expenses:Food", "Expenses:Transport"])
        assert result is not None
        assert result["category"] == "Expenses:Food"
        assert result["description"] == "Improved"

    def test_run_llm_flow_step2_missing_category_uses_chosen(self) -> None:
        """When step2 response omits 'category', final category is step1 chosen."""
        predictor = CategoryPredictor(llm_config=LlmConfig(mode=LlmMode.ONLINE, endpoint="http://x", model_name="y"))
        tx = _make_target_tx()
        with patch.object(predictor, "_query_llm_extract", return_value=None):
            with patch.object(predictor, "_query_llm_step1", return_value={
                "improved_description": "Improved",
                "confidence": 8,
                "confident": True,
                "category": "Expenses:Food",
            }):
                with patch.object(predictor, "_query_llm_step2", return_value={
                    "description": "Final desc",
                    # no "category" key
                }):
                    result = predictor._run_llm_flow(tx, [], ["Expenses:Food", "Expenses:Food:Groceries"])
        assert result is not None
        assert result["category"] == "Expenses:Food"
        assert result["description"] == "Final desc"

    def test_run_llm_flow_returns_none_when_no_allowed_paths(self) -> None:
        """When allowed_paths is empty or has no GBP expense/income paths, _run_llm_flow returns None."""
        predictor = CategoryPredictor(llm_config=LlmConfig(mode=LlmMode.ONLINE, endpoint="http://x", model_name="y"))
        tx = _make_target_tx()
        result = predictor._run_llm_flow(tx, [], [])
        assert result is None

    def test_run_llm_flow_returns_none_when_step1_returns_none(self) -> None:
        """When step1 returns None (LLM timeout/invalid), _run_llm_flow returns None."""
        predictor = CategoryPredictor(llm_config=LlmConfig(mode=LlmMode.ONLINE, endpoint="http://x", model_name="y"))
        tx = _make_target_tx()
        with patch.object(predictor, "_query_llm_extract", return_value=None):
            with patch.object(predictor, "_query_llm_step1", return_value=None):
                result = predictor._run_llm_flow(tx, [], ["Expenses:Food"])
        assert result is None

    def test_run_llm_flow_with_no_emails_runs_step1_without_email_block(self) -> None:
        """With no emails, extraction is skipped and step1 gets no Extracted from emails / Emails block."""
        predictor = CategoryPredictor(llm_config=LlmConfig(mode=LlmMode.ONLINE, endpoint="http://x", model_name="y"))
        tx = _make_target_tx()
        with patch.object(predictor, "_query_llm_step1", return_value={
            "improved_description": "Done",
            "confidence": 6,
            "confident": False,
            "category": "Expenses:Food",
        }) as mock_step1:
            result = predictor._run_llm_flow(tx, [], ["Expenses:Food"])
        assert result is not None
        assert result["category"] == "Expenses:Food"
        mock_step1.assert_called_once()
        call_kw = mock_step1.call_args[1]
        assert call_kw.get("emails_sorted") == []
        assert call_kw.get("extracted_from_emails") is None

    def test_run_llm_flow_uses_extracted_block_when_extraction_succeeds(self) -> None:
        """When extraction endpoint is set and extraction returns data, step1 receives extracted_from_emails."""
        predictor = CategoryPredictor(llm_config=LlmConfig(
            mode=LlmMode.ONLINE, endpoint="http://x", model_name="y",
            extraction_endpoint="http://extract", extraction_model="m",
        ))
        tx = _make_target_tx()
        em = EmailEvidence(
            evidence_id="e1", message_id="m1", sender="s@b.com", subject="Order",
            sent_at=datetime(2025, 1, 15, tzinfo=timezone.utc),
            body_snippet="Total £25.00 thanks.",
        )
        with patch.object(predictor, "_query_llm_extract", return_value={
            "seller_name": "Shop", "items": [{"description": "Item", "amount": "25"}],
            "order_ids": ["ORD1"], "transaction_ids": [],
        }):
            with patch.object(predictor, "_query_llm_step1", return_value={
                "improved_description": "Shop order ORD1",
                "confidence": 8,
                "confident": True,
                "category": "Expenses:Food",
            }) as mock_step1:
                result = predictor._run_llm_flow(tx, [em], ["Expenses:Food"])
        assert result is not None
        call_kw = mock_step1.call_args[1]
        assert call_kw.get("extracted_from_emails") is not None
        assert "Shop" in call_kw["extracted_from_emails"]
        assert call_kw.get("emails_sorted") is None

    def test_run_llm_check_returns_none_when_flow_returns_none(self) -> None:
        """run_llm_check returns None when _run_llm_flow returns None (API returns 400, proposal not updated)."""
        predictor = CategoryPredictor(llm_config=LlmConfig(mode=LlmMode.ONLINE, endpoint="http://x", model_name="y"))
        tx = _make_target_tx()
        with patch.object(predictor, "_run_llm_flow", return_value=None):
            result = predictor.run_llm_check(tx, [], None, ["Expenses:Food"])
        assert result is None

    def test_propose_skip_llm_true_does_not_call_run_llm_flow(self) -> None:
        """propose(..., skip_llm=True) does not invoke the LLM (run phase default)."""
        predictor = CategoryPredictor(llm_config=LlmConfig(mode=LlmMode.ONLINE, endpoint="http://x", model_name="y"))
        tx = _make_target_tx(desc="Unknown")
        with patch.object(predictor, "_run_llm_flow") as mock_flow:
            proposal = predictor.propose(tx, [], None, account_paths=["Expenses:Food"], skip_llm=True)
        mock_flow.assert_not_called()
        assert proposal.suggested_splits[0].account_path != ""

    def test_query_llm_returns_llm_suggestion_from_run_llm_flow(self) -> None:
        """_query_llm converts _run_llm_flow dict to LlmSuggestion (used when use_llm_during_run and confidence < 60%)."""
        from gnc_enrich.ml.predictor import LlmSuggestion
        predictor = CategoryPredictor(llm_config=LlmConfig(mode=LlmMode.ONLINE, endpoint="http://x", model_name="y"))
        tx = _make_target_tx()
        with patch.object(predictor, "_run_llm_flow", return_value={
            "extraction": None,
            "category": "Expenses:Transport",
            "description": "Train ticket",
            "confidence": 0.85,
        }):
            out = predictor._query_llm(tx, [], ["Expenses:Food", "Expenses:Transport"])
        assert out is not None
        assert isinstance(out, LlmSuggestion)
        assert out.category == "Expenses:Transport"
        assert out.description == "Train ticket"
        assert out.confidence == 0.85


class TestFeedbackTrainer:

    def test_record_feedback_to_state(self, tmp_path: Path) -> None:
        trainer = FeedbackTrainer(state_dir=tmp_path)
        proposal = Proposal(
            proposal_id="p1",
            tx_id="tx1",
            suggested_description="Test",
            suggested_splits=[Split(account_path="Expenses:Food", amount=Decimal("10.00"))],
            confidence=0.8,
            rationale="test",
        )
        trainer.record_feedback(proposal, accepted=True, note="correct")

        from gnc_enrich.state.repository import StateRepository
        repo = StateRepository(tmp_path)
        feedback = repo.load_feedback()
        assert len(feedback) == 1
        assert feedback[0]["accepted"] is True
        assert feedback[0]["proposal_id"] == "p1"

    def test_record_feedback_without_state_dir(self) -> None:
        trainer = FeedbackTrainer()
        proposal = Proposal(
            proposal_id="p1",
            tx_id="tx1",
            suggested_description="Test",
            suggested_splits=[Split(account_path="Expenses:Food", amount=Decimal("10.00"))],
            confidence=0.8,
            rationale="test",
        )
        trainer.record_feedback(proposal, accepted=False)
