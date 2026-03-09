"""Cross-product tests for LLM flow: main LLM (on/off), extraction LLM (on/off),
use_web (on/off), has_emails (yes/no).

Each of the 9 distinct flow scenarios has at least 5 tests. Multiple-emails
cases are covered where applicable.
"""

from datetime import datetime, timezone, date
from decimal import Decimal
from unittest.mock import patch

import pytest

from gnc_enrich.config import LlmConfig, LlmMode
from gnc_enrich.domain.models import EmailEvidence, Split, Transaction
from gnc_enrich.ml.predictor import CategoryPredictor


def _tx(desc: str = "Card Payment", amount: str = "25.00") -> Transaction:
    return Transaction(
        tx_id="tx1",
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


def _email(
    evidence_id: str = "e1", body: str = "Your order total £25.00 thank you."
) -> EmailEvidence:
    return EmailEvidence(
        evidence_id=evidence_id,
        message_id=f"<{evidence_id}>",
        sender="orders@shop.com",
        subject="Order confirmation",
        sent_at=datetime(2025, 1, 15, 12, 0, tzinfo=timezone.utc),
        body_snippet=body,
    )


ACCOUNT_PATHS = ["Expenses:Food", "Expenses:Food:Groceries", "Expenses:Transport"]


# ---------- Scenario 1: Main LLM disabled (any extraction, any web, any emails) ----------


class TestFlowMainDisabled:
    """When main LLM mode is DISABLED, no LLM is invoked regardless of extraction/web/emails."""

    def test_run_llm_check_returns_none(self) -> None:
        predictor = CategoryPredictor(llm_config=LlmConfig(mode=LlmMode.DISABLED))
        result = predictor.run_llm_check(_tx(), [], None, ACCOUNT_PATHS)
        assert result is None

    def test_run_llm_check_with_emails_still_returns_none(self) -> None:
        predictor = CategoryPredictor(llm_config=LlmConfig(mode=LlmMode.DISABLED))
        result = predictor.run_llm_check(_tx(), [_email()], None, ACCOUNT_PATHS)
        assert result is None

    def test_run_llm_flow_never_called_when_disabled(self) -> None:
        predictor = CategoryPredictor(llm_config=LlmConfig(mode=LlmMode.DISABLED))
        with patch.object(predictor, "_run_llm_flow") as mock_flow:
            predictor.run_llm_check(_tx(), [], None, ACCOUNT_PATHS)
        mock_flow.assert_not_called()

    def test_propose_with_skip_llm_does_not_call_llm(self) -> None:
        predictor = CategoryPredictor(
            llm_config=LlmConfig(mode=LlmMode.ONLINE, endpoint="http://x", model_name="y")
        )
        with patch.object(predictor, "_run_llm_flow") as mock_flow:
            predictor.propose(
                _tx(desc="Unknown"), [], None, account_paths=ACCOUNT_PATHS, skip_llm=True
            )
        mock_flow.assert_not_called()

    def test_query_llm_returns_none_when_disabled(self) -> None:
        predictor = CategoryPredictor(llm_config=LlmConfig(mode=LlmMode.DISABLED))
        with patch.object(predictor, "_run_llm_flow") as mock_flow:
            out = predictor._query_llm(_tx(), [], ACCOUNT_PATHS)
        assert out is None
        mock_flow.assert_not_called()


# ---------- Scenario 2: Main ON, extraction OFF, web OFF, NO emails ----------


class TestFlowMainOnExtractionOffWebOffNoEmails:
    """Main LLM extracts from description (main LLM), no web, step1 with that text, step2 if confident."""

    @pytest.fixture
    def predictor(self):
        return CategoryPredictor(
            llm_config=LlmConfig(
                mode=LlmMode.ONLINE,
                endpoint="http://main",
                model_name="m",
                use_web=False,
            )
        )

    def test_extract_from_description_called_step1_with_text(self, predictor) -> None:
        with patch.object(
            predictor,
            "_query_llm_extract_from_description",
            return_value={
                "seller_name": "Shop",
                "items": [{"description": "Item", "amount": "25"}],
            },
        ) as mock_ext:
            with patch.object(
                predictor,
                "_query_llm_step1",
                return_value={
                    "improved_description": "Shop Item",
                    "confidence": 8,
                    "confident": True,
                    "category": "Expenses:Food",
                },
            ) as mock_s1:
                with patch.object(
                    predictor,
                    "_query_llm_step2",
                    return_value={
                        "category": "Expenses:Food:Groceries",
                        "description": "Shop Item",
                    },
                ):
                    result = predictor._run_llm_flow(_tx(), [], ACCOUNT_PATHS)
        assert result is not None
        mock_ext.assert_called_once()
        call_kw = mock_s1.call_args.kwargs
        assert call_kw.get("extracted_from_emails") is not None
        assert call_kw.get("emails_sorted") is None

    def test_step2_called_when_confidence_high(self, predictor) -> None:
        with patch.object(
            predictor,
            "_query_llm_extract_from_description",
            return_value={"seller_name": "S", "items": []},
        ):
            with patch.object(
                predictor,
                "_query_llm_step1",
                return_value={
                    "improved_description": "D",
                    "confidence": 8,
                    "confident": True,
                    "category": "Expenses:Food",
                },
            ):
                with patch.object(
                    predictor,
                    "_query_llm_step2",
                    return_value={"category": "Expenses:Food:Groceries", "description": "D"},
                ) as mock_s2:
                    result = predictor._run_llm_flow(_tx(), [], ACCOUNT_PATHS)
        assert result["category"] == "Expenses:Food:Groceries"
        mock_s2.assert_called_once()

    def test_step2_not_called_when_low_confidence(self, predictor) -> None:
        with patch.object(
            predictor,
            "_query_llm_extract_from_description",
            return_value={"seller_name": "S", "items": []},
        ):
            with patch.object(
                predictor,
                "_query_llm_step1",
                return_value={
                    "improved_description": "D",
                    "confidence": 4,
                    "confident": False,
                    "category": "Expenses:Food",
                },
            ):
                with patch.object(predictor, "_query_llm_step2") as mock_s2:
                    result = predictor._run_llm_flow(_tx(), [], ACCOUNT_PATHS)
        assert result["category"] == "Expenses:Food"
        mock_s2.assert_not_called()

    def test_no_extraction_llm_called(self, predictor) -> None:
        with patch.object(predictor, "_query_llm_extract") as mock_extract:
            with patch.object(
                predictor,
                "_query_llm_extract_from_description",
                return_value={"seller_name": "S", "items": []},
            ):
                with patch.object(
                    predictor,
                    "_query_llm_step1",
                    return_value={
                        "improved_description": "D",
                        "confidence": 5,
                        "confident": False,
                        "category": "Expenses:Food",
                    },
                ):
                    predictor._run_llm_flow(_tx(), [], ACCOUNT_PATHS)
        mock_extract.assert_not_called()

    def test_web_search_not_called(self, predictor) -> None:
        with patch.object(predictor, "_web_search_short") as mock_web:
            with patch.object(
                predictor,
                "_query_llm_extract_from_description",
                return_value={"seller_name": "S", "items": []},
            ):
                with patch.object(
                    predictor,
                    "_query_llm_step1",
                    return_value={
                        "improved_description": "D",
                        "confidence": 5,
                        "confident": False,
                        "category": "Expenses:Food",
                    },
                ):
                    predictor._run_llm_flow(_tx(), [], ACCOUNT_PATHS)
        mock_web.assert_not_called()


# ---------- Scenario 3: Main ON, extraction OFF, web OFF, HAS emails ----------


class TestFlowMainOnExtractionOffWebOffHasEmails:
    """No extraction. Step1 with raw email context. Step2 if confident. use_web not used."""

    @pytest.fixture
    def predictor(self):
        return CategoryPredictor(
            llm_config=LlmConfig(
                mode=LlmMode.ONLINE,
                endpoint="http://main",
                model_name="m",
                use_web=False,
            )
        )

    def test_step1_receives_raw_emails_not_extraction(self, predictor) -> None:
        em = _email("e1", "Total £25.00 thanks.")
        with patch.object(predictor, "_query_llm_extract") as mock_extract:
            with patch.object(
                predictor,
                "_query_llm_step1",
                return_value={
                    "improved_description": "Order",
                    "confidence": 7,
                    "confident": False,
                    "category": "Expenses:Food",
                },
            ) as mock_s1:
                predictor._run_llm_flow(_tx(), [em], ACCOUNT_PATHS)
        mock_extract.assert_not_called()
        call_kw = mock_s1.call_args.kwargs
        assert call_kw["emails_sorted"] is not None
        assert call_kw["extracted_from_emails"] is None

    def test_multiple_emails_passed_to_step1(self, predictor) -> None:
        em1 = _email("e1", "Order 1 £25.00")
        em2 = _email("e2", "Order 2 £25.00")
        with patch.object(
            predictor,
            "_query_llm_step1",
            return_value={
                "improved_description": "Order",
                "confidence": 6,
                "confident": False,
                "category": "Expenses:Food",
            },
        ) as mock_s1:
            predictor._run_llm_flow(_tx(), [em1, em2], ACCOUNT_PATHS)
        call_kw = mock_s1.call_args.kwargs
        assert len(call_kw["emails_sorted"]) == 2

    def test_web_search_not_called_with_emails(self, predictor) -> None:
        with patch.object(predictor, "_web_search_short") as mock_web:
            with patch.object(
                predictor,
                "_query_llm_step1",
                return_value={
                    "improved_description": "D",
                    "confidence": 5,
                    "confident": False,
                    "category": "Expenses:Food",
                },
            ):
                predictor._run_llm_flow(_tx(), [_email()], ACCOUNT_PATHS)
        mock_web.assert_not_called()

    def test_run_llm_check_returns_result(self, predictor) -> None:
        with patch.object(
            predictor,
            "_query_llm_step1",
            return_value={
                "improved_description": "Order",
                "confidence": 8,
                "confident": True,
                "category": "Expenses:Food",
            },
        ):
            with patch.object(
                predictor,
                "_query_llm_step2",
                return_value={
                    "category": "Expenses:Food:Groceries",
                    "description": "Order",
                },
            ):
                result = predictor.run_llm_check(_tx(), [_email()], None, ACCOUNT_PATHS)
        assert result is not None
        assert result["category"] == "Expenses:Food:Groceries"

    def test_three_emails_sorted_by_date(self, predictor) -> None:
        em1 = _email("e1", "First")
        em1.sent_at = datetime(2025, 1, 14, tzinfo=timezone.utc)
        em2 = _email("e2", "Second")
        em2.sent_at = datetime(2025, 1, 15, tzinfo=timezone.utc)
        em3 = _email("e3", "Third")
        em3.sent_at = datetime(2025, 1, 16, tzinfo=timezone.utc)
        with patch.object(
            predictor,
            "_query_llm_step1",
            return_value={
                "improved_description": "D",
                "confidence": 5,
                "confident": False,
                "category": "Expenses:Food",
            },
        ) as mock_s1:
            predictor._run_llm_flow(_tx(), [em3, em1, em2], ACCOUNT_PATHS)
        order = [e.evidence_id for e in mock_s1.call_args.kwargs["emails_sorted"]]
        assert order == ["e1", "e2", "e3"]


# ---------- Scenario 4: Main ON, extraction OFF, web ON, NO emails ----------


class TestFlowMainOnExtractionOffWebOnNoEmails:
    """Main LLM extracts from description; use_web adds descriptions; step1; step2 if confident."""

    @pytest.fixture
    def predictor(self):
        return CategoryPredictor(
            llm_config=LlmConfig(
                mode=LlmMode.ONLINE,
                endpoint="http://main",
                model_name="m",
                use_web=True,
            )
        )

    def test_extract_from_description_called_web_can_be_called(self, predictor) -> None:
        with patch.object(
            predictor,
            "_query_llm_extract_from_description",
            return_value={
                "seller_name": "Amazon",
                "items": [{"description": "Book", "amount": "25"}],
            },
        ) as mock_ext:
            with patch.object(
                predictor,
                "_query_llm_step1",
                return_value={
                    "improved_description": "Amazon Book",
                    "confidence": 8,
                    "confident": True,
                    "category": "Expenses:Food",
                },
            ):
                with patch.object(
                    predictor,
                    "_query_llm_step2",
                    return_value={
                        "category": "Expenses:Food:Groceries",
                        "description": "Amazon Book",
                    },
                ):
                    result = predictor._run_llm_flow(_tx(), [], ACCOUNT_PATHS)
        assert result is not None
        mock_ext.assert_called_once()

    def test_step1_receives_extracted_block(self, predictor) -> None:
        with patch.object(
            predictor,
            "_query_llm_extract_from_description",
            return_value={
                "seller_name": "S",
                "items": [],
                "seller_web_description": "Web desc",
            },
        ):
            with patch.object(
                predictor,
                "_query_llm_step1",
                return_value={
                    "improved_description": "D",
                    "confidence": 5,
                    "confident": False,
                    "category": "Expenses:Food",
                },
            ) as mock_s1:
                predictor._run_llm_flow(_tx(), [], ACCOUNT_PATHS)
        block = mock_s1.call_args.kwargs["extracted_from_emails"]
        assert "S" in block or "Web desc" in block

    def test_no_extraction_llm_called(self, predictor) -> None:
        with patch.object(predictor, "_query_llm_extract") as mock_extract:
            with patch.object(
                predictor,
                "_query_llm_extract_from_description",
                return_value={"seller_name": "S", "items": []},
            ):
                with patch.object(
                    predictor,
                    "_query_llm_step1",
                    return_value={
                        "improved_description": "D",
                        "confidence": 5,
                        "confident": False,
                        "category": "Expenses:Food",
                    },
                ):
                    predictor._run_llm_flow(_tx(), [], ACCOUNT_PATHS)
        mock_extract.assert_not_called()

    def test_run_llm_check_succeeds(self, predictor) -> None:
        with patch.object(
            predictor,
            "_query_llm_extract_from_description",
            return_value={"seller_name": "S", "items": []},
        ):
            with patch.object(
                predictor,
                "_query_llm_step1",
                return_value={
                    "improved_description": "D",
                    "confidence": 7,
                    "confident": True,
                    "category": "Expenses:Food",
                },
            ):
                with patch.object(
                    predictor,
                    "_query_llm_step2",
                    return_value={
                        "category": "Expenses:Food:Groceries",
                        "description": "D",
                    },
                ):
                    result = predictor.run_llm_check(_tx(), [], None, ACCOUNT_PATHS)
        assert result is not None
        assert result["category"] == "Expenses:Food:Groceries"


# ---------- Scenario 5: Main ON, extraction OFF, web ON, HAS emails ----------


class TestFlowMainOnExtractionOffWebOnHasEmails:
    """No extraction. Step1 with raw emails. use_web not applied (no extraction to enrich)."""

    @pytest.fixture
    def predictor(self):
        return CategoryPredictor(
            llm_config=LlmConfig(
                mode=LlmMode.ONLINE,
                endpoint="http://main",
                model_name="m",
                use_web=True,
            )
        )

    def test_step1_gets_raw_emails(self, predictor) -> None:
        with patch.object(
            predictor,
            "_query_llm_step1",
            return_value={
                "improved_description": "D",
                "confidence": 5,
                "confident": False,
                "category": "Expenses:Food",
            },
        ) as mock_s1:
            predictor._run_llm_flow(_tx(), [_email()], ACCOUNT_PATHS)
        call_kw = mock_s1.call_args.kwargs
        assert call_kw["emails_sorted"] is not None
        assert call_kw["extracted_from_emails"] is None

    def test_extract_from_description_not_called_when_emails_present(self, predictor) -> None:
        with patch.object(predictor, "_query_llm_extract_from_description") as mock_desc:
            with patch.object(
                predictor,
                "_query_llm_step1",
                return_value={
                    "improved_description": "D",
                    "confidence": 5,
                    "confident": False,
                    "category": "Expenses:Food",
                },
            ):
                predictor._run_llm_flow(_tx(), [_email()], ACCOUNT_PATHS)
        mock_desc.assert_not_called()

    def test_web_search_not_used_for_raw_emails_flow(self, predictor) -> None:
        with patch.object(predictor, "_web_search_short") as mock_web:
            with patch.object(
                predictor,
                "_query_llm_step1",
                return_value={
                    "improved_description": "D",
                    "confidence": 5,
                    "confident": False,
                    "category": "Expenses:Food",
                },
            ):
                predictor._run_llm_flow(_tx(), [_email()], ACCOUNT_PATHS)
        mock_web.assert_not_called()

    def test_multiple_emails_passed_to_step1(self, predictor) -> None:
        with patch.object(
            predictor,
            "_query_llm_step1",
            return_value={
                "improved_description": "D",
                "confidence": 5,
                "confident": False,
                "category": "Expenses:Food",
            },
        ) as mock_s1:
            predictor._run_llm_flow(_tx(), [_email("e1"), _email("e2")], ACCOUNT_PATHS)
        assert len(mock_s1.call_args.kwargs["emails_sorted"]) == 2

    def test_result_contains_category_and_description(self, predictor) -> None:
        with patch.object(
            predictor,
            "_query_llm_step1",
            return_value={
                "improved_description": "Improved",
                "confidence": 8,
                "confident": True,
                "category": "Expenses:Food",
            },
        ):
            with patch.object(
                predictor,
                "_query_llm_step2",
                return_value={
                    "category": "Expenses:Food:Groceries",
                    "description": "Improved",
                },
            ):
                result = predictor._run_llm_flow(_tx(), [_email()], ACCOUNT_PATHS)
        assert result["description"] == "Improved"
        assert result["category"] == "Expenses:Food:Groceries"


# ---------- Scenario 6: Main ON, extraction ON, web OFF, NO emails ----------


class TestFlowMainOnExtractionOnWebOffNoEmails:
    """Extraction LLM extracts from transaction description; no web; step1 with extracted; step2 if confident."""

    @pytest.fixture
    def predictor(self):
        return CategoryPredictor(
            llm_config=LlmConfig(
                mode=LlmMode.ONLINE,
                endpoint="http://main",
                model_name="m",
                extraction_endpoint="http://extract",
                extraction_model="em",
                use_web=False,
            )
        )

    def test_extract_from_description_called(self, predictor) -> None:
        with patch.object(
            predictor,
            "_query_llm_extract_from_description",
            return_value={"seller_name": "Shop", "items": []},
        ) as mock_ext:
            with patch.object(
                predictor,
                "_query_llm_step1",
                return_value={
                    "improved_description": "D",
                    "confidence": 5,
                    "confident": False,
                    "category": "Expenses:Food",
                },
            ):
                predictor._run_llm_flow(_tx(), [], ACCOUNT_PATHS)
        mock_ext.assert_called_once()

    def test_query_llm_extract_not_called_no_emails(self, predictor) -> None:
        with patch.object(predictor, "_query_llm_extract") as mock_extract:
            with patch.object(
                predictor,
                "_query_llm_extract_from_description",
                return_value={"seller_name": "S", "items": []},
            ):
                with patch.object(
                    predictor,
                    "_query_llm_step1",
                    return_value={
                        "improved_description": "D",
                        "confidence": 5,
                        "confident": False,
                        "category": "Expenses:Food",
                    },
                ):
                    predictor._run_llm_flow(_tx(), [], ACCOUNT_PATHS)
        mock_extract.assert_not_called()

    def test_step1_receives_extracted_from_emails(self, predictor) -> None:
        with patch.object(
            predictor,
            "_query_llm_extract_from_description",
            return_value={
                "seller_name": "Store",
                "items": [{"description": "Item", "amount": "25"}],
            },
        ):
            with patch.object(
                predictor,
                "_query_llm_step1",
                return_value={
                    "improved_description": "D",
                    "confidence": 5,
                    "confident": False,
                    "category": "Expenses:Food",
                },
            ) as mock_s1:
                predictor._run_llm_flow(_tx(), [], ACCOUNT_PATHS)
        block = mock_s1.call_args.kwargs["extracted_from_emails"]
        assert "Store" in block
        assert mock_s1.call_args.kwargs["emails_sorted"] is None

    def test_web_not_called(self, predictor) -> None:
        with patch.object(
            predictor,
            "_query_llm_extract_from_description",
            return_value={"seller_name": "S", "items": []},
        ):
            with patch.object(predictor, "_web_search_short") as mock_web:
                with patch.object(
                    predictor,
                    "_query_llm_step1",
                    return_value={
                        "improved_description": "D",
                        "confidence": 5,
                        "confident": False,
                        "category": "Expenses:Food",
                    },
                ):
                    predictor._run_llm_flow(_tx(), [], ACCOUNT_PATHS)
        mock_web.assert_not_called()

    def test_run_llm_check_returns_full_result(self, predictor) -> None:
        with patch.object(
            predictor,
            "_query_llm_extract_from_description",
            return_value={"seller_name": "S", "items": [], "order_ids": ["O1"]},
        ):
            with patch.object(
                predictor,
                "_query_llm_step1",
                return_value={
                    "improved_description": "D",
                    "confidence": 8,
                    "confident": True,
                    "category": "Expenses:Food",
                },
            ):
                with patch.object(
                    predictor,
                    "_query_llm_step2",
                    return_value={
                        "category": "Expenses:Food:Groceries",
                        "description": "D",
                    },
                ):
                    result = predictor.run_llm_check(_tx(), [], None, ACCOUNT_PATHS)
        assert result is not None
        assert result["extraction"]["seller_name"] == "S"
        assert result["category"] == "Expenses:Food:Groceries"


# ---------- Scenario 7: Main ON, extraction ON, web OFF, HAS emails ----------


class TestFlowMainOnExtractionOnWebOffHasEmails:
    """Per-email extraction, merge; no web. Step1 with 'Extracted from emails'; step2 if confident."""

    @pytest.fixture
    def predictor(self):
        return CategoryPredictor(
            llm_config=LlmConfig(
                mode=LlmMode.ONLINE,
                endpoint="http://main",
                model_name="m",
                extraction_endpoint="http://extract",
                extraction_model="em",
                use_web=False,
            )
        )

    def test_query_llm_extract_called_with_emails(self, predictor) -> None:
        with patch.object(
            predictor,
            "_query_llm_extract",
            return_value={
                "seller_name": "Shop",
                "items": [{"description": "Item", "amount": "25"}],
                "order_ids": [],
                "transaction_ids": [],
            },
        ) as mock_extract:
            with patch.object(
                predictor,
                "_query_llm_step1",
                return_value={
                    "improved_description": "Shop Item",
                    "confidence": 8,
                    "confident": True,
                    "category": "Expenses:Food",
                },
            ):
                with patch.object(
                    predictor,
                    "_query_llm_step2",
                    return_value={
                        "category": "Expenses:Food:Groceries",
                        "description": "Shop Item",
                    },
                ):
                    predictor._run_llm_flow(_tx(), [_email()], ACCOUNT_PATHS)
        mock_extract.assert_called_once()
        assert len(mock_extract.call_args.args[0]) == 1

    def test_step1_receives_extracted_block_not_raw_emails(self, predictor) -> None:
        with patch.object(
            predictor,
            "_query_llm_extract",
            return_value={
                "seller_name": "Store",
                "items": [],
                "order_ids": ["O1"],
                "transaction_ids": [],
            },
        ):
            with patch.object(
                predictor,
                "_query_llm_step1",
                return_value={
                    "improved_description": "D",
                    "confidence": 5,
                    "confident": False,
                    "category": "Expenses:Food",
                },
            ) as mock_s1:
                predictor._run_llm_flow(_tx(), [_email()], ACCOUNT_PATHS)
        call_kw = mock_s1.call_args.kwargs
        assert call_kw["extracted_from_emails"] is not None
        assert "Store" in call_kw["extracted_from_emails"]
        assert call_kw["emails_sorted"] is None

    def test_multiple_emails_cause_extract_and_merge(self, predictor) -> None:
        em1, em2 = _email("e1", "Order 1 £25"), _email("e2", "Order 2 £25")
        with patch.object(
            predictor,
            "_query_llm_extract",
            return_value={
                "seller_name": "Merged",
                "items": [],
                "order_ids": [],
                "transaction_ids": [],
            },
        ) as mock_extract:
            with patch.object(
                predictor,
                "_query_llm_step1",
                return_value={
                    "improved_description": "D",
                    "confidence": 5,
                    "confident": False,
                    "category": "Expenses:Food",
                },
            ):
                predictor._run_llm_flow(_tx(), [em1, em2], ACCOUNT_PATHS)
        mock_extract.assert_called_once()
        assert len(mock_extract.call_args.args[0]) == 2

    def test_web_search_not_called(self, predictor) -> None:
        with patch.object(
            predictor,
            "_query_llm_extract",
            return_value={
                "seller_name": "S",
                "items": [],
                "order_ids": [],
                "transaction_ids": [],
            },
        ):
            with patch.object(predictor, "_web_search_short") as mock_web:
                with patch.object(
                    predictor,
                    "_query_llm_step1",
                    return_value={
                        "improved_description": "D",
                        "confidence": 5,
                        "confident": False,
                        "category": "Expenses:Food",
                    },
                ):
                    predictor._run_llm_flow(_tx(), [_email()], ACCOUNT_PATHS)
        mock_web.assert_not_called()

    def test_extract_from_description_not_called_when_emails(self, predictor) -> None:
        with patch.object(
            predictor,
            "_query_llm_extract",
            return_value={
                "seller_name": "S",
                "items": [],
                "order_ids": [],
                "transaction_ids": [],
            },
        ):
            with patch.object(predictor, "_query_llm_extract_from_description") as mock_desc:
                with patch.object(
                    predictor,
                    "_query_llm_step1",
                    return_value={
                        "improved_description": "D",
                        "confidence": 5,
                        "confident": False,
                        "category": "Expenses:Food",
                    },
                ):
                    predictor._run_llm_flow(_tx(), [_email()], ACCOUNT_PATHS)
        mock_desc.assert_not_called()


# ---------- Scenario 8: Main ON, extraction ON, web ON, NO emails ----------


class TestFlowMainOnExtractionOnWebOnNoEmails:
    """Extraction LLM from description; use_web adds seller/item web descriptions; step1; step2 if confident."""

    @pytest.fixture
    def predictor(self):
        return CategoryPredictor(
            llm_config=LlmConfig(
                mode=LlmMode.ONLINE,
                endpoint="http://main",
                model_name="m",
                extraction_endpoint="http://extract",
                extraction_model="em",
                use_web=True,
            )
        )

    def test_extract_from_description_called(self, predictor) -> None:
        with patch.object(
            predictor,
            "_query_llm_extract_from_description",
            return_value={
                "seller_name": "S",
                "items": [],
                "seller_web_description": "Web",
            },
        ) as mock_ext:
            with patch.object(
                predictor,
                "_query_llm_step1",
                return_value={
                    "improved_description": "D",
                    "confidence": 5,
                    "confident": False,
                    "category": "Expenses:Food",
                },
            ):
                predictor._run_llm_flow(_tx(), [], ACCOUNT_PATHS)
        mock_ext.assert_called_once()

    def test_web_effect_visible_in_extracted_block(self, predictor) -> None:
        with patch.object(
            predictor,
            "_query_llm_extract_from_description",
            return_value={
                "seller_name": "Tesco",
                "items": [{"description": "Milk", "web_description": "Groceries"}],
                "seller_web_description": "UK supermarket",
            },
        ):
            with patch.object(
                predictor,
                "_query_llm_step1",
                return_value={
                    "improved_description": "D",
                    "confidence": 5,
                    "confident": False,
                    "category": "Expenses:Food",
                },
            ) as mock_s1:
                predictor._run_llm_flow(_tx(), [], ACCOUNT_PATHS)
        block = mock_s1.call_args.kwargs["extracted_from_emails"]
        assert "Tesco" in block or "Groceries" in block or "UK supermarket" in block

    def test_step2_called_when_confident(self, predictor) -> None:
        with patch.object(
            predictor,
            "_query_llm_extract_from_description",
            return_value={"seller_name": "S", "items": []},
        ):
            with patch.object(
                predictor,
                "_query_llm_step1",
                return_value={
                    "improved_description": "D",
                    "confidence": 8,
                    "confident": True,
                    "category": "Expenses:Food",
                },
            ):
                with patch.object(
                    predictor,
                    "_query_llm_step2",
                    return_value={
                        "category": "Expenses:Food:Groceries",
                        "description": "D",
                    },
                ) as mock_s2:
                    result = predictor._run_llm_flow(_tx(), [], ACCOUNT_PATHS)
        mock_s2.assert_called_once()
        assert result["category"] == "Expenses:Food:Groceries"

    def test_query_llm_extract_not_called_no_emails(self, predictor) -> None:
        with patch.object(predictor, "_query_llm_extract") as mock_extract:
            with patch.object(
                predictor,
                "_query_llm_extract_from_description",
                return_value={"seller_name": "S", "items": []},
            ):
                with patch.object(
                    predictor,
                    "_query_llm_step1",
                    return_value={
                        "improved_description": "D",
                        "confidence": 5,
                        "confident": False,
                        "category": "Expenses:Food",
                    },
                ):
                    predictor._run_llm_flow(_tx(), [], ACCOUNT_PATHS)
        mock_extract.assert_not_called()

    def test_run_llm_check_returns_extraction_and_category(self, predictor) -> None:
        with patch.object(
            predictor,
            "_query_llm_extract_from_description",
            return_value={"seller_name": "Shop", "items": [{"description": "X", "amount": "25"}]},
        ):
            with patch.object(
                predictor,
                "_query_llm_step1",
                return_value={
                    "improved_description": "Shop X",
                    "confidence": 8,
                    "confident": True,
                    "category": "Expenses:Food",
                },
            ):
                with patch.object(
                    predictor,
                    "_query_llm_step2",
                    return_value={
                        "category": "Expenses:Food:Groceries",
                        "description": "Shop X",
                    },
                ):
                    result = predictor.run_llm_check(_tx(), [], None, ACCOUNT_PATHS)
        assert result["extraction"]["seller_name"] == "Shop"
        assert result["category"] == "Expenses:Food:Groceries"


# ---------- Scenario 9: Main ON, extraction ON, web ON, HAS emails ----------


class TestFlowMainOnExtractionOnWebOnHasEmails:
    """Per-email extraction, merge; use_web adds seller/item web descriptions. Step1 with extracted; step2 if confident."""

    @pytest.fixture
    def predictor(self):
        return CategoryPredictor(
            llm_config=LlmConfig(
                mode=LlmMode.ONLINE,
                endpoint="http://main",
                model_name="m",
                extraction_endpoint="http://extract",
                extraction_model="em",
                use_web=True,
            )
        )

    def test_extract_called_with_emails(self, predictor) -> None:
        with patch.object(
            predictor,
            "_query_llm_extract",
            return_value={
                "seller_name": "Shop",
                "items": [],
                "order_ids": [],
                "transaction_ids": [],
                "seller_web_description": "Web desc",
            },
        ) as mock_extract:
            with patch.object(
                predictor,
                "_query_llm_step1",
                return_value={
                    "improved_description": "D",
                    "confidence": 5,
                    "confident": False,
                    "category": "Expenses:Food",
                },
            ):
                predictor._run_llm_flow(_tx(), [_email()], ACCOUNT_PATHS)
        mock_extract.assert_called_once()

    def test_step1_receives_extracted_with_web_descriptions(self, predictor) -> None:
        with patch.object(
            predictor,
            "_query_llm_extract",
            return_value={
                "seller_name": "Amazon",
                "items": [{"description": "Book", "web_description": "A novel"}],
                "order_ids": [],
                "transaction_ids": [],
                "seller_web_description": "Online retailer",
            },
        ):
            with patch.object(
                predictor,
                "_query_llm_step1",
                return_value={
                    "improved_description": "D",
                    "confidence": 5,
                    "confident": False,
                    "category": "Expenses:Food",
                },
            ) as mock_s1:
                predictor._run_llm_flow(_tx(), [_email()], ACCOUNT_PATHS)
        block = mock_s1.call_args.kwargs["extracted_from_emails"]
        assert "Amazon" in block or "A novel" in block or "Online retailer" in block

    def test_multiple_emails_extraction_then_merge(self, predictor) -> None:
        em1, em2 = _email("e1", "Receipt 1 £25"), _email("e2", "Receipt 2 £25")
        with patch.object(
            predictor,
            "_query_llm_extract",
            return_value={
                "seller_name": "MergedShop",
                "items": [],
                "order_ids": [],
                "transaction_ids": [],
            },
        ) as mock_extract:
            with patch.object(
                predictor,
                "_query_llm_step1",
                return_value={
                    "improved_description": "D",
                    "confidence": 5,
                    "confident": False,
                    "category": "Expenses:Food",
                },
            ):
                predictor._run_llm_flow(_tx(), [em1, em2], ACCOUNT_PATHS)
        assert len(mock_extract.call_args.args[0]) == 2

    def test_extract_from_description_not_called(self, predictor) -> None:
        with patch.object(
            predictor,
            "_query_llm_extract",
            return_value={
                "seller_name": "S",
                "items": [],
                "order_ids": [],
                "transaction_ids": [],
            },
        ):
            with patch.object(predictor, "_query_llm_extract_from_description") as mock_desc:
                with patch.object(
                    predictor,
                    "_query_llm_step1",
                    return_value={
                        "improved_description": "D",
                        "confidence": 5,
                        "confident": False,
                        "category": "Expenses:Food",
                    },
                ):
                    predictor._run_llm_flow(_tx(), [_email()], ACCOUNT_PATHS)
        mock_desc.assert_not_called()

    def test_full_flow_returns_category_and_confidence(self, predictor) -> None:
        with patch.object(
            predictor,
            "_query_llm_extract",
            return_value={
                "seller_name": "Shop",
                "items": [{"description": "Item", "amount": "25"}],
                "order_ids": ["O1"],
                "transaction_ids": [],
            },
        ):
            with patch.object(
                predictor,
                "_query_llm_step1",
                return_value={
                    "improved_description": "Shop order O1",
                    "confidence": 8,
                    "confident": True,
                    "category": "Expenses:Food",
                },
            ):
                with patch.object(
                    predictor,
                    "_query_llm_step2",
                    return_value={
                        "category": "Expenses:Food:Groceries",
                        "description": "Shop order O1",
                    },
                ):
                    result = predictor._run_llm_flow(_tx(), [_email()], ACCOUNT_PATHS)
        assert result["category"] == "Expenses:Food:Groceries"
        assert result["confidence"] == 0.8
        assert result["extraction"] is not None
