"""Tests for ML category predictor and feedback trainer."""

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

    def test_description_includes_gb_date(self) -> None:
        predictor = CategoryPredictor()
        tx = _make_target_tx(desc="Card Payment")
        proposal = predictor.propose(tx, [], None)
        assert "15/01/2025" in proposal.suggested_description

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
        llm_cfg = LlmConfig(mode=LlmMode.ONLINE, endpoint="http://fake:1234/v1/chat/completions")
        predictor = CategoryPredictor(llm_config=llm_cfg)
        tx = _make_target_tx(desc="Unknown purchase")

        with patch("requests.post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.json.return_value = {
                "choices": [{"message": {"content": "Expenses:Shopping"}}]
            }
            mock_resp.raise_for_status = MagicMock()
            mock_post.return_value = mock_resp

            proposal = predictor.propose(tx, [], None)
            mock_post.assert_called_once()
            assert "LLM suggestion" in proposal.rationale


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
