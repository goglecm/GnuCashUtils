"""Tests for the enrichment pipeline orchestration."""

import gzip
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

from gnc_enrich.config import LlmConfig, LlmMode, RunConfig
from gnc_enrich.services.pipeline import EnrichmentPipeline, _test_llm_connection
from gnc_enrich.state.repository import StateRepository
from tests.conftest import SAMPLE_GNUCASH_XML

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _setup_pipeline_dirs(tmp_path: Path) -> dict[str, Path]:
    gnucash = tmp_path / "book.gnucash"
    with gzip.open(gnucash, "wb") as f:
        f.write(SAMPLE_GNUCASH_XML.encode("utf-8"))

    emails = FIXTURES_DIR / "emails"

    receipts = tmp_path / "receipts"
    receipts.mkdir()

    processed = tmp_path / "processed"
    state = tmp_path / "state"

    return {
        "gnucash": gnucash,
        "emails": emails,
        "receipts": receipts,
        "processed": processed,
        "state": state,
    }


def _make_config(dirs: dict[str, Path]) -> RunConfig:
    return RunConfig(
        gnucash_path=dirs["gnucash"],
        emails_dir=dirs["emails"],
        receipts_dir=dirs["receipts"],
        processed_receipts_dir=dirs["processed"],
        state_dir=dirs["state"],
    )


def test_llm_connection_test_success() -> None:
    """When the endpoint returns a valid chat completion, _test_llm_connection returns True."""
    cfg = LlmConfig(
        mode=LlmMode.ONLINE,
        endpoint="http://localhost:11434/v1/chat/completions",
        model_name="llama3",
    )
    with patch("gnc_enrich.services.pipeline.LlmClient.chat") as mock_chat:
        mock_chat.return_value = {"choices": [{"message": {"content": "OK"}}]}
        assert _test_llm_connection(cfg) is True


def test_pipeline_warmup_failure_is_non_fatal(tmp_path: Path) -> None:
    """When LLM warmup raises, pipeline continues (non-fatal)."""
    dirs = _setup_pipeline_dirs(tmp_path)
    config = RunConfig(
        gnucash_path=dirs["gnucash"],
        emails_dir=dirs["emails"],
        receipts_dir=dirs["receipts"],
        processed_receipts_dir=dirs["processed"],
        state_dir=dirs["state"],
        llm=LlmConfig(
            mode=LlmMode.ONLINE,
            endpoint="http://localhost:11434",
            model_name="llama3",
            warmup_on_start=True,
        ),
    )
    with patch("gnc_enrich.services.pipeline.LlmClient") as mock_client:
        mock_client.return_value.enabled = True
        mock_client.return_value.chat.return_value = {"choices": [{"message": {"content": "OK"}}]}
        mock_client.return_value.warmup.side_effect = RuntimeError("Warmup failed")
        result = EnrichmentPipeline().run(config)
    assert result.proposal_count == 3


def test_llm_connection_test_missing_endpoint_returns_false() -> None:
    """Missing endpoint or model returns False without making a request."""
    cfg_empty_endpoint = LlmConfig(mode=LlmMode.ONLINE, endpoint="", model_name="llama3")
    cfg_empty_model = LlmConfig(
        mode=LlmMode.ONLINE,
        endpoint="http://localhost:11434",
        model_name="",
    )
    assert _test_llm_connection(cfg_empty_endpoint) is False
    assert _test_llm_connection(cfg_empty_model) is False


def test_pipeline_generates_proposals(tmp_path: Path) -> None:
    dirs = _setup_pipeline_dirs(tmp_path)
    config = _make_config(dirs)

    pipeline = EnrichmentPipeline()
    result = pipeline.run(config)

    assert result.proposal_count == 3
    assert result.skipped_count == 0


def test_proposals_persisted_to_state(tmp_path: Path) -> None:
    dirs = _setup_pipeline_dirs(tmp_path)
    config = _make_config(dirs)

    EnrichmentPipeline().run(config)

    state = StateRepository(dirs["state"])
    proposals = state.load_proposals()
    assert len(proposals) == 3
    assert all(p.suggested_splits for p in proposals)
    assert all(p.rationale for p in proposals)


def test_pipeline_respects_skipped(tmp_path: Path) -> None:
    dirs = _setup_pipeline_dirs(tmp_path)
    config = _make_config(dirs)

    state = StateRepository(dirs["state"])
    from gnc_enrich.domain.models import SkipRecord

    state.save_skip(SkipRecord(tx_id="tx_unspec1", reason="skip test"))

    pipeline = EnrichmentPipeline()
    result = pipeline.run(config)

    assert result.proposal_count == 2
    assert result.skipped_count == 1


def test_pipeline_include_skipped(tmp_path: Path) -> None:
    dirs = _setup_pipeline_dirs(tmp_path)
    config = _make_config(dirs)
    config = RunConfig(
        gnucash_path=config.gnucash_path,
        emails_dir=config.emails_dir,
        receipts_dir=config.receipts_dir,
        processed_receipts_dir=config.processed_receipts_dir,
        state_dir=config.state_dir,
        include_skipped=True,
    )

    state = StateRepository(dirs["state"])
    from gnc_enrich.domain.models import SkipRecord

    state.save_skip(SkipRecord(tx_id="tx_unspec1", reason="skip test"))

    pipeline = EnrichmentPipeline()
    result = pipeline.run(config)

    assert result.proposal_count == 3


def test_run_config_metadata_saved(tmp_path: Path) -> None:
    dirs = _setup_pipeline_dirs(tmp_path)
    config = _make_config(dirs)
    EnrichmentPipeline().run(config)

    state = StateRepository(dirs["state"])
    meta = state.load_metadata("run_config")
    assert meta is not None
    assert "gnucash_path" in meta


def test_run_config_persists_llm_and_extraction_when_set(tmp_path: Path) -> None:
    """Run metadata includes LLM mode/endpoint/model and extraction endpoint/model when configured (spec §5, §13)."""
    from gnc_enrich.config import LlmConfig, LlmMode

    dirs = _setup_pipeline_dirs(tmp_path)
    config = RunConfig(
        gnucash_path=dirs["gnucash"],
        emails_dir=dirs["emails"],
        receipts_dir=dirs["receipts"],
        processed_receipts_dir=dirs["processed"],
        state_dir=dirs["state"],
        llm=LlmConfig(
            mode=LlmMode.ONLINE,
            endpoint="http://llm:8080",
            model_name="test-model",
            use_web=True,
            warmup_on_start=True,
            extraction_endpoint="http://extract:8080",
            extraction_model="extract-model",
        ),
    )
    EnrichmentPipeline().run(config)

    state = StateRepository(dirs["state"])
    meta = state.load_metadata("run_config")
    assert meta is not None
    assert meta.get("llm_mode") == "online"
    assert meta.get("llm_endpoint") == "http://llm:8080"
    assert meta.get("llm_model") == "test-model"
    assert meta.get("llm_use_web") is True
    assert meta.get("llm_extraction_endpoint") == "http://extract:8080"
    assert meta.get("llm_extraction_model") == "extract-model"
    assert meta.get("llm_warmup_on_start") is True


def test_account_paths_saved_after_pipeline_run(tmp_path: Path) -> None:
    """Pipeline persists GnuCash account paths to state for review UI dropdown."""
    dirs = _setup_pipeline_dirs(tmp_path)
    config = _make_config(dirs)
    EnrichmentPipeline().run(config)

    state = StateRepository(dirs["state"])
    meta = state.load_metadata("account_paths")
    assert meta is not None
    paths = meta.get("paths", [])
    assert isinstance(paths, list)
    assert "Expenses:Food" in paths or "Current Account" in paths


def test_proposals_have_evidence(tmp_path: Path) -> None:
    dirs = _setup_pipeline_dirs(tmp_path)
    config = _make_config(dirs)
    pipeline = EnrichmentPipeline()
    proposals = pipeline.build_proposals(config)

    for p in proposals:
        assert p.evidence is not None
        assert p.confidence > 0


def test_build_proposals_raises_when_gnucash_has_no_book(tmp_path: Path) -> None:
    """When GnuCash file has no <gnc:book>, build_proposals raises ValueError (invalid file)."""
    invalid = tmp_path / "invalid.gnucash"
    invalid.write_text('<?xml version="1.0"?><root><other/></root>', encoding="utf-8")
    dirs = _setup_pipeline_dirs(tmp_path)
    config = RunConfig(
        gnucash_path=invalid,
        emails_dir=dirs["emails"],
        receipts_dir=dirs["receipts"],
        processed_receipts_dir=dirs["processed"],
        state_dir=dirs["state"],
    )
    pipeline = EnrichmentPipeline()
    with pytest.raises(ValueError, match="gnc:book|No.*book"):
        pipeline.build_proposals(config)


def test_run_raises_when_gnucash_has_no_book(tmp_path: Path) -> None:
    """When GnuCash file has no <gnc:book>, run() propagates ValueError."""
    invalid = tmp_path / "invalid.gnucash"
    invalid.write_text('<?xml version="1.0"?><root><other/></root>', encoding="utf-8")
    dirs = _setup_pipeline_dirs(tmp_path)
    config = RunConfig(
        gnucash_path=invalid,
        emails_dir=dirs["emails"],
        receipts_dir=dirs["receipts"],
        processed_receipts_dir=dirs["processed"],
        state_dir=dirs["state"],
    )
    with pytest.raises(ValueError, match="gnc:book|No.*book"):
        EnrichmentPipeline().run(config)


def test_email_index_min_date_uses_earliest_expense_candidate(tmp_path: Path) -> None:
    """Pipeline must index emails from (earliest Unspecified/Imbalance-GBP tx date - window), not transfers."""
    dirs = _setup_pipeline_dirs(tmp_path)
    config = _make_config(dirs)
    # Sample book expense candidates: 2025-01-15, 2025-01-20, 2025-02-01 (transfer excluded). Earliest = 2025-01-15.
    # With date_window_days=7, min_email_date should be 2025-01-08.
    expected_min = date(2025, 1, 15) - timedelta(days=7)
    assert expected_min == date(2025, 1, 8)

    with patch("gnc_enrich.services.pipeline.EmailIndexRepository") as mock_repo_class:
        mock_repo = mock_repo_class.return_value
        pipeline = EnrichmentPipeline()
        pipeline.build_proposals(config)

        mock_repo.build_or_load.assert_called_once()
        call_kw = mock_repo.build_or_load.call_args[1]
        assert call_kw["min_date"] == expected_min
