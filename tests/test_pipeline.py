"""Tests for the enrichment pipeline orchestration."""

import gzip
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch

from gnc_enrich.config import RunConfig
from gnc_enrich.services.pipeline import EnrichmentPipeline
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


def test_pipeline_generates_proposals(tmp_path: Path) -> None:
    dirs = _setup_pipeline_dirs(tmp_path)
    config = _make_config(dirs)

    pipeline = EnrichmentPipeline()
    result = pipeline.run(config)

    assert result.proposal_count == 4
    assert result.skipped_count == 0


def test_proposals_persisted_to_state(tmp_path: Path) -> None:
    dirs = _setup_pipeline_dirs(tmp_path)
    config = _make_config(dirs)

    EnrichmentPipeline().run(config)

    state = StateRepository(dirs["state"])
    proposals = state.load_proposals()
    assert len(proposals) == 4
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

    assert result.proposal_count == 3
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

    assert result.proposal_count == 4


def test_run_config_metadata_saved(tmp_path: Path) -> None:
    dirs = _setup_pipeline_dirs(tmp_path)
    config = _make_config(dirs)
    EnrichmentPipeline().run(config)

    state = StateRepository(dirs["state"])
    meta = state.load_metadata("run_config")
    assert meta is not None
    assert "gnucash_path" in meta


def test_proposals_have_evidence(tmp_path: Path) -> None:
    dirs = _setup_pipeline_dirs(tmp_path)
    config = _make_config(dirs)
    pipeline = EnrichmentPipeline()
    proposals = pipeline.build_proposals(config)

    for p in proposals:
        assert p.evidence is not None
        assert p.confidence > 0


def test_email_index_min_date_uses_earliest_candidate(tmp_path: Path) -> None:
    """Pipeline must index emails from (earliest candidate date - window) so old candidates get matches."""
    dirs = _setup_pipeline_dirs(tmp_path)
    config = _make_config(dirs)
    # Sample book candidates: 2025-01-15, 2025-01-20, 2025-02-01, 2025-01-25 (transfer). Earliest = 2025-01-15.
    # With date_window_days=7, min_email_date should be 2025-01-08.
    expected_min = date(2025, 1, 15) - timedelta(days=7)
    assert expected_min == date(2025, 1, 8)

    with patch(
        "gnc_enrich.services.pipeline.EmailIndexRepository"
    ) as mock_repo_class:
        mock_repo = mock_repo_class.return_value
        pipeline = EnrichmentPipeline()
        pipeline.build_proposals(config)

        mock_repo.build_or_load.assert_called_once()
        call_kw = mock_repo.build_or_load.call_args[1]
        assert call_kw["min_date"] == expected_min
