"""Tests for the enrichment pipeline orchestration."""

import gzip
from pathlib import Path

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


def test_proposals_have_evidence(tmp_path: Path) -> None:
    dirs = _setup_pipeline_dirs(tmp_path)
    config = _make_config(dirs)
    pipeline = EnrichmentPipeline()
    proposals = pipeline.build_proposals(config)

    for p in proposals:
        assert p.evidence is not None
        assert p.confidence > 0
