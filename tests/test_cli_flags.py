"""Tests for every CLI flag — parser-level and dispatch-level."""

from __future__ import annotations

import gzip
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gnc_enrich.cli import build_parser, main


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MINIMAL_GNUCASH_XML = """\
<?xml version="1.0" encoding="utf-8" ?>
<gnc-v2
     xmlns:gnc="http://www.gnucash.org/XML/gnc"
     xmlns:act="http://www.gnucash.org/XML/act"
     xmlns:book="http://www.gnucash.org/XML/book"
     xmlns:cd="http://www.gnucash.org/XML/cd"
     xmlns:cmdty="http://www.gnucash.org/XML/cmdty"
     xmlns:slot="http://www.gnucash.org/XML/slot"
     xmlns:split="http://www.gnucash.org/XML/split"
     xmlns:trn="http://www.gnucash.org/XML/trn"
     xmlns:ts="http://www.gnucash.org/XML/ts">
<gnc:count-data cd:type="book">1</gnc:count-data>
<gnc:book version="2.0.0">
<book:id type="guid">flagbook01</book:id>
<gnc:commodity version="2.0.0">
  <cmdty:space>CURRENCY</cmdty:space><cmdty:id>GBP</cmdty:id>
</gnc:commodity>
<gnc:account version="2.0.0">
  <act:name>Root Account</act:name>
  <act:id type="guid">root01</act:id>
  <act:type>ROOT</act:type>
</gnc:account>
<gnc:account version="2.0.0">
  <act:name>Current</act:name>
  <act:id type="guid">acct_cur</act:id>
  <act:type>BANK</act:type>
  <act:commodity><cmdty:space>CURRENCY</cmdty:space><cmdty:id>GBP</cmdty:id></act:commodity>
  <act:parent type="guid">root01</act:parent>
</gnc:account>
<gnc:account version="2.0.0">
  <act:name>Unspecified</act:name>
  <act:id type="guid">acct_unspec</act:id>
  <act:type>EXPENSE</act:type>
  <act:commodity><cmdty:space>CURRENCY</cmdty:space><cmdty:id>GBP</cmdty:id></act:commodity>
  <act:parent type="guid">root01</act:parent>
</gnc:account>
<gnc:transaction version="2.0.0">
  <trn:id type="guid">tx_flag1</trn:id>
  <trn:currency><cmdty:space>CURRENCY</cmdty:space><cmdty:id>GBP</cmdty:id></trn:currency>
  <trn:date-posted><ts:date>2023-06-15 00:00:00 +0000</ts:date></trn:date-posted>
  <trn:description>Flag Test Payment</trn:description>
  <trn:splits>
    <trn:split><split:id type="guid">sp_f1a</split:id><split:value>-2000/100</split:value><split:account type="guid">acct_cur</split:account></trn:split>
    <trn:split><split:id type="guid">sp_f1b</split:id><split:value>2000/100</split:value><split:account type="guid">acct_unspec</split:account></trn:split>
  </trn:splits>
</gnc:transaction>
</gnc:book>
</gnc-v2>
"""


@pytest.fixture()
def cli_env(tmp_path: Path) -> dict[str, Path]:
    gnc = tmp_path / "book.gnucash"
    with gzip.open(gnc, "wb") as f:
        f.write(MINIMAL_GNUCASH_XML.encode())
    (tmp_path / "emails").mkdir()
    (tmp_path / "receipts").mkdir()
    (tmp_path / "processed").mkdir()
    (tmp_path / "state").mkdir()
    return {
        "gnucash": gnc,
        "emails": tmp_path / "emails",
        "receipts": tmp_path / "receipts",
        "processed": tmp_path / "processed",
        "state": tmp_path / "state",
    }


def _run_args(env: dict[str, Path], extra: list[str] | None = None) -> list[str]:
    base = [
        "run",
        "--gnucash-path", str(env["gnucash"]),
        "--emails-dir", str(env["emails"]),
        "--receipts-dir", str(env["receipts"]),
        "--processed-receipts-dir", str(env["processed"]),
        "--state-dir", str(env["state"]),
    ]
    return base + (extra or [])


# ---------------------------------------------------------------------------
# Global flags
# ---------------------------------------------------------------------------


class TestVerboseFlag:
    def test_verbose_flag_parsed(self, cli_env: dict[str, Path]) -> None:
        ns = build_parser().parse_args(["-v"] + _run_args(cli_env))
        assert ns.verbose is True

    def test_no_verbose_flag_parsed(self, cli_env: dict[str, Path]) -> None:
        ns = build_parser().parse_args(_run_args(cli_env))
        assert ns.verbose is False

    def test_verbose_calls_configure_logging(self, cli_env: dict[str, Path]) -> None:
        with patch("gnc_enrich.cli._configure_logging") as mock_cfg:
            main(["-v"] + _run_args(cli_env))
            mock_cfg.assert_called_once_with(True)

    def test_no_verbose_calls_configure_logging(self, cli_env: dict[str, Path]) -> None:
        with patch("gnc_enrich.cli._configure_logging") as mock_cfg:
            main(_run_args(cli_env))
            mock_cfg.assert_called_once_with(False)


# ---------------------------------------------------------------------------
# `run` subcommand flags
# ---------------------------------------------------------------------------


class TestRunFlags:
    def test_date_window_days_non_default(self, cli_env: dict[str, Path]) -> None:
        ns = build_parser().parse_args(_run_args(cli_env, ["--date-window-days", "14"]))
        assert ns.date_window_days == 14

    def test_date_window_days_default(self, cli_env: dict[str, Path]) -> None:
        ns = build_parser().parse_args(_run_args(cli_env))
        assert ns.date_window_days == 7

    def test_amount_tolerance_non_default(self, cli_env: dict[str, Path]) -> None:
        ns = build_parser().parse_args(_run_args(cli_env, ["--amount-tolerance", "1.25"]))
        assert ns.amount_tolerance == 1.25

    def test_amount_tolerance_default(self, cli_env: dict[str, Path]) -> None:
        ns = build_parser().parse_args(_run_args(cli_env))
        assert ns.amount_tolerance == 0.50

    def test_include_skipped_flag(self, cli_env: dict[str, Path]) -> None:
        ns = build_parser().parse_args(_run_args(cli_env, ["--include-skipped"]))
        assert ns.include_skipped is True

    def test_include_skipped_default_false(self, cli_env: dict[str, Path]) -> None:
        ns = build_parser().parse_args(_run_args(cli_env))
        assert ns.include_skipped is False

    def test_include_skipped_changes_pipeline_behavior(self, cli_env: dict[str, Path]) -> None:
        """Skipped transactions should reappear when --include-skipped is set."""
        from gnc_enrich.services.pipeline import EnrichmentPipeline
        from gnc_enrich.state.repository import StateRepository
        from gnc_enrich.review.service import ReviewQueueService
        from gnc_enrich.domain.models import ReviewDecision
        from gnc_enrich.config import RunConfig

        config = RunConfig(
            gnucash_path=cli_env["gnucash"],
            emails_dir=cli_env["emails"],
            receipts_dir=cli_env["receipts"],
            processed_receipts_dir=cli_env["processed"],
            state_dir=cli_env["state"],
        )
        EnrichmentPipeline().run(config)
        state = StateRepository(cli_env["state"])
        svc = ReviewQueueService(state)
        p = svc.all_proposals()[0]
        svc.submit_decision(ReviewDecision(
            tx_id=p.tx_id, action="skip", final_description="", final_splits=[],
        ))

        config_no_skip = RunConfig(
            gnucash_path=cli_env["gnucash"],
            emails_dir=cli_env["emails"],
            receipts_dir=cli_env["receipts"],
            processed_receipts_dir=cli_env["processed"],
            state_dir=cli_env["state"],
            include_skipped=False,
        )
        r1 = EnrichmentPipeline().run(config_no_skip)
        assert r1.proposal_count == 0

        config_with_skip = RunConfig(
            gnucash_path=cli_env["gnucash"],
            emails_dir=cli_env["emails"],
            receipts_dir=cli_env["receipts"],
            processed_receipts_dir=cli_env["processed"],
            state_dir=cli_env["state"],
            include_skipped=True,
        )
        r2 = EnrichmentPipeline().run(config_with_skip)
        assert r2.proposal_count == 1

    def test_llm_mode_disabled(self, cli_env: dict[str, Path]) -> None:
        ns = build_parser().parse_args(_run_args(cli_env))
        assert ns.llm_mode == "disabled"

    def test_llm_mode_offline(self, cli_env: dict[str, Path]) -> None:
        ns = build_parser().parse_args(_run_args(cli_env, ["--llm-mode", "offline"]))
        assert ns.llm_mode == "offline"

    def test_llm_mode_online(self, cli_env: dict[str, Path]) -> None:
        ns = build_parser().parse_args(_run_args(cli_env, ["--llm-mode", "online"]))
        assert ns.llm_mode == "online"

    def test_llm_mode_invalid_rejected(self, cli_env: dict[str, Path]) -> None:
        with pytest.raises(SystemExit):
            build_parser().parse_args(_run_args(cli_env, ["--llm-mode", "invalid"]))

    def test_llm_endpoint_and_model(self, cli_env: dict[str, Path]) -> None:
        ns = build_parser().parse_args(_run_args(cli_env, [
            "--llm-mode", "online",
            "--llm-endpoint", "http://localhost:11434/v1/chat/completions",
            "--llm-model", "mistral",
        ]))
        assert ns.llm_endpoint == "http://localhost:11434/v1/chat/completions"
        assert ns.llm_model == "mistral"

    def test_llm_use_web_flag(self, cli_env: dict[str, Path]) -> None:
        ns = build_parser().parse_args(_run_args(cli_env, [
            "--llm-use-web",
        ]))
        assert getattr(ns, "llm_use_web", False) is True

    def test_run_missing_required_arg_fails(self) -> None:
        with pytest.raises(SystemExit):
            build_parser().parse_args(["run", "--gnucash-path", "/tmp/x"])

    def test_run_dispatch_via_main(self, cli_env: dict[str, Path]) -> None:
        rc = main(_run_args(cli_env))
        assert rc == 0


# ---------------------------------------------------------------------------
# `review` subcommand flags
# ---------------------------------------------------------------------------


class TestReviewFlags:
    def test_host_default(self) -> None:
        ns = build_parser().parse_args(["review", "--state-dir", "/tmp/s"])
        assert ns.host == "127.0.0.1"

    def test_host_custom(self) -> None:
        ns = build_parser().parse_args(["review", "--state-dir", "/tmp/s", "--host", "0.0.0.0"])
        assert ns.host == "0.0.0.0"

    def test_port_default(self) -> None:
        ns = build_parser().parse_args(["review", "--state-dir", "/tmp/s"])
        assert ns.port == 7860

    def test_port_custom(self) -> None:
        ns = build_parser().parse_args(["review", "--state-dir", "/tmp/s", "--port", "9090"])
        assert ns.port == 9090

    def test_review_dispatch_starts_webapp(self, cli_env: dict[str, Path]) -> None:
        """main() with 'review' should construct and run ReviewWebApp."""
        from gnc_enrich.config import RunConfig
        from gnc_enrich.services.pipeline import EnrichmentPipeline
        RunConfig(
            gnucash_path=cli_env["gnucash"],
            emails_dir=cli_env["emails"],
            receipts_dir=cli_env["receipts"],
            processed_receipts_dir=cli_env["processed"],
            state_dir=cli_env["state"],
        )
        EnrichmentPipeline().run(RunConfig(
            gnucash_path=cli_env["gnucash"],
            emails_dir=cli_env["emails"],
            receipts_dir=cli_env["receipts"],
            processed_receipts_dir=cli_env["processed"],
            state_dir=cli_env["state"],
        ))

        with patch("gnc_enrich.review.webapp.ReviewWebApp") as MockWebApp:
            mock_instance = MagicMock()
            MockWebApp.return_value = mock_instance
            mock_instance.run.return_value = None

            rc = main(["review", "--state-dir", str(cli_env["state"]),
                        "--host", "0.0.0.0", "--port", "5000"])
            assert rc == 0
            mock_instance.run.assert_called_once_with("0.0.0.0", 5000)


# ---------------------------------------------------------------------------
# `apply` subcommand flags
# ---------------------------------------------------------------------------


class TestApplyFlags:
    def test_dry_run_flag(self) -> None:
        ns = build_parser().parse_args(["apply", "--state-dir", "/tmp/s", "--dry-run"])
        assert ns.dry_run is True

    def test_dry_run_default_false(self) -> None:
        ns = build_parser().parse_args(["apply", "--state-dir", "/tmp/s"])
        assert ns.dry_run is False

    def test_create_backup_flag(self) -> None:
        ns = build_parser().parse_args(["apply", "--state-dir", "/tmp/s", "--create-backup"])
        assert ns.create_backup is True

    def test_create_backup_default_true(self) -> None:
        ns = build_parser().parse_args(["apply", "--state-dir", "/tmp/s"])
        assert ns.create_backup is True

    def test_no_backup_flag(self) -> None:
        ns = build_parser().parse_args(["apply", "--state-dir", "/tmp/s", "--no-backup"])
        assert ns.create_backup is False

    def test_backup_retention_flag(self) -> None:
        ns = build_parser().parse_args(["apply", "--state-dir", "/tmp/s", "--backup-retention", "5"])
        assert ns.backup_retention == 5

    def test_backup_dir_flag(self, tmp_path: Path) -> None:
        ns = build_parser().parse_args([
            "apply", "--state-dir", "/tmp/s",
            "--backup-dir", str(tmp_path / "backups"),
        ])
        assert ns.backup_dir == tmp_path / "backups"

    def test_backup_dir_default_none(self) -> None:
        ns = build_parser().parse_args(["apply", "--state-dir", "/tmp/s"])
        assert ns.backup_dir is None

    def test_in_place_flag(self) -> None:
        ns = build_parser().parse_args(["apply", "--state-dir", "/tmp/s", "--in-place"])
        assert ns.in_place is True

    def test_in_place_default_true(self) -> None:
        ns = build_parser().parse_args(["apply", "--state-dir", "/tmp/s"])
        assert ns.in_place is True

    def test_no_in_place_flag(self) -> None:
        ns = build_parser().parse_args(["apply", "--state-dir", "/tmp/s", "--no-in-place"])
        assert ns.in_place is False

    def test_apply_dry_run_dispatch(self, cli_env: dict[str, Path]) -> None:
        """main() with 'apply --dry-run' generates a report."""
        from gnc_enrich.config import RunConfig
        from gnc_enrich.services.pipeline import EnrichmentPipeline
        from gnc_enrich.state.repository import StateRepository
        from gnc_enrich.review.service import ReviewQueueService
        from gnc_enrich.domain.models import ReviewDecision

        EnrichmentPipeline().run(RunConfig(
            gnucash_path=cli_env["gnucash"],
            emails_dir=cli_env["emails"],
            receipts_dir=cli_env["receipts"],
            processed_receipts_dir=cli_env["processed"],
            state_dir=cli_env["state"],
        ))
        state = StateRepository(cli_env["state"])
        svc = ReviewQueueService(state)
        for p in svc.all_proposals():
            svc.submit_decision(ReviewDecision(
                tx_id=p.tx_id, action="approve",
                final_description=p.suggested_description,
                final_splits=p.suggested_splits,
            ))

        rc = main(["apply", "--state-dir", str(cli_env["state"]), "--dry-run"])
        assert rc == 0

    def test_apply_in_place_dispatch(self, cli_env: dict[str, Path]) -> None:
        """main() with 'apply --in-place' applies changes."""
        from gnc_enrich.config import RunConfig
        from gnc_enrich.services.pipeline import EnrichmentPipeline
        from gnc_enrich.state.repository import StateRepository
        from gnc_enrich.review.service import ReviewQueueService
        from gnc_enrich.domain.models import ReviewDecision

        EnrichmentPipeline().run(RunConfig(
            gnucash_path=cli_env["gnucash"],
            emails_dir=cli_env["emails"],
            receipts_dir=cli_env["receipts"],
            processed_receipts_dir=cli_env["processed"],
            state_dir=cli_env["state"],
        ))
        state = StateRepository(cli_env["state"])
        svc = ReviewQueueService(state)
        for p in svc.all_proposals():
            svc.submit_decision(ReviewDecision(
                tx_id=p.tx_id, action="approve",
                final_description=p.suggested_description,
                final_splits=p.suggested_splits,
            ))

        rc = main(["apply", "--state-dir", str(cli_env["state"]), "--in-place"])
        assert rc == 0

    def test_all_apply_flags_combined(self, tmp_path: Path) -> None:
        ns = build_parser().parse_args([
            "apply", "--state-dir", str(tmp_path),
            "--create-backup",
            "--backup-dir", str(tmp_path / "bak"),
            "--in-place",
        ])
        assert ns.create_backup is True
        assert ns.backup_dir == tmp_path / "bak"
        assert ns.in_place is True
        assert ns.dry_run is False


class TestRollbackFlags:
    def test_rollback_defaults(self) -> None:
        ns = build_parser().parse_args(["rollback", "--state-dir", "/tmp/s"])
        assert ns.command == "rollback"
        assert ns.backup == ""
        assert ns.list_backups is False

    def test_rollback_list_backups(self) -> None:
        ns = build_parser().parse_args(["rollback", "--state-dir", "/tmp/s", "--list-backups"])
        assert ns.list_backups is True

    def test_rollback_missing_run_config_returns_error_code(self, tmp_path: Path) -> None:
        """When run_config metadata is missing, rollback prints an error and returns non-zero."""
        state_dir = tmp_path / "state"
        state_dir.mkdir()
        rc = main(["rollback", "--state-dir", str(state_dir)])
        assert rc == 1
