"""CLI entrypoints for run/review/apply workflows."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from gnc_enrich.config import ApplyConfig, LlmConfig, LlmMode, ReviewConfig, RunConfig

logger = logging.getLogger(__name__)


def _configure_logging(verbose: bool) -> None:
    """Set up logging with DEBUG level if verbose, else INFO."""
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s %(levelname)-7s [%(name)s] %(message)s"
    logging.basicConfig(level=level, format=fmt)
    if verbose:
        logger.debug("Verbose logging enabled")


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser with run/review/apply subcommands."""
    parser = argparse.ArgumentParser(prog="gnc-enrich")
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable debug-level trace logging",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Build candidate proposals from source data")
    run.add_argument("--gnucash-path", type=Path, required=True)
    run.add_argument("--emails-dir", type=Path, required=True)
    run.add_argument("--receipts-dir", type=Path, required=True)
    run.add_argument("--processed-receipts-dir", type=Path, required=True)
    run.add_argument("--state-dir", type=Path, required=True)
    run.add_argument("--date-window-days", type=int, default=7)
    run.add_argument("--amount-tolerance", type=float, default=0.50)
    run.add_argument("--include-skipped", action="store_true")
    run.add_argument(
        "--llm-mode",
        choices=["disabled", "offline", "online"],
        default="disabled",
    )
    run.add_argument("--llm-endpoint", default="")
    run.add_argument("--llm-model", default="")
    run.add_argument(
        "--llm-extraction-endpoint",
        default="",
        help="Optional separate LLM endpoint for email extraction (when set, extracts seller/items/order IDs from emails before categorisation)",
    )
    run.add_argument("--llm-extraction-model", default="")
    run.add_argument("--llm-extraction-api-key", default="")
    run.add_argument(
        "--llm-use-web",
        action="store_true",
        help="When using a local LLM, run web searches and inject results into the prompt for better category suggestions",
    )
    run.add_argument(
        "--llm-timeout",
        type=int,
        default=180,
        metavar="SECS",
        help="Timeout in seconds for LLM API calls (default 180; local Ollama can be slow)",
    )
    run.add_argument(
        "--llm-warmup-on-start",
        action="store_true",
        help="Send a small warmup request to the LLM at the start of run (may reduce first-call latency).",
    )

    review = sub.add_parser("review", help="Run local web review app")
    review.add_argument("--state-dir", type=Path, required=True)
    review.add_argument("--host", default="127.0.0.1")
    review.add_argument("--port", type=int, default=7860)

    apply_cmd = sub.add_parser("apply", help="Apply approved changes to GnuCash")
    apply_cmd.add_argument("--state-dir", type=Path, required=True)
    apply_cmd.add_argument(
        "--create-backup", action="store_true", default=True,
        help="Create a timestamped backup before writing (default: true)",
    )
    apply_cmd.add_argument(
        "--no-backup", action="store_false", dest="create_backup",
        help="Skip backup creation",
    )
    apply_cmd.add_argument("--backup-dir", type=Path)
    apply_cmd.add_argument("--in-place", action="store_true", default=True)
    apply_cmd.add_argument(
        "--no-in-place", action="store_false", dest="in_place",
        help="Write to a new file instead of modifying the original",
    )
    apply_cmd.add_argument("--dry-run", action="store_true")
    apply_cmd.add_argument(
        "--backup-retention",
        type=int,
        default=None,
        metavar="N",
        help="Optional maximum number of backups to retain per GnuCash file (default: unlimited).",
    )

    rollback = sub.add_parser("rollback", help="Rollback GnuCash file from backup")
    rollback.add_argument("--state-dir", type=Path, required=True)
    rollback.add_argument(
        "--backup",
        default="",
        help="Optional backup filename (in the state's backups directory) to restore. "
             "If omitted, the most recent backup is used.",
    )
    rollback.add_argument(
        "--list-backups",
        action="store_true",
        help="List available backups for the current state-dir and exit.",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse CLI arguments and dispatch to the appropriate workflow."""
    parser = build_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.verbose)

    if args.command == "run":
        llm = LlmConfig(
            mode=LlmMode(args.llm_mode),
            endpoint=args.llm_endpoint,
            model_name=args.llm_model,
            use_web=getattr(args, "llm_use_web", False),
            timeout_seconds=getattr(args, "llm_timeout", 180),
            extraction_endpoint=getattr(args, "llm_extraction_endpoint", ""),
            extraction_model=getattr(args, "llm_extraction_model", ""),
            extraction_api_key=getattr(args, "llm_extraction_api_key", ""),
            warmup_on_start=getattr(args, "llm_warmup_on_start", False),
        )
        config = RunConfig(
            gnucash_path=args.gnucash_path,
            emails_dir=args.emails_dir,
            receipts_dir=args.receipts_dir,
            processed_receipts_dir=args.processed_receipts_dir,
            state_dir=args.state_dir,
            date_window_days=args.date_window_days,
            amount_tolerance=args.amount_tolerance,
            include_skipped=args.include_skipped,
            llm=llm,
        )
        from gnc_enrich.services.pipeline import EnrichmentPipeline

        result = EnrichmentPipeline().run(config)
        print(
            f"Pipeline complete: {result.proposal_count} proposals, "
            f"{result.skipped_count} skipped"
        )
        return 0

    if args.command == "review":
        config = ReviewConfig(state_dir=args.state_dir, host=args.host, port=args.port)
        from gnc_enrich.review.webapp import ReviewWebApp
        from gnc_enrich.review.service import ReviewQueueService
        from gnc_enrich.state.repository import StateRepository

        state_repo = StateRepository(config.state_dir)
        service = ReviewQueueService(state_repo)
        ReviewWebApp(service, config).run(config.host, config.port)
        return 0

    if args.command == "apply":
        config = ApplyConfig(
            state_dir=args.state_dir,
            create_backup=args.create_backup,
            backup_dir=args.backup_dir,
            in_place=args.in_place,
            dry_run=args.dry_run,
            backup_retention=args.backup_retention,
        )
        from gnc_enrich.apply.engine import ApplyEngine

        engine = ApplyEngine()
        if config.dry_run:
            report = engine.generate_dry_run_report(config.state_dir)
            print(f"Dry-run report written to {report}")
        else:
            engine.apply(
                config.state_dir,
                create_backup=config.create_backup,
                backup_dir=config.backup_dir,
                in_place=config.in_place,
                backup_retention=config.backup_retention,
            )
            print("Changes applied successfully.")
        return 0

    if args.command == "rollback":
        from gnc_enrich.apply.engine import ApplyEngine
        from gnc_enrich.state.repository import StateRepository

        state_dir = args.state_dir
        state = StateRepository(state_dir)
        meta = state.load_metadata("run_config")
        if not meta or "gnucash_path" not in meta:
            print(
                "Error: no run_config metadata found in the state directory; "
                "run 'gnc-enrich run' first so backups are registered.",
                file=sys.stderr,
            )
            return 1
        gnucash_path = Path(meta["gnucash_path"])
        backup_dir = state_dir / "backups"
        if args.list_backups:
            if not backup_dir.exists():
                print("No backup directory found.")
                return 0
            backups = sorted(backup_dir.glob(f"{gnucash_path.stem}.*{gnucash_path.suffix}"))
            if not backups:
                print("No backup files found.")
                return 0
            for b in backups:
                print(b.name)
            return 0
        engine = ApplyEngine()
        backup_name = args.backup or None
        engine.rollback(state_dir, backup_name=backup_name)
        print("Rollback completed.")
        return 0

    parser.error(f"Unknown command: {args.command}")
