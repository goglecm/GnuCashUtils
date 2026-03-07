from gnc_enrich.cli import build_parser


def test_parser_exposes_expected_commands() -> None:
    parser = build_parser()
    ns = parser.parse_args(["review", "--state-dir", "/tmp/state"])
    assert ns.command == "review"
    assert ns.host == "127.0.0.1"
    assert ns.port == 7860


def test_run_command_defaults() -> None:
    parser = build_parser()
    ns = parser.parse_args([
        "run",
        "--gnucash-path", "/tmp/book.gnucash",
        "--emails-dir", "/tmp/emails",
        "--receipts-dir", "/tmp/receipts",
        "--processed-receipts-dir", "/tmp/done",
        "--state-dir", "/tmp/state",
    ])
    assert ns.command == "run"
    assert ns.date_window_days == 7
    assert ns.amount_tolerance == 0.50
    assert ns.include_skipped is False
    assert ns.llm_mode == "disabled"
    assert ns.llm_endpoint == ""


def test_apply_command_dry_run() -> None:
    parser = build_parser()
    ns = parser.parse_args([
        "apply",
        "--state-dir", "/tmp/state",
        "--dry-run",
    ])
    assert ns.command == "apply"
    assert ns.dry_run is True
    assert ns.create_backup is False


def test_run_command_with_llm() -> None:
    parser = build_parser()
    ns = parser.parse_args([
        "run",
        "--gnucash-path", "/tmp/book.gnucash",
        "--emails-dir", "/tmp/emails",
        "--receipts-dir", "/tmp/receipts",
        "--processed-receipts-dir", "/tmp/done",
        "--state-dir", "/tmp/state",
        "--llm-mode", "online",
        "--llm-endpoint", "http://localhost:11434",
        "--llm-model", "llama3",
    ])
    assert ns.llm_mode == "online"
    assert ns.llm_endpoint == "http://localhost:11434"
    assert ns.llm_model == "llama3"
