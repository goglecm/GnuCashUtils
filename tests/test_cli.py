from gnc_enrich.cli import build_parser


def test_parser_exposes_expected_commands() -> None:
    parser = build_parser()
    ns = parser.parse_args(["review", "--state-dir", "/tmp/state"])
    assert ns.command == "review"
    assert ns.host == "127.0.0.1"
    assert ns.port == 7860
