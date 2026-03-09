from pathlib import Path

from gnc_enrich.prompts import get_prompts_dir, load_template, render


def test_get_prompts_dir_uses_user_directory_when_present(tmp_path: Path) -> None:
    """get_prompts_dir returns the user prompts directory when it exists."""
    user_dir = tmp_path / "prompts"
    user_dir.mkdir()
    chosen = get_prompts_dir(user_dir)
    assert chosen == user_dir


def test_get_prompts_dir_falls_back_to_package_when_missing(tmp_path: Path) -> None:
    """get_prompts_dir falls back to the package prompts directory when user dir is missing."""
    # Passing a path that does not exist should fall back to the package directory.
    missing_dir = tmp_path / "does_not_exist"
    chosen = get_prompts_dir(missing_dir)
    # The fallback must not be the missing path.
    assert chosen != missing_dir
    assert chosen.is_dir()


def test_load_template_prefers_user_prompt_over_package(tmp_path: Path) -> None:
    """load_template prefers the user prompts_dir when both contain the same template."""
    user_dir = tmp_path / "prompts"
    user_dir.mkdir()
    user_template = user_dir / "extract_email.txt"
    user_template.write_text("USER TEMPLATE {{email_context}}", encoding="utf-8")

    content = load_template(user_dir, "extract_email")
    assert "USER TEMPLATE" in content
    assert "{{email_context}}" in content


def test_load_template_missing_returns_empty(tmp_path: Path) -> None:
    """load_template returns empty string when no template file exists."""
    user_dir = tmp_path / "prompts"
    user_dir.mkdir()
    content = load_template(user_dir, "nonexistent_template_name")
    assert content == ""


def test_render_replaces_placeholders_and_coerces_values() -> None:
    """render replaces {{key}} with safe stringified values."""
    template = "A={{a}} B={{b}} C={{c}}"
    out = render(template, a="one", b=2, c=None)
    assert out == "A=one B=2 C="
