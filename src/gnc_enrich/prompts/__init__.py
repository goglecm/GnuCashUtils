"""LLM prompt templates loaded from text files with {{placeholders}}."""

from pathlib import Path

_PACKAGE_PROMPTS = Path(__file__).parent


def get_prompts_dir(user_prompts_dir: Path | None) -> Path:
    """Return the directory to load prompts from (user override or package defaults)."""
    return (
        user_prompts_dir
        if user_prompts_dir is not None and user_prompts_dir.is_dir()
        else _PACKAGE_PROMPTS
    )


def load_template(prompts_dir: Path, name: str) -> str:
    """Load a prompt template by name (e.g. 'extract_email'). Tries user dir then package."""
    basename = name if name.endswith(".txt") else f"{name}.txt"
    for directory in (prompts_dir, _PACKAGE_PROMPTS):
        path = directory / basename
        if path.is_file():
            return path.read_text(encoding="utf-8").strip()
    return ""


def render(template: str, **kwargs: str) -> str:
    """Replace {{key}} placeholders with kwargs. Keys are case-sensitive."""
    result = template
    for key, value in kwargs.items():
        if value is None:
            safe_val = ""
        elif isinstance(value, str):
            safe_val = value
        else:
            safe_val = str(value)
        result = result.replace("{{" + key + "}}", safe_val)
    return result
