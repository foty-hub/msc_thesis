import os
from pathlib import Path

from dotenv import load_dotenv


def project_root() -> Path:
    """Return this repository's root directory."""
    return Path(__file__).resolve().parents[3]


def load_env() -> None:
    """Load the repository's .env file without replacing process variables."""
    load_dotenv(dotenv_path=project_root() / ".env", override=False)


def get_models_dir() -> Path:
    """Resolve the configured models directory."""
    load_env()
    raw = os.getenv("MODELS_DIR", "models")
    p = Path(raw).expanduser()
    base = p if p.is_absolute() else (project_root() / p)
    return base.resolve()


def models_subdir(*parts: str) -> Path:
    """Convenience helper: join subdirectories under the models base dir."""
    return get_models_dir().joinpath(*parts)
