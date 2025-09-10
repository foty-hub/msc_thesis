import os
from functools import lru_cache
from pathlib import Path

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional at import time
    load_dotenv = None  # type: ignore


@lru_cache(maxsize=1)
def project_root() -> Path:
    """Return the repository root.

    Strategy (in order):
    - PROJECT_ROOT env var if set.
    - Walk upwards from this file looking for a directory containing
      'pyproject.toml' or '.git'.
    - Fallback to current working directory.
    """
    # 1) Explicit override
    env_override = os.getenv("PROJECT_ROOT")
    if env_override:
        p = Path(env_override).expanduser()
        return p.resolve()

    # 2) Walk up from package location
    here = Path(__file__).resolve()
    for ancestor in [here, *here.parents]:
        # Stop at the first directory that contains pyproject or .git
        if (ancestor / "pyproject.toml").exists() or (ancestor / ".git").exists():
            # If the match was inside src/crl/utils, jump to the project root
            # which should be the directory containing pyproject/.git
            return ancestor

    # 3) Fallback
    return Path.cwd().resolve()


@lru_cache(maxsize=1)
def load_env() -> None:
    """Load .env from project root, if python-dotenv is available."""
    if load_dotenv is None:
        return None
    dotenv_path = project_root() / ".env"
    # Do not override already-set env vars
    load_dotenv(dotenv_path=dotenv_path, override=False)
    return None


@lru_cache(maxsize=1)
def get_models_dir() -> Path:
    """Resolve the base models directory.

    - Reads MODELS_DIR from .env (loaded from project root) or env.
    - If relative, resolve relative to project_root().
    - If missing, default to 'models' at the project root.
    """
    load_env()
    raw = os.getenv("MODELS_DIR", "models")
    p = Path(raw).expanduser()
    base = p if p.is_absolute() else (project_root() / p)
    return base.resolve()


def models_subdir(*parts: str) -> Path:
    """Convenience helper: join subdirectories under the models base dir."""
    return get_models_dir().joinpath(*parts)
