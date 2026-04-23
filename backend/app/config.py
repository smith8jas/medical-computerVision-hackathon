from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIR = REPO_ROOT / "backend" / "models"
DEFAULT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
FRONTEND_DIR = REPO_ROOT / "frontend"


def load_local_env() -> None:
    """Load simple KEY=VALUE pairs from a local .env without overriding real env vars."""
    env_path = REPO_ROOT / ".env"
    if not env_path.exists():
        return

    import os

    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


load_local_env()
