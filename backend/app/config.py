from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIR = REPO_ROOT / "backend" / "models"
DEFAULT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
FRONTEND_DIR = REPO_ROOT / "frontend"
