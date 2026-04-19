# Medical CV Hackathon

Deep learning for medical image classification — heart, brain, skin, and other conditions.

**Team:** 5 members
**Duration:** 1 week
**Stack:** Python 3.11 · PyTorch · MONAI · uv

---

## Quick start

### 1. Prerequisites

Install these on your machine (one-time):

- **Python 3.11** — [python.org/downloads](https://www.python.org/downloads/)
- **Git** — [git-scm.com/downloads](https://git-scm.com/downloads)
- **uv** (Python package manager):
  - macOS / Linux: `curl -LsSf https://astral.sh/uv/install.sh | sh`
  - Windows: `powershell -c "irm https://astral.sh/uv/install.ps1 | iex"`
- **VSCode** (recommended) — [code.visualstudio.com](https://code.visualstudio.com/)

### 2. Clone the repo

    git clone https://github.com/sonderbot-ar/medical-cv-hackathon.git
    cd medical-cv-hackathon

### 3. Install dependencies

    uv sync

This reads `uv.lock` and installs exact package versions — everyone gets an identical environment.

### 4. Verify your hardware

    uv run python check_device.py

Expected output:
- **Apple Silicon Mac** → `Selected device: mps`
- **Windows/Linux with NVIDIA GPU** → `Selected device: cuda`
- **CPU-only machine** → `Selected device: cpu` (heavier training should go to Colab)

### 5. Open in VSCode

    code .

---

## Project structure

- `data/` — datasets (gitignored)
- `notebooks/` — Jupyter notebooks for exploration
- `src/data/` — dataset loaders, transforms
- `src/models/` — model architectures
- `src/training/` — training loops, loss functions
- `src/utils/` — helper functions
- `configs/` — experiment configs
- `outputs/` — checkpoints, logs, predictions (gitignored)
- `check_device.py` — hardware verification script

**Important:** data and model checkpoints are NOT stored in git.

---

## Git workflow

Everyone pushes to `master`. Keep it simple:

    git pull                           # 1. Start every session
    # ... do work ...
    git status                         # 2. See what changed
    git add .                          # 3. Stage changes
    git commit -m "description"        # 4. Commit locally
    git push                           # 5. Share with team

**Rules:**
- Pull before you start, push when you finish.
- Commit often with clear messages.
- If `git push` fails with "your branch is behind," run `git pull` first, then push.

---