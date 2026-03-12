"""Pytest configuration for finetuning tests."""

import sys
from pathlib import Path

# Add repo root to path so finetuning package is importable
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
