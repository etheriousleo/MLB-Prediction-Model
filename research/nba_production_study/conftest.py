# Puts the study folder on sys.path so `python -m pytest` works from the
# study folder or the repo root, and the tests import config/pull directly.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
