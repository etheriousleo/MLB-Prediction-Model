# Puts the study folder on sys.path so `python -m pytest` works from the
# study folder or the repo root, and the tests import config/pull directly.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def pytest_configure(config):
    # Short tracebacks by default. pytest's long style prints every frame's
    # arguments, which here are the raw game logs, and nothing from the
    # confirm season may be shown before the lock (spec §4). An explicit
    # --tb on the command line still wins.
    if config.option.tbstyle == "auto":
        config.option.tbstyle = "short"
