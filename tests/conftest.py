from __future__ import annotations

import sys
from pathlib import Path


def pytest_configure(config):
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        from parallel.mpi_bootstrap import bootstrap_mpi

        bootstrap_mpi()
    except Exception:
        pass
