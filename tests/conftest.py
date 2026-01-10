from __future__ import annotations

import os
import sys
from pathlib import Path


def _detect_rank() -> int | None:
    for key in (
        "OMPI_COMM_WORLD_RANK",
        "PMI_RANK",
        "SLURM_PROCID",
        "MPI_LOCALRANKID",
        "MV2_COMM_WORLD_RANK",
    ):
        value = os.environ.get(key)
        if value is None:
            continue
        try:
            return int(value)
        except Exception:
            pass
    return None


def pytest_configure(config):
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    config.addinivalue_line("markers", "mpi: tests that require mpiexec to run")
    base_root = os.environ.get("DROPLET_PYTEST_BASETEMP_ROOT")
    if base_root:
        rank = _detect_rank()
        pid = os.getpid()
        tag = f"rank{rank:03d}" if rank is not None else "rankNA"
        config.option.basetemp = str(Path(base_root) / f"{tag}-pid{pid}")
    try:
        from parallel.mpi_bootstrap import bootstrap_mpi

        bootstrap_mpi()
    except Exception:
        pass
