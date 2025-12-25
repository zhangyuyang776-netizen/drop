from __future__ import annotations

import os
import shutil
import subprocess
import sys

import pytest


def test_mpi_import_smoke():
    mpiexec = shutil.which("mpiexec") or shutil.which("mpirun")
    if mpiexec is None:
        pytest.skip("mpiexec/mpirun not found on PATH")

    code = (
        "from petsc4py import PETSc; "
        "import sys; "
        "sys.stdout.write(str(PETSc.COMM_WORLD.getSize()))"
    )

    cmd = [mpiexec, "-n", "2", sys.executable, "-c", code]
    env = os.environ.copy()

    res = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        timeout=60,
    )

    assert res.returncode == 0, (
        "MPI import smoke failed.\n"
        f"cmd: {' '.join(cmd)}\n"
        f"stdout:\n{res.stdout}\n"
        f"stderr:\n{res.stderr}\n"
    )

    out = (res.stdout or "").strip()
    assert out.endswith("2"), f"Expected COMM_WORLD size=2, got stdout='{out}', stderr='{res.stderr}'"
