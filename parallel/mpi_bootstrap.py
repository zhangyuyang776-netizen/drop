from __future__ import annotations

_BOOTSTRAPPED = False


def bootstrap_mpi() -> None:
    """
    Ensure mpi4py initializes before petsc4py.
    """
    global _BOOTSTRAPPED
    if _BOOTSTRAPPED:
        return
    _BOOTSTRAPPED = True

    try:
        from mpi4py import MPI  # noqa: F401
    except Exception:
        return


def bootstrap_mpi_before_petsc() -> None:
    bootstrap_mpi()
