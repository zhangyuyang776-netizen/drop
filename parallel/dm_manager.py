from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from petsc4py import PETSc


@dataclass(slots=True)
class DMManager:
    comm: PETSc.Comm
    dm: PETSc.DM
    dm_liq: PETSc.DM
    dm_gas: PETSc.DM
    dm_if: PETSc.DM
    n_if: int
    dof_liq: int
    dof_gas: int
    Nl: int
    Ng: int


def _layout_has_block(layout, name: str) -> bool:
    try:
        return layout.has_block(name) and layout.block_size(name) > 0
    except Exception:
        return False


def _get_counts(cfg, layout) -> Tuple[int, int, int, int, int]:
    if layout is not None:
        Nl = int(getattr(layout, "Nl", getattr(cfg.geometry, "N_liq")))
        Ng = int(getattr(layout, "Ng", getattr(cfg.geometry, "N_gas")))

        dof_liq = 0
        if _layout_has_block(layout, "Tl"):
            dof_liq += 1
        if _layout_has_block(layout, "Yl"):
            dof_liq += int(getattr(layout, "Ns_l_eff", 0))

        dof_gas = 0
        if _layout_has_block(layout, "Tg"):
            dof_gas += 1
        if _layout_has_block(layout, "Yg"):
            dof_gas += int(getattr(layout, "Ns_g_eff", 0))

        n_if = 0
        for b in ("Ts", "mpp", "Rd"):
            if _layout_has_block(layout, b):
                n_if += 1
        return Nl, Ng, dof_liq, dof_gas, n_if

    Nl = int(cfg.geometry.N_liq)
    Ng = int(cfg.geometry.N_gas)
    dof_liq = int(bool(cfg.physics.solve_Tl))
    dof_gas = int(bool(cfg.physics.solve_Tg))

    if cfg.physics.solve_Yl:
        dof_liq += max(0, len(cfg.species.liq_species) - 1)
    if cfg.physics.solve_Yg:
        dof_gas += max(0, len(cfg.species.gas_species_full) - 1)

    n_if = int(bool(cfg.physics.include_Ts)) + int(bool(cfg.physics.include_mpp)) + int(bool(cfg.physics.include_Rd))
    return Nl, Ng, dof_liq, dof_gas, n_if


def _create_dmda_1d(comm, n: int, dof: int, sw: int = 1) -> PETSc.DM:
    if n <= 0:
        raise ValueError(f"DMDA requires positive grid size, got n={n}")
    if dof <= 0:
        raise ValueError(f"DMDA requires positive dof per cell, got dof={dof}")
    if comm.getSize() > n:
        raise ValueError(
            f"DMDA setup fails: n={n} < nproc={comm.getSize()} (More processes than data points). "
            "Increase Nl/Ng in MPI tests."
        )

    dm = PETSc.DMDA().create(
        sizes=[n],
        dof=dof,
        stencil_width=sw,
        boundary_type=(PETSc.DMDA.BoundaryType.NONE,),
        stencil_type=PETSc.DMDA.StencilType.BOX,
        comm=comm,
    )
    dm.setUp()
    return dm


def _create_interface_shell_dm(comm, n_if: int) -> PETSc.DM:
    if n_if < 0:
        raise ValueError(f"Interface dof must be >= 0, got n_if={n_if}")

    dm_if = PETSc.DMShell().create(comm=comm)

    def _create_global_vec(dm):
        loc = n_if if comm.getRank() == 0 else 0
        v = PETSc.Vec().createMPI(loc, n_if, comm=comm)
        v.setUp()
        return v

    def _create_local_vec(dm):
        v = PETSc.Vec().createSeq(n_if, comm=PETSc.COMM_SELF)
        v.setUp()
        return v

    dm_if.setCreateGlobalVector(_create_global_vec)
    dm_if.setCreateLocalVector(_create_local_vec)

    try:
        mpicomm = comm.tompi4py()
    except Exception:
        mpicomm = None

    def _bcast_root0(arr_root0):
        if mpicomm is not None:
            return mpicomm.bcast(arr_root0, root=0)
        try:
            gathered = comm.allgather(arr_root0)
            for item in gathered:
                if item is not None:
                    return item
            return None
        except Exception:
            return arr_root0

    def _is_add_mode(mode) -> bool:
        try:
            return mode == PETSc.InsertMode.ADD_VALUES or int(mode) == int(PETSc.InsertMode.ADD_VALUES)
        except Exception:
            return False

    def _g2l(dm, *args):
        if n_if == 0:
            return
        if len(args) != 3:
            raise TypeError(f"DMShell globalToLocal expects 3 args, got {len(args)}")
        Xg, a1, a2 = args
        if isinstance(a1, PETSc.Vec):
            Xl, _mode = a1, a2
        else:
            _mode, Xl = a1, a2

        if comm.getRank() == 0:
            try:
                buf = np.asarray(Xg.getArray(readonly=True), dtype=np.float64).copy()
            except TypeError:
                buf = np.asarray(Xg.getArray(), dtype=np.float64).copy()
        else:
            buf = None
        buf = _bcast_root0(buf)
        if buf is None:
            buf = np.zeros(n_if, dtype=np.float64)

        xl = Xl.getArray()
        xl[:] = buf
        Xl.assemble()

    def _l2g(dm, *args):
        if n_if == 0:
            return
        if len(args) != 3:
            raise TypeError(f"DMShell localToGlobal expects 3 args, got {len(args)}")
        Xl, a1, a2 = args
        if isinstance(a1, PETSc.Vec):
            Xg, mode = a1, a2
        else:
            mode, Xg = a1, a2

        try:
            f_loc = np.asarray(Xl.getArray(readonly=True), dtype=np.float64)
        except TypeError:
            f_loc = np.asarray(Xl.getArray(), dtype=np.float64)

        if mpicomm is not None:
            f_sum = mpicomm.allreduce(f_loc)
        else:
            try:
                f_sum = comm.allreduce(f_loc)
            except Exception:
                f_sum = f_loc

        if comm.getRank() == 0:
            xg = Xg.getArray()
            if xg.size != f_sum.size:
                raise ValueError(f"Interface global vec size mismatch: {xg.size} vs {f_sum.size}")
            if _is_add_mode(mode):
                xg[:] += f_sum
            else:
                xg[:] = f_sum
            Xg.assemble()

    dm_if.setGlobalToLocal(_g2l)
    dm_if.setLocalToGlobal(_l2g)

    dm_if.setUp()
    return dm_if


def build_dm(cfg, layout, comm: Optional[PETSc.Comm] = None) -> DMManager:
    comm = PETSc.COMM_WORLD if comm is None else comm
    Nl, Ng, dof_liq, dof_gas, n_if = _get_counts(cfg, layout)

    dm_liq = _create_dmda_1d(comm, Nl, dof_liq, sw=1)
    dm_gas = _create_dmda_1d(comm, Ng, dof_gas, sw=1)
    dm_if = _create_interface_shell_dm(comm, n_if)

    dm = PETSc.DMComposite().create(comm=comm)
    dm.addDM(dm_liq)
    dm.addDM(dm_gas)
    dm.addDM(dm_if)
    dm.setUp()

    return DMManager(
        comm=comm,
        dm=dm,
        dm_liq=dm_liq,
        dm_gas=dm_gas,
        dm_if=dm_if,
        n_if=n_if,
        dof_liq=dof_liq,
        dof_gas=dof_gas,
        Nl=Nl,
        Ng=Ng,
    )


def create_global_vec(mgr: DMManager) -> PETSc.Vec:
    return mgr.dm.createGlobalVec()


def create_local_vecs(mgr: DMManager):
    liq = mgr.dm_liq.createLocalVec()
    gas = mgr.dm_gas.createLocalVec()
    iface = mgr.dm_if.createLocalVec()
    return liq, gas, iface


def global_to_local(mgr: DMManager, Xg: PETSc.Vec):
    X_liq, X_gas, X_if = mgr.dm.getAccess(Xg)
    try:
        Xl_liq = mgr.dm_liq.createLocalVec()
        Xl_gas = mgr.dm_gas.createLocalVec()
        mgr.dm_liq.globalToLocal(X_liq, Xl_liq, addv=PETSc.InsertMode.INSERT_VALUES)
        mgr.dm_gas.globalToLocal(X_gas, Xl_gas, addv=PETSc.InsertMode.INSERT_VALUES)
        Xl_if = mgr.dm_if.createLocalVec()
        mgr.dm_if.globalToLocal(X_if, Xl_if, addv=PETSc.InsertMode.INSERT_VALUES)
    finally:
        mgr.dm.restoreAccess(Xg, (X_liq, X_gas, X_if))
    return Xl_liq, Xl_gas, Xl_if


def local_to_global_add(
    mgr: DMManager,
    Fl_liq: PETSc.Vec,
    Fl_gas: PETSc.Vec,
    F_if: PETSc.Vec,
) -> PETSc.Vec:
    Fg = mgr.dm.createGlobalVec()
    Fg.set(0.0)

    F_liq, F_gas, F_if_g = mgr.dm.getAccess(Fg)
    try:
        mgr.dm_liq.localToGlobal(Fl_liq, F_liq, addv=PETSc.InsertMode.ADD_VALUES)
        mgr.dm_gas.localToGlobal(Fl_gas, F_gas, addv=PETSc.InsertMode.ADD_VALUES)
        mgr.dm_if.localToGlobal(F_if, F_if_g, addv=PETSc.InsertMode.ADD_VALUES)
    finally:
        mgr.dm.restoreAccess(Fg, (F_liq, F_gas, F_if_g))
    return Fg
