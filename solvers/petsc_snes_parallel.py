# -*- coding: utf-8 -*-
"""
PETSc SNES nonlinear solver wrapper (parallel-only).

Scope:
- Always uses PETSc.COMM_WORLD as the communicator.
- Auto-builds DM + dm_manager + layout_dist if missing from ctx.meta.
- Matrix-free Jacobian (SNES MFFD); supports mf and mfpc_sparse_fd modes.
- Simple global PC (ASM) only; no fieldsplit variants.
- This module does NOT provide serial fallback; use solvers.petsc_snes for serial or hybrid modes.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List

import numpy as np

from assembly.residual_global import residual_only, residual_petsc
from solvers.linear_types import JacobianMode
from solvers.nonlinear_context import NonlinearContext
from solvers.nonlinear_types import NonlinearDiagnostics, NonlinearSolveResult
from solvers.petsc_linear import apply_structured_pc
from solvers.petsc_snes import _enable_snes_matrix_free, _finalize_ksp_config, _get_petsc

logger = logging.getLogger(__name__)


def _create_identity_precond_mat(comm, dm):
    """
    Stage P1.5: DM-driven MPIAIJ identity preconditioner matrix.
    Ownership ranges follow dm.createGlobalVec().
    """
    PETSc = _get_petsc()
    x_template = dm.createGlobalVec()
    nloc = int(x_template.getLocalSize())
    n_glob = int(x_template.getSize())
    r0, r1 = x_template.getOwnershipRange()

    P = PETSc.Mat().create(comm=comm)
    P.setSizes(((nloc, n_glob), (nloc, n_glob)))
    try:
        P.setType(PETSc.Mat.Type.AIJ)
    except Exception:
        P.setType("aij")
    try:
        P.setPreallocationNNZ(1)
    except Exception:
        pass
    try:
        P.setUp()
    except Exception:
        pass

    for row in range(int(r0), int(r1)):
        P.setValue(row, row, 1.0)

    P.assemblyBegin()
    P.assemblyEnd()
    return P


def solve_nonlinear_petsc_parallel(
    ctx: NonlinearContext,
    u0: np.ndarray,
) -> NonlinearSolveResult:
    """
    Solve F(u) = 0 using PETSc SNES (parallel-only, DM + MFFD).
    """
    PETSc = _get_petsc()
    comm = PETSc.COMM_WORLD
    world_rank = int(comm.getRank())
    world_size = int(comm.getSize())

    cfg = ctx.cfg
    nl_cfg = getattr(cfg, "nonlinear", None)
    if nl_cfg is None or not getattr(nl_cfg, "enabled", False):
        raise ValueError("Nonlinear solver requested but cfg.nonlinear.enabled is False or missing.")

    if world_rank == 0:
        logger.info(
            "PETSc parallel SNES (COMM_WORLD) starting: size=%d (no serial fallback in this module).",
            world_size,
        )

    petsc_cfg = getattr(cfg, "petsc", None)

    meta = getattr(ctx, "meta", None)
    if meta is None:
        meta = {}
        ctx.meta = meta
    elif not isinstance(meta, dict):
        meta = dict(meta)
        ctx.meta = meta

    dm_mgr = meta.get("dm_manager") or meta.get("dm_mgr") or getattr(ctx, "dm_manager", None)
    dm = meta.get("dm") or getattr(ctx, "dm", None)
    ld = meta.get("layout_dist") or meta.get("ld") or getattr(ctx, "layout_dist", None)

    if dm_mgr is None:
        from parallel.dm_manager import build_dm
        from core.layout_dist import LayoutDistributed

        if world_rank == 0:
            logger.info("Parallel SNES auto-building DM/layout_dist for COMM_WORLD.")

        dm_mgr = build_dm(cfg, ctx.layout, comm=comm)
        dm = dm_mgr.dm
        ld = LayoutDistributed.build(comm, dm_mgr, ctx.layout)

        meta["dm_manager"] = dm_mgr
        meta["dm_mgr"] = dm_mgr
        meta["dm"] = dm
        meta["layout_dist"] = ld
        meta["ld"] = ld
        try:
            ctx.dm_manager = dm_mgr
        except Exception:
            pass
        try:
            ctx.dm = dm
        except Exception:
            pass
        try:
            ctx.layout_dist = ld
        except Exception:
            pass
    else:
        if dm is None:
            dm = getattr(dm_mgr, "dm", None)
        if ld is None:
            from core.layout_dist import LayoutDistributed

            ld = LayoutDistributed.build(comm, dm_mgr, ctx.layout)
            meta["layout_dist"] = ld
            meta["ld"] = ld
            try:
                ctx.layout_dist = ld
            except Exception:
                pass
        if dm is not None:
            meta.setdefault("dm", dm)
        meta.setdefault("dm_manager", dm_mgr)
        meta.setdefault("dm_mgr", dm_mgr)

    if dm is None or ld is None:
        raise RuntimeError("Parallel SNES failed to initialize DM/layout_dist; check configuration.")

    max_outer_iter = int(getattr(nl_cfg, "max_outer_iter", 20))
    # Fixed tolerances for the dedicated MPI backend.
    f_rtol = 1.0e-6
    f_atol = 1.0e-8
    verbose = bool(getattr(nl_cfg, "verbose", False))
    log_every = int(getattr(nl_cfg, "log_every", 5))
    if log_every < 1:
        log_every = 1

    jacobian_mode_raw = "mf"
    if petsc_cfg is not None:
        jacobian_mode_raw = getattr(petsc_cfg, "jacobian_mode", "mf")
    jacobian_mode = JacobianMode.normalize(jacobian_mode_raw)
    if jacobian_mode not in (JacobianMode.MF, JacobianMode.MFPC_SPARSE_FD):
        raise ValueError(
            "Parallel SNES backend supports jacobian_mode='mf' or 'mfpc_sparse_fd' only. "
            f"Got: {jacobian_mode!r}."
        )
    use_mfpc_sparse_fd = (jacobian_mode == JacobianMode.MFPC_SPARSE_FD)
    fd_eps = float(getattr(petsc_cfg, "fd_eps", 1.0e-8)) if petsc_cfg is not None else 1.0e-8
    precond_drop_tol = float(getattr(petsc_cfg, "precond_drop_tol", 0.0)) if petsc_cfg is not None else 0.0
    precond_max_nnz_row = None
    if petsc_cfg is not None and hasattr(petsc_cfg, "precond_max_nnz_row"):
        try:
            precond_max_nnz_row = int(getattr(petsc_cfg, "precond_max_nnz_row"))
        except Exception:
            precond_max_nnz_row = None

    # Note: dedicated MPI backend does not support scaled unknowns/residuals.

    prefix = ""
    if petsc_cfg is not None:
        prefix = getattr(petsc_cfg, "options_prefix", "") or ""
    prefix = str(prefix)
    if prefix and not prefix.endswith("_"):
        prefix += "_"

    snes_type = "newtonls"
    linesearch_type = "bt"
    snes_monitor = bool(getattr(petsc_cfg, "snes_monitor", False)) if petsc_cfg is not None else False

    ksp_type = "gmres"
    ksp_rtol = 1.0e-6
    ksp_atol = 1.0e-12
    ksp_max_it = 200
    ksp_restart = 30

    if world_rank == 0:
        logger.info(
            "Parallel SNES fixed config: snes=%s/%s, ksp=%s, pc=asm+ilu(0), overlap=1",
            snes_type,
            linesearch_type,
            ksp_type,
        )

    u0 = np.asarray(u0, dtype=np.float64)
    n_layout = int(ctx.layout.n_dof())
    if u0.shape != (n_layout,):
        raise ValueError(f"u0 shape {u0.shape} incompatible with layout size {n_layout}.")

    from assembly.residual_local import ResidualLocalCtx, pack_local_to_layout, scatter_layout_to_local
    from parallel.dm_manager import global_to_local, local_state_to_global
    from assembly.residual_global import residual_only_owned_rows

    ctx_local = ResidualLocalCtx(layout=ctx.layout, ld=ld)

    def _layout_to_dm_vec(u_layout: np.ndarray):
        Xl_liq = dm_mgr.dm_liq.createLocalVec()
        Xl_gas = dm_mgr.dm_gas.createLocalVec()
        Xl_if = dm_mgr.dm_if.createLocalVec()
        Xl_liq.set(0.0)
        Xl_gas.set(0.0)
        Xl_if.set(0.0)
        aXl_liq = dm_mgr.dm_liq.getVecArray(Xl_liq)
        aXl_gas = dm_mgr.dm_gas.getVecArray(Xl_gas)
        aXl_if = Xl_if.getArray()
        scatter_layout_to_local(
            ctx_local,
            u_layout,
            aXl_liq,
            aXl_gas,
            aXl_if,
            rank=world_rank,
            owned_only=True,
        )
        return local_state_to_global(dm_mgr, Xl_liq, Xl_gas, Xl_if)

    def _dm_vec_to_layout(Xg_vec) -> np.ndarray:
        Xl_liq, Xl_gas, Xl_if = global_to_local(dm_mgr, Xg_vec)
        aXl_liq = dm_mgr.dm_liq.getVecArray(Xl_liq)
        aXl_gas = dm_mgr.dm_gas.getVecArray(Xl_gas)
        aXl_if = Xl_if.getArray()
        u_local = pack_local_to_layout(ctx_local, aXl_liq, aXl_gas, aXl_if, rank=world_rank)
        if comm.getSize() == 1:
            return u_local
        try:
            from mpi4py import MPI

            mpicomm = comm.tompi4py()
            return mpicomm.allreduce(u_local, op=MPI.SUM)
        except Exception as exc:
            raise RuntimeError("mpi4py is required for DM to layout gather in MPI mode.") from exc

    def residual_petsc_owned_rows(dm_mgr, ctx, X, Fg):
        """
        MPI: each rank writes only its owned rows of residual into Fg.
        """
        import os

        PETSc = _get_petsc()
        comm = PETSc.COMM_WORLD
        rank = int(comm.getRank())

        u_layout = _dm_vec_to_layout(X)

        rstart, rend = Fg.getOwnershipRange()
        ownership_range = (int(rstart), int(rend))

        r_owned = residual_only_owned_rows(u_layout, ctx, ownership_range)
        r_owned = np.asarray(r_owned, dtype=np.float64)

        if not np.all(np.isfinite(r_owned)):
            r_owned = np.where(np.isfinite(r_owned), r_owned, 1.0e20)

        nloc = int(rend - rstart)
        if r_owned.size != nloc:
            raise RuntimeError(
                f"[residual_petsc_owned_rows][rank={rank}] size mismatch: "
                f"len(r_owned)={r_owned.size} vs nloc={nloc} range=[{rstart},{rend})"
            )

        Fg.set(0.0)
        idx = np.arange(rstart, rend, dtype=PETSc.IntType)
        Fg.setValues(idx, r_owned, addv=PETSc.InsertMode.INSERT_VALUES)
        Fg.assemblyBegin()
        Fg.assemblyEnd()

        if os.environ.get("DROPLET_PETSC_DEBUG", "0") == "1":
            try:
                f_view = Fg.getArray()
                f_loc = np.array(f_view, dtype=np.float64, copy=True)
            finally:
                try:
                    Fg.restoreArray()
                except Exception:
                    pass
            err = float(np.max(np.abs(f_loc - r_owned))) if f_loc.size else 0.0
            logger.warning(
                "[DBG][rank=%d] OwnedRows writeback check: ||Fg_local - r_owned||_inf = %.3e",
                rank,
                err,
            )

        return r_owned

    def _dbg_log_Y(u_layout: np.ndarray, tag: str) -> None:
        if not DEBUG:
            return
        try:
            state_dbg = ctx.make_state_from_u(u_layout)
            Yg = np.asarray(state_dbg.Yg, dtype=np.float64)
            if Yg.ndim != 2:
                _dbg_rank("%s Yg shape=%r", tag, getattr(Yg, "shape", None))
                return
            cell_idx = 7
            if Yg.shape[1] <= cell_idx:
                _dbg_rank("%s Yg cell=%d out of range (ncell=%d)", tag, cell_idx, int(Yg.shape[1]))
                return
            Y = Yg[:, cell_idx]
            s = float(np.sum(Y)) if Y.size else 0.0
            mn = float(np.min(Y)) if Y.size else 0.0
            mx = float(np.max(Y)) if Y.size else 0.0
            idx = np.argsort(-Y)[:3] if Y.size else np.array([], dtype=np.int64)
            top = [(int(i), float(Y[i])) for i in idx]
            _dbg_rank(
                "%s Y@cell%d sum=%.6g min=%.3e max=%.3e top=%s",
                tag,
                cell_idx,
                s,
                mn,
                mx,
                top,
            )
        except Exception as exc:
            _dbg_rank("%s Y debug failed: %r", tag, exc)

    history_inf: List[float] = []
    last_inf = {"val": np.nan}
    t_solve0 = time.perf_counter()
    ctr = {"n_func_eval": 0, "n_jac_eval": 0, "ksp_its_total": 0}
    tim = {"time_func": 0.0, "time_jac": 0.0, "time_linear_total": 0.0}

    def snes_func(snes_obj, X, F):
        t0 = time.perf_counter()
        sig_changed = False
        if DEBUG:
            try:
                x_inf = float(X.norm(PETSc.NormType.NORM_INFINITY))
                x_2 = float(X.norm(PETSc.NormType.NORM_2))
                try:
                    x_view = X.getArray()
                    x_loc = np.array(x_view, dtype=np.float64, copy=True)
                finally:
                    try:
                        X.restoreArray()
                    except Exception:
                        pass
                stride = max(1, int(x_loc.size // 16)) if x_loc.size else 1
                x_chk = float(x_loc[::stride].sum()) if x_loc.size else 0.0
                sig = (x_inf, x_2, x_chk, int(x_loc.size))
                prev = ctx.meta.get("_dbg_prev_x_sig")
                if prev != sig:
                    ctx.meta["_dbg_prev_x_sig"] = sig
                    sig_changed = True
                    _dbg_rank(
                        "Xsig inf=%.3e 2=%.3e chk=%.6e nloc=%d",
                        x_inf,
                        x_2,
                        x_chk,
                        int(x_loc.size),
                    )
            except Exception:
                _dbg_rank("Xsig debug failed")
        if DEBUG:
            try:
                u_layout_dbg = _dm_vec_to_layout(X)
                _dbg_log_Y(u_layout_dbg, "AFTER_VEC2STATE_BEFORE_PROPS")
            except Exception as exc:
                _dbg_rank("Y debug before props failed: %r", exc)
        use_owned = bool(int(os.environ.get("DROPLET_PETSC_RESID_OWNED", "1")))
        if use_owned:
            residual_petsc_owned_rows(dm_mgr, ctx, X, F)
        else:
            residual_petsc(dm_mgr, ld, ctx, X, Fg=F)
        try:
            res_inf = float(F.norm(PETSc.NormType.NORM_INFINITY))
        except Exception:
            res_inf = float(F.norm())
        last_inf["val"] = res_inf
        ctr["n_func_eval"] += 1
        tim["time_func"] += (time.perf_counter() - t0)
        if DEBUG:
            if sig_changed:
                _dbg_rank("Fsig inf=%.3e", res_inf)
            try:
                rstart_f, rend_f = F.getOwnershipRange()
                try:
                    f_view = F.getArray()
                    f_loc = np.array(f_view, dtype=np.float64, copy=True)
                finally:
                    try:
                        F.restoreArray()
                    except Exception:
                        pass
                _dbg_rank(
                    "Fg(local) ||F||_inf=%.3e  local_n=%d  range=[%d,%d)",
                    float(np.max(np.abs(f_loc))),
                    int(f_loc.size),
                    int(rstart_f),
                    int(rend_f),
                )
            except Exception:
                _dbg_rank("Fg(local) debug failed")

    def snes_monitor_fn(snes_obj, its, fnorm):
        v = float(last_inf["val"])
        if np.isfinite(v):
            history_inf.append(v)
        if snes_monitor or verbose:
            if (its + 1) % log_every == 0:
                logger.info("snes iter=%d fnorm=%.3e res_inf=%.3e", its + 1, float(fnorm), v)

    snes = PETSc.SNES().create(comm=comm)
    if prefix:
        snes.setOptionsPrefix(prefix)

    try:
        snes.setDM(dm)
    except Exception:
        pass

    F = dm.createGlobalVec()
    try:
        snes.setFunction(snes_func, F)
    except TypeError:
        snes.setFunction(F, snes_func)
    try:
        snes.setFromOptions()
    except Exception:
        pass
    import os
    DEBUG = os.environ.get("DROPLET_PETSC_DEBUG", "0") == "1"
    DEBUG_ONCE = os.environ.get("DROPLET_PETSC_DEBUG_ONCE", "1") == "1"

    def _dbg(msg, *args):
        if DEBUG and world_rank == 0:
            logger.warning("[DBG] " + msg, *args)

    def _dbg_rank(msg, *args):
        if DEBUG:
            logger.warning("[DBG][rank=%d] " + msg, world_rank, *args)

    if DEBUG:
        X0 = None
        F0 = None
        try:
            X0 = _layout_to_dm_vec(u0)
            F0 = dm.createGlobalVec()
            residual_petsc(dm_mgr, ld, ctx, X0, Fg=F0)

            rstart_dm, rend_dm = F0.getOwnershipRange()
            _dbg_rank("DM ownership_range = [%d, %d)", int(rstart_dm), int(rend_dm))

            ld_range = None
            for key in ("ownership_range", "owned_range", "range_owned", "owned_dof_range"):
                if hasattr(ld, key):
                    ld_range = getattr(ld, key)
                    break
            if ld_range is not None:
                try:
                    _dbg_rank("LD ownership_range = %s", str(tuple(int(x) for x in ld_range)))
                except Exception:
                    _dbg_rank("LD ownership_range(raw) = %r", ld_range)
            else:
                _dbg_rank("LD ownership_range = <missing>")

            try:
                x0_view = X0.getArray()
                x0_loc = np.array(x0_view, dtype=np.float64, copy=True)
            finally:
                try:
                    X0.restoreArray()
                except Exception:
                    pass
            _dbg_rank(
                "X0 local ||x||_inf=%.3e  local_size=%d",
                float(np.max(np.abs(x0_loc))),
                int(x0_loc.size),
            )

            try:
                u_back = _dm_vec_to_layout(X0)
                du = u_back - u0
                _dbg_rank("Roundtrip ||u_back-u0||_inf=%.3e", float(np.max(np.abs(du))))
                imax = int(np.argmax(np.abs(du)))
                _dbg_rank(
                    "Roundtrip max@i=%d: u0=%.6e u_back=%.6e du=%.6e",
                    imax,
                    float(u0[imax]),
                    float(u_back[imax]),
                    float(du[imax]),
                )
            except Exception as exc:
                _dbg_rank("Roundtrip failed: %r", exc)
        finally:
            if X0 is not None:
                try:
                    X0.destroy()
                except Exception:
                    pass
            if F0 is not None:
                try:
                    F0.destroy()
                except Exception:
                    pass
        if DEBUG_ONCE:
            os.environ["DROPLET_PETSC_DEBUG"] = "0"

    try:
        snes.setType(snes_type)
    except Exception:
        logger.warning("Unknown snes_type='%s', falling back to newtonls", snes_type)
        snes.setType("newtonls")
    snes.setTolerances(rtol=f_rtol, atol=f_atol, max_it=max_outer_iter)
    try:
        ls = snes.getLineSearch()
        ls.setType(linesearch_type)
    except Exception:
        logger.debug("Unable to set linesearch_type='%s'", linesearch_type)

    _enable_snes_matrix_free(snes, prefix)

    X = dm.createGlobalVec()
    P = None
    fd_stats: Dict[str, Any] = {}
    if use_mfpc_sparse_fd:
        from assembly.build_sparse_fd_jacobian import build_sparse_fd_jacobian

        x_template = None
        try:
            x_template = dm.createGlobalVec()
            nloc = int(x_template.getLocalSize())
            n_glob = int(x_template.getSize())
        finally:
            if x_template is not None:
                try:
                    x_template.destroy()
                except Exception:
                    pass

        P_dm = PETSc.Mat().create(comm=comm)
        P_dm.setSizes(((nloc, n_glob), (nloc, n_glob)))
        try:
            P_dm.setType(PETSc.Mat.Type.AIJ)
        except Exception:
            P_dm.setType("aij")

        t0 = time.perf_counter()
        try:
            P, fd_stats = build_sparse_fd_jacobian(
                ctx,
                np.asarray(u0, dtype=np.float64),
                eps=fd_eps,
                drop_tol=precond_drop_tol,
                pattern=None,
                mat=P_dm,
            )
        except Exception as exc:
            raise RuntimeError("Failed to build mfpc_sparse_fd preconditioner.") from exc
        tim["time_jac"] += (time.perf_counter() - t0)
        ctr["n_jac_eval"] += 1
        try:
            p_range = tuple(int(v) for v in P.getOwnershipRange())
            x_range = tuple(int(v) for v in X.getOwnershipRange())
        except Exception:
            p_range = None
            x_range = None
        if p_range is not None and x_range is not None and p_range != x_range:
            raise RuntimeError(
                "mfpc_sparse_fd ownership range mismatch: "
                f"P={p_range} vs X={x_range}. "
                "Use jacobian_mode='mf' or update build_sparse_fd_jacobian to use DM ownership."
            )
        p_type_check = str(P.getType()).lower() if P is not None else ""
        if "aij" not in p_type_check:
            raise RuntimeError(f"mfpc_sparse_fd requires AIJ/MPIAIJ, got {p_type_check}")
    else:
        P = _create_identity_precond_mat(comm, dm)

    J = None
    try:
        J = PETSc.Mat().createSNESMF(snes)
    except Exception:
        try:
            J = PETSc.Mat().create(comm=comm)
            J.setType("mffd")
            J.setUp()
        except Exception:
            J = None
    try:
        snes.setJacobian(J=J, P=P, func=None)
    except TypeError:
        try:
            snes.setJacobian(J, P, None)
        except TypeError:
            try:
                snes.setJacobian(J, P)
            except TypeError:
                snes.setJacobian(P=P)

    ksp = snes.getKSP()

    try:
        ksp.setFromOptions()
    except Exception:
        pass

    try:
        ksp.setType(ksp_type)
    except Exception:
        logger.warning("Unknown ksp_type='%s', falling back to gmres", ksp_type)
        ksp.setType("gmres")

    ksp.setInitialGuessNonzero(False)
    ksp.setTolerances(rtol=ksp_rtol, atol=ksp_atol, max_it=ksp_max_it)

    try:
        ksp_type_eff = str(ksp.getType()).lower()
        if ksp_type_eff in ("gmres", "fgmres"):
            ksp.setGMRESRestart(ksp_restart)
        elif ksp_type_eff == "lgmres" and hasattr(ksp, "setLGMRESRestart"):
            ksp.setLGMRESRestart(ksp_restart)
    except Exception:
        logger.debug("Unable to set restart for ksp_type='%s'", ksp.getType())

    diag_pc = apply_structured_pc(
        ksp=ksp,
        cfg=cfg,
        layout=ctx.layout,
        A=J,
        P=P,
        pc_type_override="asm",
    )
    snes.setMonitor(snes_monitor_fn)
    _finalize_ksp_config(ksp, diag_pc, from_options=False)

    ksp_state = {"monitor_enabled": False, "last_its": 0}
    ksp_t = {"in_solve": False, "t0": 0.0}

    def _ksp_monitor(ksp_obj, its, rnorm):
        if its == 0:
            if ksp_t["in_solve"]:
                tim["time_linear_total"] += (time.perf_counter() - ksp_t["t0"])
            ksp_t["in_solve"] = True
            ksp_t["t0"] = time.perf_counter()
            ksp_state["last_its"] = 0
            return
        if its > ksp_state["last_its"]:
            ctr["ksp_its_total"] += int(its - ksp_state["last_its"])
            ksp_state["last_its"] = int(its)

    try:
        ksp.setMonitor(_ksp_monitor)
        ksp_state["monitor_enabled"] = True
    except Exception:
        pass

    X_init = _layout_to_dm_vec(u0)
    try:
        X_init.copy(X)
    except Exception:
        X.set(0.0)
        X.axpy(1.0, X_init)

    snes.solve(None, X)
    if ksp_t.get("in_solve", False):
        tim["time_linear_total"] += (time.perf_counter() - ksp_t["t0"])
        ksp_t["in_solve"] = False

    u_final = _dm_vec_to_layout(X)

    reason = int(snes.getConvergedReason())
    ksp_reason = int(ksp.getConvergedReason())
    ksp_it = int(ksp.getIterationNumber())
    converged = reason > 0
    n_iter = int(snes.getIterationNumber())

    res_final = residual_only(u_final, ctx)
    res_norm_2 = float(np.linalg.norm(res_final))
    res_norm_inf = float(np.linalg.norm(res_final, ord=np.inf))

    if not converged and world_rank == 0:
        logger.warning("SNES not converged: reason=%d res_inf=%.3e", reason, res_norm_inf)

    j_type = "none"
    p_type = "none"
    x_type = ""
    f_type = ""
    try:
        x_type = str(X.getType())
        f_type = str(F.getType())
    except Exception:
        pass
    try:
        if J is not None:
            j_type = str(J.getType()).lower()
    except Exception:
        j_type = "none"
    try:
        if P is not None:
            p_type = str(P.getType()).lower()
    except Exception:
        p_type = "none"

    ksp_a_type = ""
    ksp_p_type = ""
    ksp_a_is_p = False
    try:
        ksp_a, ksp_p = ksp.getOperators()
        if ksp_a is not None:
            ksp_a_type = str(ksp_a.getType()).lower()
        if ksp_p is not None:
            ksp_p_type = str(ksp_p.getType()).lower()
        if ksp_a is not None and ksp_p is not None:
            try:
                ksp_a_is_p = bool(ksp_a.handle == ksp_p.handle)
            except Exception:
                ksp_a_is_p = bool(ksp_a is ksp_p)
    except Exception:
        pass

    if not j_type or j_type == "none":
        j_type = "mffd"

    extra: Dict[str, Any] = {
        "snes_reason": reason,
        "snes_reason_str": str(reason),
        "snes_type": snes.getType(),
        "linesearch_type": linesearch_type,
        "ksp_type": ksp.getType(),
        "pc_type": ksp.getPC().getType(),
        "ksp_reason": ksp_reason,
        "ksp_reason_str": str(ksp_reason),
        "ksp_it": ksp_it,
        "snes_iter": n_iter,
        "jacobian_mode": jacobian_mode,
        "n_func_eval": int(ctr["n_func_eval"]),
        "n_jac_eval": int(ctr["n_jac_eval"]),
        "ksp_its_total": int(ctr["ksp_its_total"]),
        "time_func": float(tim["time_func"]),
        "time_jac": float(tim["time_jac"]),
        "time_linear_total": float(tim["time_linear_total"]),
        "time_total": float(time.perf_counter() - t_solve0),
        "snes_mf_enabled": True,
        "pc_type_overridden": True,
        "J_mat_type": j_type,
        "P_mat_type": p_type,
    }
    if ksp_a_type:
        extra["KSP_A_type"] = ksp_a_type
    if ksp_p_type:
        extra["KSP_P_type"] = ksp_p_type
    extra["KSP_A_is_P"] = bool(ksp_a_is_p)
    if use_mfpc_sparse_fd:
        stats = dict(fd_stats) if isinstance(fd_stats, dict) else {}
        stats.setdefault("eps", float(fd_eps))
        stats.setdefault("drop_tol", float(precond_drop_tol))
        if precond_max_nnz_row is not None:
            stats.setdefault("precond_max_nnz_row", int(precond_max_nnz_row))
        extra["mfpc_sparse_fd"] = stats
    extra["parallel_backend"] = "petsc_snes_parallel"
    extra["comm_kind"] = "world"

    try:
        extra["comm_size"] = int(comm.getSize())
    except Exception:
        pass
    if x_type:
        extra["X_vec_type"] = x_type
    if f_type:
        extra["F_vec_type"] = f_type
    try:
        extra["X_local_size"] = int(X.getLocalSize())
        extra["X_ownership_range"] = tuple(int(v) for v in X.getOwnershipRange())
    except Exception:
        pass
    try:
        extra["dm_type"] = str(dm.getType())
    except Exception:
        pass
    if diag_pc:
        extra["pc_structured"] = dict(diag_pc)
    if history_inf:
        extra["history_res_inf"] = list(history_inf)

    diag = NonlinearDiagnostics(
        converged=bool(converged),
        method=f"snes:{snes.getType()}",
        n_iter=int(n_iter),
        res_norm_2=res_norm_2,
        res_norm_inf=res_norm_inf,
        history_res_inf=list(history_inf),
        message=None if converged else f"SNES diverged (reason={reason})",
        extra=extra,
    )

    if world_rank == 0 and verbose:
        logger.info(
            "PETSc parallel SNES done: conv=%s reason=%d its=%d |F|_inf=%.3e ksp_its_total=%d",
            converged,
            reason,
            n_iter,
            res_norm_inf,
            int(ctr["ksp_its_total"]),
        )

    return NonlinearSolveResult(u=np.asarray(u_final, dtype=np.float64), diag=diag)
