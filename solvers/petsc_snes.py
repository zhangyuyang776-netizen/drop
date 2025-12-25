"""
PETSc SNES nonlinear solver wrapper (serial-only stage).
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List

import numpy as np

from assembly.build_precond_aij import build_precond_mat_aij_from_A, fill_precond_mat_aij_from_A
from assembly.build_sparse_fd_jacobian import build_sparse_fd_jacobian
from assembly.jacobian_pattern import build_jacobian_pattern
from assembly.residual_global import build_global_residual, residual_only
from solvers.nonlinear_context import NonlinearContext
from solvers.nonlinear_types import NonlinearDiagnostics, NonlinearSolveResult
from solvers.petsc_linear import apply_fieldsplit_subksp_defaults, apply_structured_pc, _normalize_pc_type

logger = logging.getLogger(__name__)


def _get_petsc():
    try:
        from petsc4py import PETSc
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("petsc4py is required for PETSc SNES backend.") from exc
    return PETSc


def _cfg_get(obj, name: str, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _finalize_ksp_config(ksp, diag_pc) -> None:
    try:
        ksp.setFromOptions()
    except Exception:
        pass
    try:
        ksp.setUp()
    except Exception:
        pass
    apply_fieldsplit_subksp_defaults(ksp, diag_pc)
    try:
        ksp.setUp()
    except Exception:
        pass


def solve_nonlinear_petsc(
    ctx: NonlinearContext,
    u0: np.ndarray,
) -> NonlinearSolveResult:
    """
    Solve F(u)=0 using PETSc SNES with optional scaling.
    """
    PETSc = _get_petsc()
    world = PETSc.COMM_WORLD
    world_size = world.getSize()
    world_rank = world.getRank()

    cfg = ctx.cfg
    petsc_cfg = cfg.petsc

    mpi_mode = "self"
    try:
        mpi_mode = str(getattr(petsc_cfg, "mpi_mode", "self")).strip().lower()
    except Exception:
        mpi_mode = "self"

    if world_size > 1:
        if mpi_mode in ("self", "redundant", "serial"):
            if world_rank == 0:
                logger.warning(
                    "MPI detected (size=%d) but petsc.mpi_mode='%s': using COMM_SELF (redundant per-rank solve) for Stage 5.0.",
                    world_size,
                    mpi_mode,
                )
            comm = PETSc.COMM_SELF
        elif mpi_mode in ("world", "distributed"):
            raise NotImplementedError(
                "petsc.mpi_mode='world' requires Stage 5.1+ (DM-based local assembly). "
                "For Stage 5.0, set petsc.mpi_mode='self'."
            )
        else:
            raise ValueError(f"Unknown petsc.mpi_mode='{mpi_mode}' (use 'self' for Stage 5.0).")
    else:
        comm = world

    nl = getattr(cfg, "nonlinear", None)
    if nl is None or not getattr(nl, "enabled", False):
        raise ValueError("Nonlinear solver requested but cfg.nonlinear.enabled is False or missing.")

    max_outer_iter = int(getattr(nl, "max_outer_iter", 20))
    f_rtol = float(getattr(nl, "f_rtol", 1.0e-6))
    f_atol = float(getattr(nl, "f_atol", 1.0e-10))
    use_scaled_u = bool(getattr(nl, "use_scaled_unknowns", True))
    use_scaled_res = bool(getattr(nl, "use_scaled_residual", True))
    residual_scale_floor = float(getattr(nl, "residual_scale_floor", 1.0e-12))
    verbose = bool(getattr(nl, "verbose", False))
    log_every = int(getattr(nl, "log_every", 5))
    if log_every < 1:
        log_every = 1

    prefix = getattr(petsc_cfg, "options_prefix", "")
    if prefix is None:
        prefix = ""
    prefix = str(prefix)
    if prefix and not prefix.endswith("_"):
        prefix += "_"

    snes_type = str(getattr(petsc_cfg, "snes_type", "newtonls"))
    linesearch_type = str(getattr(petsc_cfg, "linesearch_type", "bt"))
    jacobian_mode = str(getattr(petsc_cfg, "jacobian_mode", "fd")).lower()
    snes_monitor = bool(getattr(petsc_cfg, "snes_monitor", False))
    if jacobian_mode in ("mfpc_sparse", "mfpc_sparse_fd"):
        jacobian_mode = "mfpc_sparse_fd"
    elif jacobian_mode in ("mfpc_aij", "mfpc_aija"):
        jacobian_mode = "mfpc_aija"

    ksp_type = str(getattr(petsc_cfg, "ksp_type", "gmres"))
    linear_cfg = getattr(getattr(cfg, "solver", None), "linear", None)
    pc_type = _cfg_get(linear_cfg, "pc_type", None)
    if pc_type is None:
        pc_type = str(getattr(petsc_cfg, "pc_type", "ilu"))
    else:
        pc_type = str(pc_type)
    pc_type = _normalize_pc_type(pc_type) or "ilu"
    petsc_rtol = float(getattr(petsc_cfg, "rtol", 1e-8))
    petsc_atol = float(getattr(petsc_cfg, "atol", 1e-12))
    petsc_max_it = int(getattr(petsc_cfg, "max_it", 200))
    restart = int(getattr(petsc_cfg, "restart", 30))
    precond_drop_tol = float(getattr(petsc_cfg, "precond_drop_tol", 0.0))
    if precond_drop_tol < 0.0:
        precond_drop_tol = 0.0
    precond_max_nnz_row = getattr(petsc_cfg, "precond_max_nnz_row", None)
    if precond_max_nnz_row is not None:
        try:
            precond_max_nnz_row = int(precond_max_nnz_row)
        except Exception:
            precond_max_nnz_row = None

    u0 = np.asarray(u0, dtype=np.float64)
    scale = np.asarray(ctx.scale_u, dtype=np.float64)
    if u0.shape != scale.shape:
        raise ValueError(f"u0 shape {u0.shape} incompatible with scale_u {scale.shape}")
    scale_safe = np.where(scale > 0.0, scale, 1.0)

    if use_scaled_u:
        x0 = u0 / scale_safe
    else:
        x0 = u0

    scale_F = np.ones_like(u0, dtype=np.float64)
    if use_scaled_res:
        try:
            res_ref = residual_only(u0, ctx)
            if res_ref.shape != u0.shape:
                raise ValueError(
                    f"residual_only shape {res_ref.shape} incompatible with unknown shape {u0.shape}"
                )

            abs_ref = np.abs(res_ref)
            finite_all = abs_ref[np.isfinite(abs_ref)]
            if finite_all.size == 0:
                logger.warning("residual_only returned no finite entries; disabling use_scaled_residual.")
                use_scaled_res = False
            else:
                global_max = float(finite_all.max())
                if global_max <= 0.0:
                    use_scaled_res = False
                else:
                    block_rel_threshold = 1.0e-3
                    scale_F = np.ones_like(abs_ref, dtype=np.float64)
                    blocks = getattr(ctx.layout, "blocks", {}) or {}
                    for _, sl in blocks.items():
                        blk = abs_ref[sl]
                        finite = blk[np.isfinite(blk)]
                        if finite.size == 0:
                            continue
                        s_blk = float(np.percentile(finite, 90.0))
                        if not np.isfinite(s_blk) or s_blk <= 0.0:
                            continue
                        if s_blk < block_rel_threshold * global_max:
                            continue
                        if s_blk < residual_scale_floor:
                            s_blk = residual_scale_floor
                        scale_F[sl] = s_blk

                    bad_mask = ~(np.isfinite(scale_F) & (scale_F > 0.0))
                    if np.any(bad_mask):
                        scale_F[bad_mask] = 1.0

                ctx.meta["residual_scale_F"] = scale_F.copy()
        except Exception as exc:
            logger.warning(
                "Failed to build residual scaling; disabling use_scaled_residual. err=%s",
                exc,
            )
            use_scaled_res = False
            scale_F = np.ones_like(u0, dtype=np.float64)

    scale_F_safe = np.where(scale_F > 0.0, scale_F, 1.0)

    history_inf: List[float] = []
    last_inf = {"val": np.nan}
    t_solve0 = time.perf_counter()
    ctr = {
        "n_func_eval": 0,
        "n_jac_eval": 0,
        "ksp_its_total": 0,
    }
    tim = {
        "time_func": 0.0,
        "time_jac": 0.0,
        "time_linear_total": 0.0,
    }

    def _x_to_u_phys(x_arr: np.ndarray) -> np.ndarray:
        return x_arr * scale_safe if use_scaled_u else x_arr

    def _res_phys_to_res_eval(res_phys: np.ndarray) -> np.ndarray:
        return res_phys / scale_F_safe if use_scaled_res else res_phys

    class _MFJacShellCtx:
        def __init__(
            self,
            *,
            ctx_ref: NonlinearContext,
            scale_u: np.ndarray,
            scale_F: np.ndarray,
            use_scaled_u: bool,
            use_scaled_res: bool,
            fd_eps: float,
        ) -> None:
            self.ctx = ctx_ref
            self.scale_u = np.asarray(scale_u, dtype=np.float64)
            self.scale_F = np.asarray(scale_F, dtype=np.float64)
            self.use_scaled_u = bool(use_scaled_u)
            self.use_scaled_res = bool(use_scaled_res)
            self.fd_eps = float(fd_eps)

            self.x0: np.ndarray | None = None
            self.F0: np.ndarray | None = None
            self.n_mf_mult = 0
            self.n_mf_func_eval = 0
            self.time_mf_func = 0.0

        def _x_to_u_phys(self, x: np.ndarray) -> np.ndarray:
            return x * self.scale_u if self.use_scaled_u else x

        def _F_phys_to_eval(self, F_phys: np.ndarray) -> np.ndarray:
            return F_phys / self.scale_F if self.use_scaled_res else F_phys

        def set_base(self, x0: np.ndarray) -> None:
            x0 = np.asarray(x0, dtype=np.float64)
            self.x0 = x0.copy()
            u0 = self._x_to_u_phys(self.x0)
            t0 = time.perf_counter()
            F0_phys = residual_only(u0, self.ctx)
            if not np.all(np.isfinite(F0_phys)):
                F0_phys = np.where(np.isfinite(F0_phys), F0_phys, 1.0e20)
            self.n_mf_func_eval += 1
            self.time_mf_func += (time.perf_counter() - t0)
            self.F0 = self._F_phys_to_eval(F0_phys)

        def mult(self, mat, X, Y):
            try:
                v_view = X.getArray(readonly=True)
            except TypeError:
                v_view = X.getArray()
            v = np.asarray(v_view, dtype=np.float64)
            if self.x0 is None or self.F0 is None:
                self.set_base(np.zeros_like(v))

            self.n_mf_mult += 1
            nv = float(np.linalg.norm(v))
            y_view = Y.getArray()
            if nv == 0.0:
                y_view[:] = 0.0
                return

            nx = float(np.linalg.norm(self.x0))
            h = self.fd_eps * (1.0 + nx) / nv
            if h == 0.0:
                h = self.fd_eps

            x1 = self.x0 + h * v
            u1 = self._x_to_u_phys(x1)
            t0 = time.perf_counter()
            F1_phys = residual_only(u1, self.ctx)
            if not np.all(np.isfinite(F1_phys)):
                F1_phys = np.where(np.isfinite(F1_phys), F1_phys, 1.0e20)
            self.n_mf_func_eval += 1
            self.time_mf_func += (time.perf_counter() - t0)
            F1 = self._F_phys_to_eval(F1_phys)
            y_view[:] = (F1 - self.F0) / h

    def snes_func(snes, X, F):
        t0 = time.perf_counter()
        try:
            x_view = X.getArray(readonly=True)
        except TypeError:
            x_view = X.getArray()
        u_phys = _x_to_u_phys(np.asarray(x_view, dtype=np.float64))

        res_phys = residual_only(u_phys, ctx)
        if not np.all(np.isfinite(res_phys)):
            res_phys = np.where(np.isfinite(res_phys), res_phys, 1.0e20)
        res_eval = _res_phys_to_res_eval(res_phys)

        f_view = F.getArray()
        f_view[:] = res_eval

        last_inf["val"] = float(np.linalg.norm(res_eval, ord=np.inf))
        ctr["n_func_eval"] += 1
        tim["time_func"] += (time.perf_counter() - t0)

    ksp_state = {"monitor_enabled": False, "last_its": 0}

    def snes_monitor_fn(snes, its, fnorm):
        v = float(last_inf["val"])
        if np.isfinite(v):
            history_inf.append(v)
        if not ksp_state["monitor_enabled"]:
            try:
                ksp = snes.getKSP()
                ctr["ksp_its_total"] += int(ksp.getIterationNumber())
            except Exception:
                pass
        if snes_monitor or verbose:
            if (its + 1) % log_every == 0:
                logger.info(
                    "snes iter=%d fnorm=%.3e res_inf=%.3e",
                    its + 1,
                    float(fnorm),
                    float(last_inf["val"]),
                )

    N = int(u0.size)
    snes = PETSc.SNES().create(comm=comm)
    snes.setOptionsPrefix(prefix)

    F = PETSc.Vec().createSeq(N, comm=comm)
    snes.setFunction(snes_func, F)

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

    pc_overridden = False
    pc_overridden_dense = False
    precond_diag: Dict[str, Any] = {}
    mf_ctx: _MFJacShellCtx | None = None
    J = None
    P = None
    if jacobian_mode == "mfpc_sparse_fd":
        fd_eps = float(getattr(petsc_cfg, "fd_eps", 1.0e-8))
        mf_ctx = _MFJacShellCtx(
            ctx_ref=ctx,
            scale_u=scale_safe,
            scale_F=scale_F_safe,
            use_scaled_u=use_scaled_u,
            use_scaled_res=use_scaled_res,
            fd_eps=fd_eps,
        )
        try:
            J = PETSc.Mat().createPython([N, N], comm=comm)
        except Exception:
            J = PETSc.Mat().createShell([N, N], comm=comm)
        J.setPythonContext(mf_ctx)
        J.setUp()

        pattern = ctx.meta.get("jac_pattern")
        if pattern is None:
            pattern = build_jacobian_pattern(cfg, ctx.grid_ref, ctx.layout)
            ctx.meta["jac_pattern"] = pattern

        P, precond_diag = build_sparse_fd_jacobian(
            ctx,
            x0,
            eps=fd_eps,
            drop_tol=precond_drop_tol,
            pattern=pattern,
        )

        def jac_func(snes_obj, X_vec, J_mat, P_mat):
            t0 = time.perf_counter()
            ctr["n_jac_eval"] += 1
            try:
                x0_view = X_vec.getArray(readonly=True)
            except TypeError:
                x0_view = X_vec.getArray()
            x0_arr = np.asarray(x0_view, dtype=np.float64)
            if mf_ctx is not None:
                mf_ctx.set_base(x0_arr)

            _P, diagP = build_sparse_fd_jacobian(
                ctx,
                x0_arr,
                eps=fd_eps,
                drop_tol=precond_drop_tol,
                pattern=pattern,
                mat=P_mat,
            )
            precond_diag.update(diagP)
            tim["time_jac"] += (time.perf_counter() - t0)
            return True

        try:
            snes.setJacobian(J=J, P=P, func=jac_func)
        except TypeError:
            snes.setJacobian(jac_func, J, P)
    elif jacobian_mode == "mfpc_aija":
        fd_eps = float(getattr(petsc_cfg, "fd_eps", 1.0e-8))
        mf_ctx = _MFJacShellCtx(
            ctx_ref=ctx,
            scale_u=scale_safe,
            scale_F=scale_F_safe,
            use_scaled_u=use_scaled_u,
            use_scaled_res=use_scaled_res,
            fd_eps=fd_eps,
        )
        try:
            J = PETSc.Mat().createPython([N, N], comm=comm)
        except Exception:
            J = PETSc.Mat().createShell([N, N], comm=comm)
        J.setPythonContext(mf_ctx)
        J.setUp()

        u0_phys = _x_to_u_phys(np.asarray(x0, dtype=np.float64))
        P, precond_diag = build_precond_mat_aij_from_A(
            ctx,
            u_phys=u0_phys,
            drop_tol=precond_drop_tol,
            comm=comm,
            max_nnz_row=precond_max_nnz_row,
        )

        apply_scale = use_scaled_res or use_scaled_u
        L_vec = None
        R_vec = None
        if apply_scale:
            L_np = (1.0 / scale_F_safe) if use_scaled_res else np.ones_like(scale_F_safe)
            R_np = scale_safe if use_scaled_u else np.ones_like(scale_safe)
            idx = np.arange(N, dtype=PETSc.IntType)
            L_vec = PETSc.Vec().createSeq(N, comm=comm)
            R_vec = PETSc.Vec().createSeq(N, comm=comm)
            L_vec.setValues(idx, L_np.astype(PETSc.ScalarType, copy=False))
            R_vec.setValues(idx, R_np.astype(PETSc.ScalarType, copy=False))
            L_vec.assemblyBegin()
            L_vec.assemblyEnd()
            R_vec.assemblyBegin()
            R_vec.assemblyEnd()
            P.diagonalScale(L_vec, R_vec)

        def jac_func(snes_obj, X_vec, J_mat, P_mat):
            t0 = time.perf_counter()
            ctr["n_jac_eval"] += 1
            try:
                x0_view = X_vec.getArray(readonly=True)
            except TypeError:
                x0_view = X_vec.getArray()
            x0_arr = np.asarray(x0_view, dtype=np.float64)
            if mf_ctx is not None:
                mf_ctx.set_base(x0_arr)
            u_phys = _x_to_u_phys(x0_arr)

            diagP = fill_precond_mat_aij_from_A(
                P_mat,
                ctx,
                u_phys,
                drop_tol=precond_drop_tol,
            )
            precond_diag.update(diagP)
            if L_vec is not None and R_vec is not None:
                P_mat.diagonalScale(L_vec, R_vec)
            tim["time_jac"] += (time.perf_counter() - t0)
            return True

        try:
            snes.setJacobian(J=J, P=P, func=jac_func)
        except TypeError:
            snes.setJacobian(jac_func, J, P)
    elif jacobian_mode == "mf":
        try:
            J = PETSc.Mat().createSNESMF(snes)
            P = J
            snes.setJacobian(J, J)
        except Exception:
            fd_eps = float(getattr(petsc_cfg, "fd_eps", 1.0e-8))
            mf_ctx = _MFJacShellCtx(
                ctx_ref=ctx,
                scale_u=scale_safe,
                scale_F=scale_F_safe,
                use_scaled_u=use_scaled_u,
                use_scaled_res=use_scaled_res,
                fd_eps=fd_eps,
            )
            try:
                J = PETSc.Mat().createPython([N, N], comm=comm)
            except Exception:
                J = PETSc.Mat().createShell([N, N], comm=comm)
            J.setPythonContext(mf_ctx)
            J.setUp()

            def jac_func(snes_obj, X_vec, J_mat, P_mat):
                t0 = time.perf_counter()
                ctr["n_jac_eval"] += 1
                try:
                    x0_view = X_vec.getArray(readonly=True)
                except TypeError:
                    x0_view = X_vec.getArray()
                x0_arr = np.asarray(x0_view, dtype=np.float64)
                if mf_ctx is not None:
                    mf_ctx.set_base(x0_arr)
                tim["time_jac"] += (time.perf_counter() - t0)
                return True

            P = J
            try:
                snes.setJacobian(J=J, P=J, func=jac_func)
            except TypeError:
                snes.setJacobian(jac_func, J, J)
        if pc_type in ("lu", "ilu"):
            pc_type = "none"
            pc_overridden = True
    else:
        J = PETSc.Mat().createDense([N, N], comm=comm)
        J.setUp()
        P = J
        if pc_type in ("ilu", "jacobi", ""):
            pc_type = "lu"
            pc_overridden_dense = True

        fd_eps = float(getattr(petsc_cfg, "fd_eps", 1.0e-8))

        def jac_func(snes_obj, X_vec, J_mat, P_mat):
            t0 = time.perf_counter()
            ctr["n_jac_eval"] += 1
            try:
                x0_view = X_vec.getArray(readonly=True)
            except TypeError:
                x0_view = X_vec.getArray()
            x0_arr = np.asarray(x0_view, dtype=np.float64)
            nloc = x0_arr.size

            u0_phys = _x_to_u_phys(x0_arr)
            r0_phys = residual_only(u0_phys, ctx)
            r0_eval = _res_phys_to_res_eval(r0_phys)

            J_mat.zeroEntries()
            rows = np.arange(nloc, dtype=PETSc.IntType)
            x_work = x0_arr.copy()

            for j in range(nloc):
                dx = fd_eps * (1.0 + abs(x0_arr[j]))
                if dx == 0.0:
                    dx = fd_eps
                x_work[j] = x0_arr[j] + dx

                uj_phys = _x_to_u_phys(x_work)
                rj_phys = residual_only(uj_phys, ctx)
                rj_eval = _res_phys_to_res_eval(rj_phys)

                col = np.asarray((rj_eval - r0_eval) / dx, dtype=PETSc.ScalarType)
                cols = np.array([j], dtype=PETSc.IntType)
                J_mat.setValues(rows, cols, col.reshape(-1, 1), addv=False)

                x_work[j] = x0_arr[j]

            J_mat.assemblyBegin()
            J_mat.assemblyEnd()

            if P_mat is not J_mat:
                P_mat.assemblyBegin()
                P_mat.assemblyEnd()
            tim["time_jac"] += (time.perf_counter() - t0)
            return True

        try:
            snes.setJacobian(J=J, P=J, func=jac_func)
        except TypeError:
            snes.setJacobian(jac_func, J, J)

    ksp = snes.getKSP()
    Aop = J
    Pop = P if P is not None else J
    ksp.setOperators(Aop, Pop)
    try:
        ksp.setType(ksp_type)
    except Exception:
        logger.warning("Unknown ksp_type='%s', falling back to gmres", ksp_type)
        ksp.setType("gmres")
    pc = ksp.getPC()
    if str(pc_type).lower() != "fieldsplit":
        try:
            pc.setType(pc_type)
        except Exception:
            logger.warning("Unknown pc_type='%s', falling back to jacobi", pc_type)
            pc.setType("jacobi")

    ksp.setInitialGuessNonzero(False)
    ksp.setTolerances(rtol=petsc_rtol, atol=petsc_atol, max_it=petsc_max_it)

    try:
        ksp_type_eff = str(ksp.getType()).lower()
        if ksp_type_eff in ("gmres", "fgmres"):
            ksp.setGMRESRestart(restart)
        elif ksp_type_eff == "lgmres":
            if hasattr(ksp, "setLGMRESRestart"):
                ksp.setLGMRESRestart(restart)
    except Exception:
        logger.debug("Unable to set restart for ksp_type='%s'", ksp.getType())

    A_for_range = Pop
    diag_pc = apply_structured_pc(
        ksp=ksp,
        cfg=cfg,
        layout=ctx.layout,
        A=A_for_range,
        P=Pop,
        pc_type_override=_normalize_pc_type(pc_type),
    )

    snes.setMonitor(snes_monitor_fn)
    snes.setFromOptions()
    try:
        ksp.setFromOptions()
    except Exception:
        pass
    _finalize_ksp_config(ksp, diag_pc)
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

    X = PETSc.Vec().createSeq(N, comm=comm)
    X.getArray()[:] = np.asarray(x0, dtype=np.float64)
    snes.solve(None, X)
    if ksp_t.get("in_solve", False):
        tim["time_linear_total"] += (time.perf_counter() - ksp_t["t0"])
        ksp_t["in_solve"] = False

    x_final = np.asarray(X.getArray(), dtype=np.float64).copy()
    u_final = _x_to_u_phys(x_final)

    reason = int(snes.getConvergedReason())
    ksp_reason = int(ksp.getConvergedReason())
    ksp_it = int(ksp.getIterationNumber())
    converged = reason > 0
    n_iter = int(snes.getIterationNumber())

    res_final = residual_only(u_final, ctx)
    res_norm_2 = float(np.linalg.norm(res_final))
    res_norm_inf = float(np.linalg.norm(res_final, ord=np.inf))

    extra: Dict[str, Any] = {
        "snes_reason": reason,
        "snes_reason_str": str(reason),
        "snes_type": snes.getType(),
        "linesearch_type": linesearch_type,
        "ksp_type": ksp.getType(),
        "pc_type": pc.getType(),
        "ksp_reason": ksp_reason,
        "ksp_reason_str": str(ksp_reason),
        "ksp_it": ksp_it,
        "snes_iter": n_iter,
        "jacobian_mode": jacobian_mode,
        "petsc_snes_reason": int(reason),
        "petsc_snes_reason_str": str(reason),
        "n_func_eval": int(ctr["n_func_eval"]),
        "n_jac_eval": int(ctr["n_jac_eval"]),
        "ksp_its_total": int(ctr["ksp_its_total"]),
        "time_func": float(tim["time_func"]),
        "time_jac": float(tim["time_jac"]),
        "time_linear_total": float(tim["time_linear_total"]),
        "time_total": float(time.perf_counter() - t_solve0),
    }
    if diag_pc:
        extra["pc_structured"] = dict(diag_pc)
    if jacobian_mode in ("mfpc_aija", "mfpc_sparse_fd"):
        extra["precond_drop_tol"] = float(precond_drop_tol)
        if precond_max_nnz_row is not None:
            extra["precond_max_nnz_row"] = int(precond_max_nnz_row)
        if precond_diag:
            extra["precond_diag"] = dict(precond_diag)
    if mf_ctx is not None:
        extra["n_mf_mult"] = int(mf_ctx.n_mf_mult)
        extra["n_mf_func_eval"] = int(mf_ctx.n_mf_func_eval)
        extra["time_mf_func"] = float(mf_ctx.time_mf_func)
    if pc_overridden:
        extra["pc_type_overridden"] = True
    if pc_overridden_dense:
        extra["pc_type_overridden_dense"] = True

    if not history_inf and np.isfinite(last_inf["val"]):
        history_inf.append(float(last_inf["val"]))

    message = None if converged else f"SNES diverged (reason={reason})"
    if not converged:
        logger.warning("SNES not converged: reason=%d res_inf=%.3e", reason, res_norm_inf)

    diag = NonlinearDiagnostics(
        converged=converged,
        method=f"snes:{snes.getType()}",
        n_iter=n_iter,
        res_norm_2=res_norm_2,
        res_norm_inf=res_norm_inf,
        history_res_inf=history_inf,
        message=message,
        extra=extra,
    )

    if verbose:
        try:
            _, diag_res = build_global_residual(u_final, ctx)
            diag.extra["assembly_diag"] = diag_res
        except Exception as exc:
            diag.extra["assembly_diag_error"] = str(exc)

    return NonlinearSolveResult(u=u_final, diag=diag)
