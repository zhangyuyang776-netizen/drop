"""
PETSc-based linear solver backend (mirrors SciPy interface).

Design goals:
- Pure PETSc/NumPy (no SciPy dependency).
- API mirrors SciPy backend: returns LinearSolveResult.
- Strict shape checks; no state/layout mutations.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Mapping, Optional

import numpy as np

from core.types import CaseConfig
from solvers.linear_types import LinearSolveResult

logger = logging.getLogger(__name__)


def _cfg_get(obj, name: str, default=None):
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _normalize_pc_type(pc_type: Optional[str]) -> Optional[str]:
    if pc_type is None:
        return None
    val = str(pc_type).strip().lower()
    if val in ("blockjacobi", "block_jacobi", "block-jacobi"):
        return "bjacobi"
    return val


def _configure_pc_asm_or_bjacobi_subksps(
    pc,
    *,
    sub_ksp_type: Optional[str],
    sub_pc_type: Optional[str],
    sub_ksp_rtol: Optional[float] = None,
    sub_ksp_atol: Optional[float] = None,
    sub_ksp_max_it: Optional[int] = None,
    overlap: Optional[int] = None,
) -> Optional[int]:
    pctype = _normalize_pc_type(pc.getType())
    if pctype not in ("asm", "bjacobi"):
        return None

    if pctype == "asm" and overlap is not None:
        try:
            pc.setASMOverlap(int(overlap))
        except Exception:
            pass
    try:
        pc.setUp()
    except Exception:
        pass

    sub = None
    if pctype == "asm":
        try:
            sub = pc.getASMSubKSP()
        except Exception:
            sub = None
    else:
        for name in ("getBJACOBISubKSP", "getBJacobiSubKSP"):
            if hasattr(pc, name):
                try:
                    sub = getattr(pc, name)()
                except Exception:
                    sub = None
                break

    if sub is None:
        return None

    if isinstance(sub, tuple) and len(sub) == 2:
        _, subksps = sub
    else:
        subksps = sub

    for subksp in subksps:
        if sub_ksp_type:
            try:
                subksp.setType(str(sub_ksp_type))
            except Exception:
                pass
        if sub_ksp_rtol is not None or sub_ksp_atol is not None or sub_ksp_max_it is not None:
            try:
                subksp.setTolerances(rtol=sub_ksp_rtol, atol=sub_ksp_atol, max_it=sub_ksp_max_it)
            except Exception:
                pass
        try:
            spc = subksp.getPC()
            if sub_pc_type:
                spc.setType(str(sub_pc_type))
        except Exception:
            pass
        try:
            subksp.setFromOptions()
        except Exception:
            pass
        try:
            subksp.setUp()
        except Exception:
            pass

    return len(subksps)


def apply_structured_pc(
    ksp,
    cfg: CaseConfig,
    layout,
    A,
    P,
    *,
    pc_type_override: Optional[str] = None,
) -> Dict[str, Any]:
    from parallel.mpi_bootstrap import bootstrap_mpi_before_petsc

    bootstrap_mpi_before_petsc()
    from petsc4py import PETSc

    diag: Dict[str, Any] = {"enabled": False}

    linear_cfg = getattr(getattr(cfg, "solver", None), "linear", None)
    petsc_cfg = getattr(cfg, "petsc", None)
    default_ksp_rtol = float(getattr(petsc_cfg, "rtol", 1.0e-8)) if petsc_cfg is not None else 1.0e-8
    default_ksp_atol = float(getattr(petsc_cfg, "atol", 1.0e-12)) if petsc_cfg is not None else 1.0e-12
    default_ksp_max_it = int(getattr(petsc_cfg, "max_it", 200)) if petsc_cfg is not None else 200

    def _as_float(value, default):
        if value is None:
            return default
        try:
            return float(value)
        except Exception:
            return default

    def _as_int(value, default):
        if value is None:
            return default
        try:
            return int(value)
        except Exception:
            return default

    def _resolve_sub_tols(ksp_type, *, base_rtol, base_atol, base_max_it, cfg_rtol, cfg_atol, cfg_max_it):
        rtol = _as_float(cfg_rtol, base_rtol)
        atol = _as_float(cfg_atol, base_atol)
        max_it = _as_int(cfg_max_it, base_max_it)
        if str(ksp_type).lower() == "preonly":
            if cfg_rtol is None:
                rtol = 0.0
            if cfg_atol is None:
                atol = 0.0
            if cfg_max_it is None:
                max_it = 1
        return rtol, atol, max_it
    pc_type = pc_type_override if pc_type_override is not None else _cfg_get(linear_cfg, "pc_type", None)
    if pc_type is None:
        return diag
    pc_type = _normalize_pc_type(pc_type)
    if pc_type is None:
        return diag
    diag["global"] = {"pc_type": pc_type}

    if pc_type != "fieldsplit":
        pc = ksp.getPC()
        try:
            pc.setType(pc_type)
        except Exception:
            logger.warning("Unknown pc_type='%s', falling back to jacobi", pc_type)
            pc.setType("jacobi")

        if pc_type == "asm":
            asm_overlap = _cfg_get(linear_cfg, "asm_overlap", None)
            if asm_overlap is not None:
                try:
                    asm_overlap = int(asm_overlap)
                    pc.setASMOverlap(asm_overlap)
                    diag["global"]["asm_overlap"] = asm_overlap
                except Exception:
                    pass
        return diag
    if layout is None:
        raise ValueError("fieldsplit pc requires a layout to build split IS.")

    pc = ksp.getPC()
    pc.setType("fieldsplit")
    diag["enabled"] = True
    diag["global"] = {"pc_type": "fieldsplit"}
    diag["ksp_prefix"] = str(ksp.getOptionsPrefix() or "")

    fs_cfg = _cfg_get(linear_cfg, "fieldsplit", None)
    fs_type = str(_cfg_get(fs_cfg, "type", "additive")).lower()
    split_mode = _cfg_get(fs_cfg, "split_mode", None)
    if split_mode is None:
        split_mode = _cfg_get(fs_cfg, "scheme", "by_layout")
    split_mode = str(split_mode).lower()
    split_plan = "bulk_iface" if fs_type == "schur" else split_mode
    if split_plan not in ("by_layout", "bulk_iface"):
        raise ValueError(f"Unsupported fieldsplit split_mode '{split_plan}'")

    r0, r1 = A.getOwnershipRange()
    splits = layout.build_is_petsc(comm=A.getComm(), ownership_range=(r0, r1), plan=split_plan)
    diag["splits"] = {k: int(v.getSize()) for k, v in splits.items()}
    diag["split_names"] = list(splits.keys())
    diag["fieldsplit"] = {"type": fs_type, "plan": split_plan, "splits": {}}

    pairs = [(str(name), iset) for name, iset in splits.items()]
    try:
        pc.setFieldSplitIS(*pairs)
    except TypeError:
        for name, iset in pairs:
            try:
                pc.setFieldSplitIS(name, iset)
            except TypeError:
                try:
                    pc.setFieldSplitIS(iset, name)
                except TypeError:
                    pc.setFieldSplitIS((name, iset))

    try:
        pc.setFieldSplitType(fs_type)
    except Exception:
        type_map = {
            "additive": PETSc.PC.CompositeType.ADDITIVE,
            "multiplicative": PETSc.PC.CompositeType.MULTIPLICATIVE,
            "schur": PETSc.PC.CompositeType.SCHUR,
        }
        if fs_type in type_map:
            pc.setFieldSplitType(type_map[fs_type])
        else:
            raise

    diag["fieldsplit_type"] = fs_type

    if fs_type == "schur":
        schur_fact = str(_cfg_get(fs_cfg, "schur_fact_type", "lower")).lower()
        try:
            pc.setFieldSplitSchurFactType(schur_fact)
        except Exception:
            schur_map = {
                "full": PETSc.PC.SchurFactType.FULL,
                "diag": PETSc.PC.SchurFactType.DIAG,
                "lower": PETSc.PC.SchurFactType.LOWER,
                "upper": PETSc.PC.SchurFactType.UPPER,
            }
            if schur_fact in schur_map:
                pc.setFieldSplitSchurFactType(schur_map[schur_fact])
            else:
                raise
        diag["schur_fact_type"] = schur_fact

    if fs_type == "schur":
        bulk_ksp_type = str(_cfg_get(fs_cfg, "bulk_ksp_type", "gmres")).lower()
        bulk_ksp_rtol = _as_float(_cfg_get(fs_cfg, "bulk_ksp_rtol", None), default_ksp_rtol)
        bulk_ksp_atol = _as_float(_cfg_get(fs_cfg, "bulk_ksp_atol", None), default_ksp_atol)
        bulk_ksp_max_it = _as_int(_cfg_get(fs_cfg, "bulk_ksp_max_it", None), default_ksp_max_it)
        bulk_pc_type = _normalize_pc_type(_cfg_get(fs_cfg, "bulk_pc_type", "ilu"))
        bulk_pc_asm_overlap = _cfg_get(fs_cfg, "bulk_pc_asm_overlap", None)
        bulk_sub_ksp_type = str(_cfg_get(fs_cfg, "bulk_sub_ksp_type", "preonly")).lower()
        bulk_sub_pc_type = _normalize_pc_type(_cfg_get(fs_cfg, "bulk_sub_pc_type", "ilu"))
        bulk_pc_asm_sub_ksp_type = _cfg_get(fs_cfg, "bulk_pc_asm_sub_ksp_type", None)
        bulk_pc_asm_sub_pc_type = _cfg_get(fs_cfg, "bulk_pc_asm_sub_pc_type", None)
        bulk_pc_asm_sub_ksp_rtol = _cfg_get(fs_cfg, "bulk_pc_asm_sub_ksp_rtol", None)
        bulk_pc_asm_sub_ksp_atol = _cfg_get(fs_cfg, "bulk_pc_asm_sub_ksp_atol", None)
        bulk_pc_asm_sub_ksp_max_it = _cfg_get(fs_cfg, "bulk_pc_asm_sub_ksp_max_it", None)
        bulk_pc_bjacobi_sub_ksp_type = _cfg_get(fs_cfg, "bulk_pc_bjacobi_sub_ksp_type", None)
        bulk_pc_bjacobi_sub_pc_type = _cfg_get(fs_cfg, "bulk_pc_bjacobi_sub_pc_type", None)
        bulk_pc_bjacobi_sub_ksp_rtol = _cfg_get(fs_cfg, "bulk_pc_bjacobi_sub_ksp_rtol", None)
        bulk_pc_bjacobi_sub_ksp_atol = _cfg_get(fs_cfg, "bulk_pc_bjacobi_sub_ksp_atol", None)
        bulk_pc_bjacobi_sub_ksp_max_it = _cfg_get(fs_cfg, "bulk_pc_bjacobi_sub_ksp_max_it", None)
        bulk_subdomain_ksp_type = bulk_sub_ksp_type
        bulk_subdomain_pc_type = bulk_sub_pc_type
        if bulk_pc_type == "asm":
            if bulk_pc_asm_sub_ksp_type:
                bulk_subdomain_ksp_type = str(bulk_pc_asm_sub_ksp_type).lower()
            if bulk_pc_asm_sub_pc_type:
                bulk_subdomain_pc_type = _normalize_pc_type(bulk_pc_asm_sub_pc_type)
        elif bulk_pc_type == "bjacobi":
            if bulk_pc_bjacobi_sub_ksp_type:
                bulk_subdomain_ksp_type = str(bulk_pc_bjacobi_sub_ksp_type).lower()
            if bulk_pc_bjacobi_sub_pc_type:
                bulk_subdomain_pc_type = _normalize_pc_type(bulk_pc_bjacobi_sub_pc_type)
        bulk_subdomain_rtol, bulk_subdomain_atol, bulk_subdomain_max_it = _resolve_sub_tols(
            bulk_subdomain_ksp_type,
            base_rtol=bulk_ksp_rtol,
            base_atol=bulk_ksp_atol,
            base_max_it=bulk_ksp_max_it,
            cfg_rtol=(
                bulk_pc_asm_sub_ksp_rtol
                if bulk_pc_type == "asm"
                else bulk_pc_bjacobi_sub_ksp_rtol
                if bulk_pc_type == "bjacobi"
                else None
            ),
            cfg_atol=(
                bulk_pc_asm_sub_ksp_atol
                if bulk_pc_type == "asm"
                else bulk_pc_bjacobi_sub_ksp_atol
                if bulk_pc_type == "bjacobi"
                else None
            ),
            cfg_max_it=(
                bulk_pc_asm_sub_ksp_max_it
                if bulk_pc_type == "asm"
                else bulk_pc_bjacobi_sub_ksp_max_it
                if bulk_pc_type == "bjacobi"
                else None
            ),
        )
        iface_ksp_type = str(_cfg_get(fs_cfg, "iface_ksp_type", "preonly")).lower()
        iface_ksp_rtol = _as_float(_cfg_get(fs_cfg, "iface_ksp_rtol", None), default_ksp_rtol)
        iface_ksp_atol = _as_float(_cfg_get(fs_cfg, "iface_ksp_atol", None), default_ksp_atol)
        iface_ksp_max_it = _as_int(_cfg_get(fs_cfg, "iface_ksp_max_it", None), default_ksp_max_it)
        iface_pc_type = _normalize_pc_type(_cfg_get(fs_cfg, "iface_pc_type", "lu"))
        iface_pc_asm_overlap = _cfg_get(fs_cfg, "iface_pc_asm_overlap", None)
        iface_sub_ksp_type = str(_cfg_get(fs_cfg, "iface_sub_ksp_type", "preonly")).lower()
        iface_sub_pc_type = _normalize_pc_type(_cfg_get(fs_cfg, "iface_sub_pc_type", "ilu"))
        iface_pc_asm_sub_ksp_type = _cfg_get(fs_cfg, "iface_pc_asm_sub_ksp_type", None)
        iface_pc_asm_sub_pc_type = _cfg_get(fs_cfg, "iface_pc_asm_sub_pc_type", None)
        iface_pc_asm_sub_ksp_rtol = _cfg_get(fs_cfg, "iface_pc_asm_sub_ksp_rtol", None)
        iface_pc_asm_sub_ksp_atol = _cfg_get(fs_cfg, "iface_pc_asm_sub_ksp_atol", None)
        iface_pc_asm_sub_ksp_max_it = _cfg_get(fs_cfg, "iface_pc_asm_sub_ksp_max_it", None)
        iface_pc_bjacobi_sub_ksp_type = _cfg_get(fs_cfg, "iface_pc_bjacobi_sub_ksp_type", None)
        iface_pc_bjacobi_sub_pc_type = _cfg_get(fs_cfg, "iface_pc_bjacobi_sub_pc_type", None)
        iface_pc_bjacobi_sub_ksp_rtol = _cfg_get(fs_cfg, "iface_pc_bjacobi_sub_ksp_rtol", None)
        iface_pc_bjacobi_sub_ksp_atol = _cfg_get(fs_cfg, "iface_pc_bjacobi_sub_ksp_atol", None)
        iface_pc_bjacobi_sub_ksp_max_it = _cfg_get(fs_cfg, "iface_pc_bjacobi_sub_ksp_max_it", None)
        iface_subdomain_ksp_type = iface_sub_ksp_type
        iface_subdomain_pc_type = iface_sub_pc_type
        if iface_pc_type == "asm":
            if iface_pc_asm_sub_ksp_type:
                iface_subdomain_ksp_type = str(iface_pc_asm_sub_ksp_type).lower()
            if iface_pc_asm_sub_pc_type:
                iface_subdomain_pc_type = _normalize_pc_type(iface_pc_asm_sub_pc_type)
        elif iface_pc_type == "bjacobi":
            if iface_pc_bjacobi_sub_ksp_type:
                iface_subdomain_ksp_type = str(iface_pc_bjacobi_sub_ksp_type).lower()
            if iface_pc_bjacobi_sub_pc_type:
                iface_subdomain_pc_type = _normalize_pc_type(iface_pc_bjacobi_sub_pc_type)
        iface_subdomain_rtol, iface_subdomain_atol, iface_subdomain_max_it = _resolve_sub_tols(
            iface_subdomain_ksp_type,
            base_rtol=iface_ksp_rtol,
            base_atol=iface_ksp_atol,
            base_max_it=iface_ksp_max_it,
            cfg_rtol=(
                iface_pc_asm_sub_ksp_rtol
                if iface_pc_type == "asm"
                else iface_pc_bjacobi_sub_ksp_rtol
                if iface_pc_type == "bjacobi"
                else None
            ),
            cfg_atol=(
                iface_pc_asm_sub_ksp_atol
                if iface_pc_type == "asm"
                else iface_pc_bjacobi_sub_ksp_atol
                if iface_pc_type == "bjacobi"
                else None
            ),
            cfg_max_it=(
                iface_pc_asm_sub_ksp_max_it
                if iface_pc_type == "asm"
                else iface_pc_bjacobi_sub_ksp_max_it
                if iface_pc_type == "bjacobi"
                else None
            ),
        )
        diag["sub_defaults"] = {
            "default": {
                "ksp_type": bulk_ksp_type,
                "ksp_rtol": bulk_ksp_rtol,
                "ksp_atol": bulk_ksp_atol,
                "ksp_max_it": bulk_ksp_max_it,
                "pc_type": bulk_pc_type,
                "pc_asm_overlap": bulk_pc_asm_overlap,
                "subdomain_ksp_type": bulk_subdomain_ksp_type,
                "subdomain_pc_type": bulk_subdomain_pc_type,
                "subdomain_ksp_rtol": bulk_subdomain_rtol,
                "subdomain_ksp_atol": bulk_subdomain_atol,
                "subdomain_ksp_max_it": bulk_subdomain_max_it,
            },
            "by_name": {
                "bulk": {
                    "ksp_type": bulk_ksp_type,
                    "ksp_rtol": bulk_ksp_rtol,
                    "ksp_atol": bulk_ksp_atol,
                    "ksp_max_it": bulk_ksp_max_it,
                    "pc_type": bulk_pc_type,
                    "pc_asm_overlap": bulk_pc_asm_overlap,
                    "subdomain_ksp_type": bulk_subdomain_ksp_type,
                    "subdomain_pc_type": bulk_subdomain_pc_type,
                    "subdomain_ksp_rtol": bulk_subdomain_rtol,
                    "subdomain_ksp_atol": bulk_subdomain_atol,
                    "subdomain_ksp_max_it": bulk_subdomain_max_it,
                },
                "iface": {
                    "ksp_type": iface_ksp_type,
                    "ksp_rtol": iface_ksp_rtol,
                    "ksp_atol": iface_ksp_atol,
                    "ksp_max_it": iface_ksp_max_it,
                    "pc_type": iface_pc_type,
                    "pc_asm_overlap": iface_pc_asm_overlap,
                    "subdomain_ksp_type": iface_subdomain_ksp_type,
                    "subdomain_pc_type": iface_subdomain_pc_type,
                    "subdomain_ksp_rtol": iface_subdomain_rtol,
                    "subdomain_ksp_atol": iface_subdomain_atol,
                    "subdomain_ksp_max_it": iface_subdomain_max_it,
                },
            },
        }
    else:
        sub_ksp_type = str(_cfg_get(fs_cfg, "sub_ksp_type", "preonly")).lower()
        sub_ksp_rtol = _as_float(_cfg_get(fs_cfg, "sub_ksp_rtol", None), default_ksp_rtol)
        sub_ksp_atol = _as_float(_cfg_get(fs_cfg, "sub_ksp_atol", None), default_ksp_atol)
        sub_ksp_max_it = _as_int(_cfg_get(fs_cfg, "sub_ksp_max_it", None), default_ksp_max_it)
        sub_pc_type = _normalize_pc_type(_cfg_get(fs_cfg, "sub_pc_type", "ilu"))
        sub_pc_asm_overlap = _cfg_get(fs_cfg, "sub_pc_asm_overlap", None)
        sub_pc_asm_sub_ksp_type = _cfg_get(fs_cfg, "sub_pc_asm_sub_ksp_type", None)
        sub_pc_asm_sub_pc_type = _cfg_get(fs_cfg, "sub_pc_asm_sub_pc_type", None)
        sub_pc_asm_sub_ksp_rtol = _cfg_get(fs_cfg, "sub_pc_asm_sub_ksp_rtol", None)
        sub_pc_asm_sub_ksp_atol = _cfg_get(fs_cfg, "sub_pc_asm_sub_ksp_atol", None)
        sub_pc_asm_sub_ksp_max_it = _cfg_get(fs_cfg, "sub_pc_asm_sub_ksp_max_it", None)
        sub_pc_bjacobi_sub_ksp_type = _cfg_get(fs_cfg, "sub_pc_bjacobi_sub_ksp_type", None)
        sub_pc_bjacobi_sub_pc_type = _cfg_get(fs_cfg, "sub_pc_bjacobi_sub_pc_type", None)
        sub_pc_bjacobi_sub_ksp_rtol = _cfg_get(fs_cfg, "sub_pc_bjacobi_sub_ksp_rtol", None)
        sub_pc_bjacobi_sub_ksp_atol = _cfg_get(fs_cfg, "sub_pc_bjacobi_sub_ksp_atol", None)
        sub_pc_bjacobi_sub_ksp_max_it = _cfg_get(fs_cfg, "sub_pc_bjacobi_sub_ksp_max_it", None)
        subdomain_ksp_type = sub_ksp_type
        subdomain_pc_type = sub_pc_type
        if sub_pc_type == "asm":
            if sub_pc_asm_sub_ksp_type:
                subdomain_ksp_type = str(sub_pc_asm_sub_ksp_type).lower()
            if sub_pc_asm_sub_pc_type:
                subdomain_pc_type = _normalize_pc_type(sub_pc_asm_sub_pc_type)
        elif sub_pc_type == "bjacobi":
            if sub_pc_bjacobi_sub_ksp_type:
                subdomain_ksp_type = str(sub_pc_bjacobi_sub_ksp_type).lower()
            if sub_pc_bjacobi_sub_pc_type:
                subdomain_pc_type = _normalize_pc_type(sub_pc_bjacobi_sub_pc_type)
        subdomain_rtol, subdomain_atol, subdomain_max_it = _resolve_sub_tols(
            subdomain_ksp_type,
            base_rtol=sub_ksp_rtol,
            base_atol=sub_ksp_atol,
            base_max_it=sub_ksp_max_it,
            cfg_rtol=(
                sub_pc_asm_sub_ksp_rtol
                if sub_pc_type == "asm"
                else sub_pc_bjacobi_sub_ksp_rtol
                if sub_pc_type == "bjacobi"
                else None
            ),
            cfg_atol=(
                sub_pc_asm_sub_ksp_atol
                if sub_pc_type == "asm"
                else sub_pc_bjacobi_sub_ksp_atol
                if sub_pc_type == "bjacobi"
                else None
            ),
            cfg_max_it=(
                sub_pc_asm_sub_ksp_max_it
                if sub_pc_type == "asm"
                else sub_pc_bjacobi_sub_ksp_max_it
                if sub_pc_type == "bjacobi"
                else None
            ),
        )
        diag["sub_defaults"] = {
            "default": {
                "ksp_type": sub_ksp_type,
                "ksp_rtol": sub_ksp_rtol,
                "ksp_atol": sub_ksp_atol,
                "ksp_max_it": sub_ksp_max_it,
                "pc_type": sub_pc_type,
                "pc_asm_overlap": sub_pc_asm_overlap,
                "subdomain_ksp_type": subdomain_ksp_type,
                "subdomain_pc_type": subdomain_pc_type,
                "subdomain_ksp_rtol": subdomain_rtol,
                "subdomain_ksp_atol": subdomain_atol,
                "subdomain_ksp_max_it": subdomain_max_it,
            }
        }

    return diag


def apply_fieldsplit_subksp_defaults(ksp, diag: Mapping[str, Any]) -> None:
    if not diag.get("enabled", False):
        return
    pc = ksp.getPC()
    try:
        pc.setUp()
    except Exception:
        pass
    try:
        sub = pc.getFieldSplitSubKSP()
    except Exception:
        return

    if isinstance(sub, tuple) and len(sub) == 2:
        names, subksps = sub
    else:
        subksps = sub
        names = diag.get("split_names", None)

    if not names:
        names = list(diag.get("splits", {}).keys())

    defaults = diag.get("sub_defaults", {})
    by_name = defaults.get("by_name", {})
    default_cfg = defaults.get("default", None)
    fs_diag = diag.setdefault("fieldsplit", {})
    fs_splits = fs_diag.setdefault("splits", {})

    for idx, subksp in enumerate(subksps):
        name = names[idx] if idx < len(names) else f"split_{idx}"
        if isinstance(name, bytes):
            name = name.decode()
        name = str(name)
        cfg = by_name.get(name, default_cfg)
        if not cfg:
            continue
        ksp_type = cfg.get("ksp_type", None)
        pc_type = _normalize_pc_type(cfg.get("pc_type", None))
        ksp_rtol = cfg.get("ksp_rtol", None)
        ksp_atol = cfg.get("ksp_atol", None)
        ksp_max_it = cfg.get("ksp_max_it", None)
        asm_overlap = cfg.get("pc_asm_overlap", None)
        subdomain_ksp_type = cfg.get("subdomain_ksp_type", None)
        subdomain_pc_type = _normalize_pc_type(cfg.get("subdomain_pc_type", None))
        subdomain_ksp_rtol = cfg.get("subdomain_ksp_rtol", None)
        subdomain_ksp_atol = cfg.get("subdomain_ksp_atol", None)
        subdomain_ksp_max_it = cfg.get("subdomain_ksp_max_it", None)
        n_sub = None
        if ksp_type:
            try:
                subksp.setType(str(ksp_type))
            except Exception:
                pass
        if ksp_rtol is not None or ksp_atol is not None or ksp_max_it is not None:
            try:
                subksp.setTolerances(rtol=ksp_rtol, atol=ksp_atol, max_it=ksp_max_it)
            except Exception:
                pass
        try:
            spc = subksp.getPC()
            if pc_type:
                spc.setType(str(pc_type))
            if pc_type in ("asm", "bjacobi"):
                n_sub = _configure_pc_asm_or_bjacobi_subksps(
                    spc,
                    sub_ksp_type=subdomain_ksp_type,
                    sub_pc_type=subdomain_pc_type,
                    sub_ksp_rtol=subdomain_ksp_rtol,
                    sub_ksp_atol=subdomain_ksp_atol,
                    sub_ksp_max_it=subdomain_ksp_max_it,
                    overlap=asm_overlap,
                )
        except Exception:
            pass
        try:
            subksp.setFromOptions()
        except Exception:
            pass
        try:
            subksp.setUp()
        except Exception:
            pass

        split_info = fs_splits.setdefault(name, {})
        if ksp_type:
            split_info["ksp_type"] = str(ksp_type)
        if pc_type:
            split_info["pc_type"] = str(pc_type)
        if ksp_rtol is not None:
            split_info["ksp_rtol"] = float(ksp_rtol)
        if ksp_atol is not None:
            split_info["ksp_atol"] = float(ksp_atol)
        if ksp_max_it is not None:
            split_info["ksp_max_it"] = int(ksp_max_it)
        if asm_overlap is not None and pc_type == "asm":
            split_info["asm_overlap"] = asm_overlap
        if pc_type in ("asm", "bjacobi") and subdomain_ksp_type:
            split_info["subdomain_ksp_type"] = str(subdomain_ksp_type)
        if pc_type in ("asm", "bjacobi") and subdomain_pc_type:
            split_info["subdomain_pc_type"] = str(subdomain_pc_type)
        if pc_type in ("asm", "bjacobi") and subdomain_ksp_rtol is not None:
            split_info["subdomain_ksp_rtol"] = float(subdomain_ksp_rtol)
        if pc_type in ("asm", "bjacobi") and subdomain_ksp_atol is not None:
            split_info["subdomain_ksp_atol"] = float(subdomain_ksp_atol)
        if pc_type in ("asm", "bjacobi") and subdomain_ksp_max_it is not None:
            split_info["subdomain_ksp_max_it"] = int(subdomain_ksp_max_it)
        if pc_type == "asm":
            split_info["asm_subdomains"] = int(n_sub) if n_sub is not None else 0
        if pc_type == "bjacobi":
            split_info["bjacobi_subdomains"] = int(n_sub) if n_sub is not None else 0

def solve_linear_system_petsc(
    A,
    b,
    cfg: CaseConfig,
    x0: Optional[np.ndarray] = None,
    method: str = "ksp",
    *,
    layout=None,
    P=None,
) -> LinearSolveResult:
    """Solve Ax=b using PETSc KSP; returns LinearSolveResult (mirrors SciPy backend)."""
    try:
        from parallel.mpi_bootstrap import bootstrap_mpi_before_petsc

        bootstrap_mpi_before_petsc()
        from petsc4py import PETSc
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("petsc4py is required for PETSc backend.") from exc

    if not isinstance(A, PETSc.Mat):
        raise TypeError(f"Expected PETSc.Mat for A, got {type(A)}")
    if not isinstance(b, PETSc.Vec):
        raise TypeError(f"Expected PETSc.Vec for b, got {type(b)}")

    m, n = A.getSize()
    if m != n:
        raise ValueError(f"A must be square, got size {(m, n)}")
    N = n
    if b.getSize() != N:
        raise ValueError(f"b size {b.getSize()} does not match A dimension {N}")

    if x0 is not None:
        x0 = np.asarray(x0, dtype=np.float64)
        if x0.shape != (N,):
            raise ValueError(f"x0 shape {x0.shape} does not match A dimension {N}")

    petsc_cfg = cfg.petsc
    linear_cfg = getattr(getattr(cfg, "solver", None), "linear", None)
    prefix = getattr(petsc_cfg, "options_prefix", "")
    if prefix is None:
        prefix = ""
    prefix = str(prefix)
    if prefix and not prefix.endswith("_"):
        prefix += "_"
    ksp_type = str(getattr(petsc_cfg, "ksp_type", "gmres"))
    pc_type = _cfg_get(linear_cfg, "pc_type", None)
    if pc_type is None:
        pc_type = str(getattr(petsc_cfg, "pc_type", "ilu"))
    else:
        pc_type = str(pc_type)
    pc_type = _normalize_pc_type(pc_type) or "ilu"
    if method in ("direct", "lu", "preonly"):
        ksp_type, pc_type = "preonly", "lu"
    elif method in ("ksp", "gmres", "fgmres", "lgmres"):
        pass
    rtol = float(getattr(petsc_cfg, "rtol", 1e-8))
    atol = float(getattr(petsc_cfg, "atol", 1e-12))
    max_it = int(getattr(petsc_cfg, "max_it", 200))
    restart = int(getattr(petsc_cfg, "restart", 30))
    monitor = bool(getattr(petsc_cfg, "monitor", False))

    logger.debug(
        "solve_linear_system_petsc: case=%s size=%s method=%s ksp=%s pc=%s rtol=%.3e atol=%.3e max_it=%d",
        getattr(cfg.case, "id", "unknown"),
        (N, N),
        method,
        ksp_type,
        pc_type,
        rtol,
        atol,
        max_it,
    )

    comm = A.getComm()
    ksp = PETSc.KSP().create(comm=comm)
    ksp.setOptionsPrefix(prefix)
    if P is None:
        P = A
    ksp.setOperators(A, P)

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

    ksp.setTolerances(rtol=rtol, atol=atol, max_it=max_it)

    try:
        ksp_type_eff = str(ksp.getType()).lower()
        if ksp_type_eff in ("gmres", "fgmres"):
            ksp.setGMRESRestart(restart)
        elif ksp_type_eff == "lgmres":
            if hasattr(ksp, "setLGMRESRestart"):
                ksp.setLGMRESRestart(restart)
    except Exception:
        logger.debug("Unable to set restart for ksp_type='%s'", ksp.getType())

    if monitor:
        def _monitor(ksp_obj, its, rnorm):
            logger.debug("[KSP] its=%d rnorm=%.6e", its, rnorm)
        ksp.setMonitor(_monitor)

    diag_pc = apply_structured_pc(ksp, cfg, layout, A, P, pc_type_override=_normalize_pc_type(pc_type))

    ksp.setFromOptions()
    ksp.setUp()

    apply_fieldsplit_subksp_defaults(ksp, diag_pc)

    x = A.createVecRight()
    x.set(0.0)
    if x0 is not None:
        ksp_type_eff = str(ksp.getType()).lower()
        if ksp_type_eff != "preonly":
            x0_arr = np.ascontiguousarray(x0, dtype=np.float64).copy()
            x.setArray(x0_arr)
            ksp.setInitialGuessNonzero(True)

    ksp.solve(b, x)

    reason = int(ksp.getConvergedReason())
    converged = reason > 0
    n_iter = int(ksp.getIterationNumber())

    res_norm = None
    try:
        res_norm = float(ksp.getResidualNorm())
    except Exception:
        res_norm = None
    if res_norm is None or not np.isfinite(res_norm) or res_norm <= 0.0:
        r = b.duplicate()
        A.mult(x, r)
        r.aypx(-1.0, b)
        res_norm = float(r.norm())

    b_norm = float(b.norm())
    rel = res_norm / (b_norm + 1e-30)

    if not converged:
        logger.warning(
            "PETSc KSP not converged: reason=%d residual=%.3e rel=%.3e ksp=%s pc=%s",
            reason,
            res_norm,
            rel,
            ksp.getType(),
            ksp.getPC().getType(),
        )

    return LinearSolveResult(
        x=np.asarray(x.getArray(), dtype=np.float64).copy(),
        converged=converged,
        n_iter=n_iter,
        residual_norm=res_norm,
        rel_residual=rel,
        method=f"{ksp.getType()}+{ksp.getPC().getType()}",
        message=None if converged else f"PETSc KSP diverged (reason={reason})",
        diag={"pc": diag_pc, "ksp_type": str(ksp.getType()), "pc_type": str(ksp.getPC().getType())},
    )
