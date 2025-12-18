"""
Driver to run a single SciPy-based case (Step 13.1 shell).

Responsibilities:
- Load CaseConfig from YAML.
- Build grid/layout/initial state/initial properties.
- Advance in time by repeatedly calling advance_one_step_scipy.
- Log per-step summaries; stop early on failure.
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import shutil
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np
import yaml

from core.grid import build_grid
from core.layout import build_layout
from core.types import (
    CaseChecks,
    CaseConfig,
    CaseConventions,
    CaseCoolProp,
    CaseDiscretization,
    CaseEquilibrium,
    CaseGeometry,
    CaseIO,
    CaseIOFields,
    CaseInitial,
    CaseInterface,
    CaseMesh,
    CaseMeta,
    CasePaths,
    CasePETSc,
    CasePhysics,
    CaseSpecies,
    CaseTime,
    Grid1D,
    State,
)
from properties.compute_props import compute_props, get_or_build_models
from properties.gas import GasPropertiesModel
from properties.liquid import LiquidPropertiesModel
from solvers.timestepper import StepResult, advance_one_step_scipy

logger = logging.getLogger(__name__)

_WRITERS_MODULE_NAME = "io_writers_cached_driver"


# -----------------------------------------------------------------------------
# YAML loader
# -----------------------------------------------------------------------------
def _resolve_path(base: Path, value: str | Path) -> Path:
    """Resolve a possibly relative path against base."""
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def _load_case_config(cfg_path: str) -> CaseConfig:
    """Load YAML file into CaseConfig with nested dataclasses."""
    cfg_file = Path(cfg_path).expanduser().resolve()
    raw = yaml.safe_load(cfg_file.read_text())
    base = cfg_file.parent

    case_cfg = CaseMeta(**raw["case"])

    paths_raw = raw["paths"]
    output_root = _resolve_path(base, paths_raw["output_root"])
    case_dir_raw = paths_raw.get("case_dir", output_root / case_cfg.id)
    case_dir = _resolve_path(base, case_dir_raw)
    mech_dir_raw = paths_raw.get("mechanism_dir", base)
    mechanism_dir = _resolve_path(base, mech_dir_raw)
    paths_cfg = CasePaths(
        output_root=output_root,
        case_dir=case_dir,
        mechanism_dir=mechanism_dir,
        gas_mech=paths_raw["gas_mech"],
    )

    conv_cfg = CaseConventions(**raw["conventions"])

    phys_raw = raw["physics"]
    eq_raw = phys_raw.get("interface", {}).get("equilibrium", {})
    cp_raw = eq_raw.get("coolprop", {})
    coolprop_cfg = CaseCoolProp(
        backend=cp_raw.get("backend", "HEOS"),
        fluids=list(cp_raw.get("fluids", [])),
    )
    eq_cfg = CaseEquilibrium(
        method=eq_raw.get("method", "raoult_psat"),
        psat_model=eq_raw.get("psat_model", "coolprop"),
        background_fill=eq_raw.get("background_fill", "farfield"),
        condensables_gas=list(eq_raw.get("condensables_gas", [])),
        coolprop=coolprop_cfg,
    )
    iface_raw = phys_raw.get("interface", {})
    interface_cfg = CaseInterface(
        type=iface_raw.get("type", "no_condensation"),
        bc_mode=iface_raw.get("bc_mode", "Ts_fixed"),
        Ts_fixed=float(iface_raw.get("Ts_fixed", 300.0)),
        equilibrium=eq_cfg,
    )
    physics_cfg = CasePhysics(
        model=phys_raw.get("model", "droplet_1d_spherical_noChem"),
        enable_liquid=phys_raw.get("enable_liquid", True),
        include_chemistry=phys_raw.get("include_chemistry", False),
        solve_Tg=phys_raw.get("solve_Tg", True),
        solve_Yg=phys_raw.get("solve_Yg", True),
        solve_Tl=phys_raw.get("solve_Tl", True),
        solve_Yl=phys_raw.get("solve_Yl", False),
        include_Ts=phys_raw.get("include_Ts", False),
        include_mpp=phys_raw.get("include_mpp", True),
        include_Rd=phys_raw.get("include_Rd", True),
        stefan_velocity=phys_raw.get("stefan_velocity", True),
        interface=interface_cfg,
    )

    species_raw = raw["species"]
    species_cfg = CaseSpecies(
        gas_balance_species=species_raw["gas_balance_species"],
        gas_mechanism_phase=species_raw.get("gas_mechanism_phase", "gas"),
        gas_species=list(species_raw.get("gas_species", [])),
        solve_gas_mode=species_raw.get("solve_gas_mode", "all_minus_closure"),
        solve_gas_species=list(species_raw.get("solve_gas_species", [])),
        liq_species=list(species_raw.get("liq_species", [])),
        liq_balance_species=species_raw["liq_balance_species"],
        liq2gas_map=species_raw.get("liq2gas_map", {}),
        mw_kg_per_mol=species_raw.get("mw_kg_per_mol", {}),
        molar_volume_cm3_per_mol=species_raw.get("molar_volume_cm3_per_mol", {}),
    )
    allowed_modes = {"all_minus_closure", "condensables_only", "explicit_list"}
    if species_cfg.solve_gas_mode not in allowed_modes:
        raise ValueError(f"solve_gas_mode must be one of {sorted(allowed_modes)}, got {species_cfg.solve_gas_mode}")
    if species_cfg.solve_gas_mode == "explicit_list" and not species_cfg.solve_gas_species:
        raise ValueError("solve_gas_species cannot be empty when solve_gas_mode='explicit_list'")

    mesh_raw = raw["geometry"]["mesh"]
    mesh_cfg = CaseMesh(
        liq_method=mesh_raw["liq_method"],
        liq_beta=mesh_raw["liq_beta"],
        liq_center_bias=mesh_raw["liq_center_bias"],
        gas_method=mesh_raw["gas_method"],
        gas_beta=mesh_raw["gas_beta"],
        gas_center_bias=mesh_raw["gas_center_bias"],
        enforce_interface_continuity=mesh_raw["enforce_interface_continuity"],
        continuity_tol=mesh_raw["continuity_tol"],
    )
    geom_raw = raw["geometry"]
    geom_cfg = CaseGeometry(
        a0=geom_raw["a0"],
        R_inf=geom_raw["R_inf"],
        N_liq=geom_raw["N_liq"],
        N_gas=geom_raw["N_gas"],
        mesh=mesh_cfg,
    )

    time_raw = raw["time"]
    time_cfg = CaseTime(
        t0=time_raw["t0"],
        dt=time_raw["dt"],
        t_end=time_raw["t_end"],
        max_steps=time_raw.get("max_steps", None),
    )

    disc_raw = raw["discretization"]
    disc_cfg = CaseDiscretization(
        time_scheme=disc_raw["time_scheme"],
        theta=disc_raw["theta"],
        mass_matrix=disc_raw["mass_matrix"],
    )

    init_raw = raw["initial"]
    init_cfg = CaseInitial(
        T_inf=init_raw["T_inf"],
        P_inf=init_raw["P_inf"],
        T_d0=init_raw["T_d0"],
        Yg=init_raw["Yg"],
        Yl=init_raw["Yl"],
        Y_seed=init_raw["Y_seed"],
    )

    petsc_raw = raw["petsc"]
    petsc_cfg = CasePETSc(
        options_prefix=petsc_raw["options_prefix"],
        ksp_type=petsc_raw["ksp_type"],
        pc_type=petsc_raw["pc_type"],
        rtol=petsc_raw["rtol"],
        atol=petsc_raw["atol"],
        max_it=petsc_raw["max_it"],
        restart=petsc_raw["restart"],
        monitor=petsc_raw["monitor"],
    )

    io_raw = raw["io"]
    fields_raw = io_raw.get("fields", {})
    io_fields_cfg = CaseIOFields(
        scalars=list(fields_raw.get("scalars", [])),
        gas=list(fields_raw.get("gas", [])),
        liquid=list(fields_raw.get("liquid", [])),
    )
    io_cfg = CaseIO(
        write_every=io_raw["write_every"],
        formats=list(io_raw.get("formats", [])),
        save_grid=io_raw.get("save_grid", False),
        fields=io_fields_cfg,
    )

    checks_raw = raw["checks"]
    checks_cfg = CaseChecks(
        enforce_sumY=checks_raw["enforce_sumY"],
        sumY_tol=checks_raw["sumY_tol"],
        clamp_negative_Y=checks_raw["clamp_negative_Y"],
        min_Y=checks_raw["min_Y"],
        enforce_T_bounds=checks_raw["enforce_T_bounds"],
        T_min=checks_raw["T_min"],
        T_max=checks_raw["T_max"],
        enforce_unique_index=checks_raw.get("enforce_unique_index", True),
        enforce_grid_state_props_split=checks_raw.get("enforce_grid_state_props_split", True),
        enforce_assembly_purity=checks_raw.get("enforce_assembly_purity", True),
    )

    return CaseConfig(
        case=case_cfg,
        paths=paths_cfg,
        conventions=conv_cfg,
        physics=physics_cfg,
        species=species_cfg,
        geometry=geom_cfg,
        time=time_cfg,
        discretization=disc_cfg,
        initial=init_cfg,
        petsc=petsc_cfg,
        io=io_cfg,
        checks=checks_cfg,
    )


# -----------------------------------------------------------------------------
# Builders
# -----------------------------------------------------------------------------
def _maybe_fill_gas_species(cfg: CaseConfig, gas_model: GasPropertiesModel) -> None:
    """Fill cfg.species.gas_species from mechanism if not provided or mismatched."""
    mech_names = list(gas_model.gas.species_names)
    if not getattr(cfg.species, "gas_species", []):
        cfg.species.gas_species = mech_names
        return
    if len(cfg.species.gas_species) != len(mech_names):
        logger.warning(
            "cfg.species.gas_species length %s differs from mechanism %s; using mechanism order.",
            len(cfg.species.gas_species),
            len(mech_names),
        )
        cfg.species.gas_species = mech_names


def _build_mass_fractions(
    names: Sequence[str],
    values: Mapping[str, float],
    closure_name: str,
    seed: float,
    n_cells: int,
) -> np.ndarray:
    """Build full mass-fraction array with closure species filled as complement."""
    Ns = len(names)
    Y = np.full((Ns, n_cells), float(seed), dtype=np.float64)
    for i, name in enumerate(names):
        if name in values:
            Y[i, :] = float(values[name])

    if closure_name in names:
        idx = names.index(closure_name)
        others = np.sum(Y, axis=0) - Y[idx, :]
        Y[idx, :] = np.maximum(1.0 - others, 0.0)

    sums = np.sum(Y, axis=0)
    for j in range(n_cells):
        s = float(sums[j])
        if s > 0.0:
            Y[:, j] /= s
        elif Ns > 0:
            Y[0, j] = 1.0
    return Y


def _build_initial_state(
    cfg: CaseConfig,
    grid: Grid1D,
    gas_model: GasPropertiesModel,
    liq_model: Optional[LiquidPropertiesModel],
) -> State:
    """Construct initial State from CaseConfig."""
    T_inf = float(cfg.initial.T_inf)
    T_d0 = float(cfg.initial.T_d0)
    if cfg.physics.interface.bc_mode == "Ts_fixed":
        Ts0 = float(cfg.physics.interface.Ts_fixed)
    else:
        Ts0 = float(T_d0)
    seed = float(cfg.initial.Y_seed)

    gas_names = list(cfg.species.gas_species) or list(gas_model.gas.species_names)
    liq_names = list(cfg.species.liq_species)

    Tg0 = np.full(grid.Ng, T_inf, dtype=np.float64)
    Tl0 = np.full(grid.Nl, T_d0, dtype=np.float64)

    Yg0 = _build_mass_fractions(
        gas_names,
        cfg.initial.Yg,
        closure_name=cfg.species.gas_balance_species,
        seed=seed,
        n_cells=grid.Ng,
    )
    Yl0 = _build_mass_fractions(
        liq_names,
        cfg.initial.Yl,
        closure_name=cfg.species.liq_balance_species,
        seed=seed,
        n_cells=grid.Nl,
    )

    return State(
        Tg=Tg0,
        Yg=Yg0,
        Tl=Tl0,
        Yl=Yl0,
        Ts=Ts0,
        mpp=0.0,
        Rd=float(cfg.geometry.a0),
    )


def _prepare_run_dir(cfg: CaseConfig, cfg_path: str) -> Path:
    """Create per-run output directory and copy cfg yaml into it."""
    out_root = Path(cfg.paths.output_root)
    case_id = getattr(cfg.case, "id", "case")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = out_root / case_id / stamp
    run_dir.mkdir(parents=True, exist_ok=True)

    cfg.paths.case_dir = run_dir

    try:
        shutil.copy2(cfg_path, run_dir / "config.yaml")
    except Exception as exc:  # pragma: no cover - best-effort copy
        logger.warning("Failed to copy cfg to run dir: %s", exc)
    return run_dir


# -----------------------------------------------------------------------------
# Logging helpers
# -----------------------------------------------------------------------------
def _log_step(res: StepResult, step_id: int) -> None:
    """Emit one-line step summary."""
    d = res.diag
    logger.info(
        "step=%d t=[%.6e -> %.6e] dt=%.3e Ts=%.3f Rd=%.3e mpp=%.3e Tg[min,max]=[%.3f, %.3f] lin_conv=%s rel=%.3e",
        step_id,
        d.t_old,
        d.t_new,
        d.dt,
        d.Ts,
        d.Rd,
        d.mpp,
        d.Tg_min,
        d.Tg_max,
        d.linear_converged,
        d.linear_rel_residual,
    )


def _sanity_check_state(state: State) -> Optional[str]:
    """Lightweight driver-level sanity checks."""
    if not np.isfinite(state.Ts):
        return "Non-finite Ts"
    if not np.isfinite(state.Rd) or state.Rd <= 0.0:
        return "Rd is non-positive or non-finite"
    if not np.isfinite(state.mpp):
        return "Non-finite mpp"
    if np.any(~np.isfinite(state.Tg)):
        return "Non-finite Tg entries"
    if np.any(~np.isfinite(state.Tl)):
        return "Non-finite Tl entries"
    return None


# -----------------------------------------------------------------------------
# Optional spatial writer (dynamic import to avoid io module shadowing)
# -----------------------------------------------------------------------------
def _load_writers_module():
    """Dynamically import io.writers to avoid stdlib io shadowing."""
    module = sys.modules.get(_WRITERS_MODULE_NAME)
    if module is not None:
        return module
    path = Path(__file__).resolve().parent.parent / "io" / "writers.py"
    spec = importlib.util.spec_from_file_location(_WRITERS_MODULE_NAME, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load writers module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[_WRITERS_MODULE_NAME] = module
    spec.loader.exec_module(module)  # type: ignore[arg-type]
    return module


def _maybe_write_spatial(cfg: CaseConfig, grid: Grid1D, state: State, step_id: int) -> None:
    """Write spatial fields at configured frequency."""
    write_every = int(getattr(cfg.io, "write_every", 0) or 0)
    if write_every <= 0:
        return
    if step_id % write_every != 0:
        return
    try:
        module = _load_writers_module()
        write_fn = getattr(module, "write_step_spatial", None)
        if write_fn is None:
            return
        write_fn(cfg=cfg, grid=grid, state=state)
    except Exception as exc:  # pragma: no cover - best-effort output
        logger.warning("write_step_spatial failed at step %s: %s", step_id, exc)


# -----------------------------------------------------------------------------
# Main driver
# -----------------------------------------------------------------------------
def run_case(cfg_path: str, *, max_steps: Optional[int] = None, log_level: int | str = logging.INFO) -> int:
    """Run one SciPy case. Return 0 on success, non-zero on early failure."""
    level = log_level
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    if not isinstance(level, int):
        level = logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    cfg_path = str(cfg_path)
    try:
        cfg = _load_case_config(cfg_path)

        if cfg.time.dt <= 0.0:
            logger.error("cfg.time.dt must be positive (got %s)", cfg.time.dt)
            return 2
        if cfg.time.t_end <= cfg.time.t0:
            logger.error("cfg.time.t_end must exceed t0 (t0=%s, t_end=%s)", cfg.time.t0, cfg.time.t_end)
            return 2

        run_dir = _prepare_run_dir(cfg, cfg_path)
        logger.info("Run directory: %s", run_dir)

        gas_model, liq_model = get_or_build_models(cfg)
        _maybe_fill_gas_species(cfg, gas_model)

        grid = build_grid(cfg)
        layout = build_layout(cfg, grid)

        state = _build_initial_state(cfg, grid, gas_model, liq_model)
        props, _ = compute_props(cfg, grid, state)

        t = float(cfg.time.t0)
        step_id = 0
        effective_max_steps = max_steps if max_steps is not None else cfg.time.max_steps

        _maybe_write_spatial(cfg, grid, state, step_id)

        while t < cfg.time.t_end:
            step_id += 1
            if effective_max_steps is not None and step_id > effective_max_steps:
                logger.error(
                    "Reached max_steps=%s before t_end=%.3e (t=%.3e); aborting.",
                    effective_max_steps,
                    cfg.time.t_end,
                    t,
                )
                return 2

            res = advance_one_step_scipy(cfg, grid, layout, state, props, t)
            _log_step(res, step_id)

            if not res.success:
                logger.error("Step %s failed: %s", step_id, res.message)
                if res.diag.extra:
                    logger.error("Diagnostics extra: %s", res.diag.extra)
                return 2
            if not res.diag.linear_converged:
                logger.error("Linear solver not converged at step %s (diag)", step_id)
                return 2

            sanity_msg = _sanity_check_state(res.state_new)
            if sanity_msg is not None:
                logger.error("Sanity check failed at step %s: %s", step_id, sanity_msg)
                return 2

            state = res.state_new
            props = res.props_new
            t = res.diag.t_new

            _maybe_write_spatial(cfg, grid, state, step_id)

        logger.info("Completed run: t=%.6e reached t_end=%.6e after %d steps.", t, cfg.time.t_end, step_id)
        return 0
    except Exception as exc:
        tb = traceback.format_exc()
        logger.error("Unhandled exception:\n%s", tb)
        # Also print to stderr to ensure visibility even if logging is misconfigured
        print(f"\n{'='*80}", file=sys.stderr)
        print("UNHANDLED EXCEPTION IN run_case:", file=sys.stderr)
        print(f"{'='*80}", file=sys.stderr)
        print(f"Type: {type(exc).__name__}", file=sys.stderr)
        print(f"Message: {exc}", file=sys.stderr)
        print(f"\nFull traceback:", file=sys.stderr)
        print(tb, file=sys.stderr)
        print(f"{'='*80}\n", file=sys.stderr)
        return 99


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a SciPy timestepper case.")
    parser.add_argument("cfg_path", help="Path to case YAML file.")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional cap on number of steps to prevent infinite loops.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (e.g., INFO, DEBUG).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    lvl = getattr(logging, str(args.log_level).upper(), None)
    if not isinstance(lvl, int):
        lvl = logging.INFO
    return run_case(args.cfg_path, max_steps=args.max_steps, log_level=lvl)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
