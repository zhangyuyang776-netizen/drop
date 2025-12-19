from __future__ import annotations

"""
Quick utility to build and export the initial state without time marching.

Usage:
    python run_initial_preview.py cases/case_evap_single.yaml
"""

import logging
import sys
from pathlib import Path

import numpy as np

from core.grid import build_grid
from properties.compute_props import get_or_build_models
from physics.initial import build_initial_state_erfc
from driver.run_scipy_case import _load_case_config, _prepare_run_dir, _maybe_fill_gas_species

logger = logging.getLogger(__name__)


def _setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s")


def run_initial_preview(cfg_path: str) -> None:
    _setup_logging()
    cfg_path = str(cfg_path)
    logger.info("Loading case config from %s", cfg_path)

    cfg = _load_case_config(cfg_path)
    run_dir = _prepare_run_dir(cfg, cfg_path)
    logger.info("Run directory: %s", run_dir)
    try:
        logger.info("Initial Y_vap_if0 = %.3e", float(cfg.initial.Y_vap_if0))
    except Exception:
        logger.info("Initial Y_vap_if0 not available on cfg; using default in initializer.")

    gas_model, liq_model = get_or_build_models(cfg)
    _maybe_fill_gas_species(cfg, gas_model)

    grid = build_grid(cfg)
    logger.info("Grid built: Nl=%d, Ng=%d, Nc=%d", grid.Nl, grid.Ng, grid.Nc)

    state0 = build_initial_state_erfc(cfg, grid, gas_model, liq_model)
    logger.info(
        "Initial state: Ts=%.3f K, Rd=%.3e m, Tg[min,max]=[%.3f, %.3f]",
        float(state0.Ts),
        float(state0.Rd),
        float(np.min(state0.Tg)),
        float(np.max(state0.Tg)),
    )

    out_dir = Path(run_dir) / "initial_preview"
    out_dir.mkdir(parents=True, exist_ok=True)

    np.savetxt(out_dir / "r_c.csv", grid.r_c, delimiter=",")
    np.savetxt(out_dir / "r_f.csv", grid.r_f, delimiter=",")
    np.savetxt(out_dir / "Tg0.csv", state0.Tg, delimiter=",")
    np.savetxt(out_dir / "Tl0.csv", state0.Tl, delimiter=",")
    np.savetxt(out_dir / "Ts0.csv", np.array([state0.Ts]), delimiter=",")

    gas_names = list(getattr(cfg.species, "gas_species_full", []) or cfg.species.gas_species)
    header = ",".join(gas_names)
    np.savetxt(out_dir / "Yg0.csv", state0.Yg.T, delimiter=",", header=header, comments="")

    if state0.Yl is not None:
        np.savetxt(out_dir / "Yl0.csv", state0.Yl.T, delimiter=",")

    logger.info("Initial state exported to %s", out_dir)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python run_initial_preview.py path/to/case.yaml")
        sys.exit(1)
    run_initial_preview(sys.argv[1])
