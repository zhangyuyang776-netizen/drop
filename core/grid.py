"""
Grid construction from CaseConfig geometry settings.

Builds a static 1D spherical mesh (liquid + gas) using tanh stretching on each
segment, with interface-concentrated refinement.
"""

from __future__ import annotations

import numpy as np

from .types import CaseConfig, Grid1D, FloatArray


def _build_segment_tanh(L: float, N: int, *, beta: float = 3.0, center_bias: float = 0.0) -> FloatArray:
    """
    Generate positive cell widths on [0, L] using a tanh mapping.

    center_bias > 0 -> cluster toward the right end; center_bias < 0 -> left end.
    Returns an array of length N whose sum is L.
    """
    N = int(N)
    L = float(L)
    if N <= 0:
        return np.array([], dtype=np.float64)

    s = np.linspace(-1.0, 1.0, N + 1)
    y = np.tanh(float(beta) * (s + float(center_bias)))
    xi = (y - y.min()) / (y.max() - y.min())
    nodes = L * xi
    widths = np.diff(nodes)

    if np.any(~np.isfinite(widths)) or np.any(widths <= 0.0):
        raise ValueError("tanh grid produced non-positive or non-finite widths.")

    widths *= (L / widths.sum())
    return widths


def _segment_widths_to_faces(r0: float, widths: FloatArray) -> FloatArray:
    """Construct face coordinates from a starting radius and cell widths."""
    r0 = float(r0)
    widths = np.asarray(widths, dtype=np.float64)
    faces = np.empty(widths.size + 1, dtype=np.float64)
    faces[0] = r0
    np.cumsum(widths, out=faces[1:])
    faces[1:] += r0
    return faces


def build_grid(cfg: CaseConfig) -> Grid1D:
    """
    Build a static 1D spherical grid from CaseConfig.geometry.

    Segments:
      - Liquid: [0, a0] with N_liq cells (tanh, biased to interface)
      - Gas:    [a0, R_inf] with N_gas cells (tanh, biased to interface)
    Interface face index = N_liq.
    """
    gcfg = cfg.geometry
    mesh = gcfg.mesh

    Nl = int(gcfg.N_liq)
    Ng = int(gcfg.N_gas)
    a0 = float(gcfg.a0)
    Rinf = float(gcfg.R_inf)

    if mesh.liq_method != "tanh" or mesh.gas_method != "tanh":
        raise ValueError(f"Only 'tanh' mesh methods are supported (liq={mesh.liq_method}, gas={mesh.gas_method}).")
    if not (0.0 < a0 < Rinf):
        raise ValueError(f"Require 0 < a0 < R_inf, got a0={a0}, R_inf={Rinf}.")
    if Nl < 0 or Ng < 0 or (Nl + Ng) <= 0:
        raise ValueError(f"Invalid cell counts: N_liq={Nl}, N_gas={Ng}.")

    beta_liq = float(mesh.liq_beta)
    beta_gas = float(mesh.gas_beta)
    if beta_liq <= 0.0 or beta_gas <= 0.0:
        raise ValueError(f"Mesh beta must be >0 (liq_beta={beta_liq}, gas_beta={beta_gas}).")

    # store widths for optional interface continuity check
    wL: FloatArray | None = None
    wG: FloatArray | None = None

    # Liquid segment [0, a0]
    if Nl > 0:
        wL = _build_segment_tanh(L=a0, N=Nl, beta=beta_liq, center_bias=float(mesh.liq_center_bias))
        fL = _segment_widths_to_faces(0.0, wL)
    else:
        fL = np.array([0.0, a0], dtype=np.float64)

    # Gas segment [a0, R_inf]
    Lg = Rinf - a0
    if Ng > 0:
        wG = _build_segment_tanh(L=Lg, N=Ng, beta=beta_gas, center_bias=float(mesh.gas_center_bias))
        fG = _segment_widths_to_faces(a0, wG)
    else:
        fG = np.array([a0, Rinf], dtype=np.float64)

    # Optional interface continuity check on adjacent cell widths
    enforce_iface = bool(getattr(cfg.checks, "enforce_grid_state_props_split", False)) and bool(
        getattr(mesh, "enforce_interface_continuity", False)
    )
    if enforce_iface and (wL is not None) and (wG is not None):
        dr_liq = float(wL[-1])
        dr_gas = float(wG[0])
        tol = float(getattr(mesh, "continuity_tol", 0.0))
        denom = max(dr_liq, dr_gas, 1.0)
        rel_err = abs(dr_liq - dr_gas) / denom
        if rel_err > tol:
            raise ValueError(
                "Interface grid continuity violated: "
                f"dr_liq={dr_liq:.6e}, dr_gas={dr_gas:.6e}, rel_err={rel_err:.3e} > tol={tol:.3e}"
            )

    faces_r = np.concatenate([fL, fG[1:]], axis=0)

    iface = Nl
    faces_A = 4.0 * np.pi * faces_r * faces_r
    cells_rc = 0.5 * (faces_r[:-1] + faces_r[1:])
    cells_V = (4.0 * np.pi / 3.0) * (
        np.maximum(faces_r[1:], 0.0) ** 3 - np.maximum(faces_r[:-1], 0.0) ** 3
    )

    grid = Grid1D(
        Nl=Nl,
        Ng=Ng,
        Nc=Nl + Ng,
        r_c=cells_rc,
        r_f=faces_r,
        V_c=cells_V,
        A_f=faces_A,
        iface_f=iface,
    )

    # Grid1D.__post_init__ performs consistency checks.
    return grid
