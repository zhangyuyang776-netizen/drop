"""
Unknown layout definition and state pack/unpack utilities.

Principles:
- Residual block order matches layout order: (1) Tg, (2) Yg (reduced), (3) Tl, (4) Yl (reduced),
  (5) interface scalars (Ts, mpp, Rd as enabled).
- Closure species never appear in the unknown vector; they are reconstructed explicitly.
- Unknown indices must come from UnknownLayout helpers (no hand-rolled math).
- Modules read/write unknowns only via State + this layout; no direct mutation of u elsewhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Set, Tuple

import numpy as np

from .types import CaseConfig, Grid1D, State

FloatArray = np.ndarray
LAYOUT_VERSION = "step2-final"


@dataclass(slots=True)
class VarEntry:
    """Single unknown entry metadata."""

    i: int
    kind: str  # "Yg" | "Tg" | "Yl" | "Tl" | "Ts" | "Rd" | "mpp"
    phase: str  # "gas" | "liq" | "interface"
    cell: Optional[int]
    spec: Optional[int]  # reduced species index
    name: str


@dataclass(slots=True)
class UnknownLayout:
    """Layout of the global unknown vector."""

    size: int
    entries: List[VarEntry]
    blocks: Dict[str, slice]
    Ng: int
    Nl: int
    Ns_g_full: int
    Ns_g_eff: int
    Ns_l_full: int
    Ns_l_eff: int
    gas_species_full: List[str]
    gas_species_reduced: List[str]
    gas_closure_species: Optional[str]
    gas_full_to_reduced: Dict[str, Optional[int]]
    gas_reduced_to_full_idx: List[int]
    gas_closure_index: Optional[int]
    liq_species_full: List[str]
    liq_species_reduced: List[str]
    liq_closure_species: Optional[str]
    liq_full_to_reduced: Dict[str, Optional[int]]
    liq_reduced_to_full_idx: List[int]
    liq_closure_index: Optional[int]

    def n_dof(self) -> int:
        return self.size

    def has_block(self, name: str) -> bool:
        return name in self.blocks

    def require_block(self, name: str) -> slice:
        if name not in self.blocks:
            raise ValueError(f"Unknown block '{name}' not present in layout.")
        return self.blocks[name]

    def idx_Tg(self, ig: int) -> int:
        sl = self.require_block("Tg")
        if ig < 0 or ig >= self.Ng:
            raise IndexError(f"Tg cell index {ig} out of range [0,{self.Ng})")
        return sl.start + ig

    def idx_Yg(self, k_red: int, ig: int) -> int:
        sl = self.require_block("Yg")
        if k_red < 0 or k_red >= self.Ns_g_eff:
            raise IndexError(f"Yg reduced species index {k_red} out of range [0,{self.Ns_g_eff})")
        if ig < 0 or ig >= self.Ng:
            raise IndexError(f"Yg cell index {ig} out of range [0,{self.Ng})")
        return sl.start + ig * self.Ns_g_eff + k_red

    def idx_Tl(self, il: int) -> int:
        sl = self.require_block("Tl")
        if il < 0 or il >= self.Nl:
            raise IndexError(f"Tl cell index {il} out of range [0,{self.Nl})")
        return sl.start + il

    def idx_Yl(self, k_red: int, il: int) -> int:
        sl = self.require_block("Yl")
        if k_red < 0 or k_red >= self.Ns_l_eff:
            raise IndexError(f"Yl reduced species index {k_red} out of range [0,{self.Ns_l_eff})")
        if il < 0 or il >= self.Nl:
            raise IndexError(f"Yl cell index {il} out of range [0,{self.Nl})")
        return sl.start + il * self.Ns_l_eff + k_red

    def idx_Ts(self) -> int:
        sl = self.require_block("Ts")
        return sl.start

    def idx_mpp(self) -> int:
        sl = self.require_block("mpp")
        return sl.start

    def idx_Rd(self) -> int:
        sl = self.require_block("Rd")
        return sl.start


def _build_species_mapping(full: List[str], closure: Optional[str], active: Optional[Set[str]] = None):
    """
    Map mechanism-ordered full species list to reduced indices.

    active=None: all species except closure are active (backward compatible)
    active=set(...): only names in active become reduced unknowns; others map to None
    """
    if len(full) != len(set(full)):
        raise ValueError(f"Duplicate species names in list: {full}")
    if closure is not None and closure not in full:
        raise ValueError(f"Closure species '{closure}' not found in species list {full}")
    if active is not None:
        missing = [name for name in active if name not in full]
        if missing:
            raise ValueError(f"Active species not present in full list: {missing}")

    reduced: List[str] = []
    reduced_to_full_idx: List[int] = []
    full_to_reduced: Dict[str, Optional[int]] = {}
    closure_idx: Optional[int] = None

    for i_full, name in enumerate(full):
        if closure is not None and name == closure:
            full_to_reduced[name] = None
            closure_idx = i_full
            continue
        if active is not None and name not in active:
            full_to_reduced[name] = None
            continue
        k_red = len(reduced)
        reduced.append(name)
        reduced_to_full_idx.append(i_full)
        full_to_reduced[name] = k_red

    return reduced, full_to_reduced, reduced_to_full_idx, closure_idx

def build_layout(cfg: CaseConfig, grid: Grid1D) -> UnknownLayout:
    """
    Build the unknown layout following residual ordering.

    Block order (aligned with residual assembly):
      1) gas energy (Tg)
      2) gas species (reduced)
      3) liquid energy (Tl)
      4) liquid species (reduced)
      5) interface scalars (Ts, mpp, Rd in that order if enabled)
    """
    gas_full = list(cfg.species.gas_species)
    if not gas_full:
        raise ValueError("cfg.species.gas_species is empty. Preprocess must load mechanism and fill it.")
    mode = getattr(cfg.species, "solve_gas_mode", "all_minus_closure")
    gas_closure = cfg.species.gas_balance_species

    if mode == "all_minus_closure":
        gas_active: Set[str] = set(gas_full)
        gas_active.discard(gas_closure)
    elif mode == "condensables_only":
        l_name = cfg.species.liq_balance_species
        g_map = cfg.species.liq2gas_map
        if l_name not in g_map:
            raise ValueError(f"liq_balance_species '{l_name}' not found in liq2gas_map {g_map}")
        g_name = g_map[l_name]
        if g_name == gas_closure:
            raise ValueError("Condensable species cannot be closure species")
        gas_active = {g_name}
    elif mode == "explicit_list":
        names = list(getattr(cfg.species, "solve_gas_species", []))
        if not names:
            raise ValueError("solve_gas_species is empty for explicit_list")
        gas_active = set(names)
        if gas_closure in gas_active:
            raise ValueError("closure species cannot be solved explicitly")
        missing = [s for s in gas_active if s not in gas_full]
        if missing:
            raise ValueError(f"explicit gas species not in mechanism list: {missing}")
    else:
        raise ValueError(f"Unknown solve_gas_mode: {mode}")

    gas_reduced, gas_full_to_reduced, gas_reduced_to_full_idx, gas_closure_idx = _build_species_mapping(
        gas_full, gas_closure, active=gas_active
    )

    if mode == "condensables_only" and len(gas_reduced) != 1:
        raise ValueError(
            f"condensables_only mode expected 1 active species, got {len(gas_reduced)} (full={gas_full})"
        )
    if mode == "all_minus_closure":
        expected = len(gas_full) - (1 if gas_closure is not None else 0)
        if len(gas_reduced) != expected:
            raise ValueError(
                f"all_minus_closure mode expected {expected} active species, got {len(gas_reduced)} (full={gas_full})"
            )
    if mode == "explicit_list" and len(gas_reduced) != len(gas_active):
        raise ValueError(
            f"explicit_list mode expected {len(gas_active)} active species, got {len(gas_reduced)}"
        )

    liq_full = list(cfg.species.liq_species)
    liq_closure = cfg.species.liq_balance_species
    liq_reduced, liq_full_to_reduced, liq_reduced_to_full_idx, liq_closure_idx = _build_species_mapping(
        liq_full, liq_closure
    )

    Ns_g_eff = len(gas_reduced)
    Ns_l_eff = len(liq_reduced)

    include_gas_energy = cfg.physics.solve_Tg
    include_gas_species = cfg.physics.solve_Yg
    include_liq_energy = cfg.physics.solve_Tl
    include_liq_species = cfg.physics.solve_Yl
    include_Ts = cfg.physics.include_Ts
    include_mpp = cfg.physics.include_mpp
    include_Rd = cfg.physics.include_Rd

    entries: List[VarEntry] = []
    blocks: Dict[str, slice] = {}
    cursor = 0

    # 1) Gas energy (temperature)
    if include_gas_energy and grid.Ng > 0:
        start = cursor
        for ig in range(grid.Ng):
            entries.append(
                VarEntry(
                    i=cursor,
                    kind="Tg",
                    phase="gas",
                    cell=ig,
                    spec=None,
                    name=f"Tg[{ig}]",
                )
            )
            cursor += 1
        blocks["Tg"] = slice(start, cursor)

    # 2) Gas species (cell outer, reduced species inner)
    if include_gas_species and Ns_g_eff > 0 and grid.Ng > 0:
        start = cursor
        for ig in range(grid.Ng):
            for k_red, name in enumerate(gas_reduced):
                entries.append(
                    VarEntry(
                        i=cursor,
                        kind="Yg",
                        phase="gas",
                        cell=ig,
                        spec=k_red,
                        name=f"Yg[{name}@{ig}]",
                    )
                )
                cursor += 1
        blocks["Yg"] = slice(start, cursor)

    # 3) Liquid energy (temperature)
    if include_liq_energy and grid.Nl > 0:
        start = cursor
        for il in range(grid.Nl):
            entries.append(
                VarEntry(
                    i=cursor,
                    kind="Tl",
                    phase="liq",
                    cell=il,
                    spec=None,
                    name=f"Tl[{il}]",
                )
            )
            cursor += 1
        blocks["Tl"] = slice(start, cursor)

    # 4) Liquid species (cell outer, reduced species inner)
    if include_liq_species and Ns_l_eff > 0 and grid.Nl > 0:
        start = cursor
        for il in range(grid.Nl):
            for k_red, name in enumerate(liq_reduced):
                entries.append(
                    VarEntry(
                        i=cursor,
                        kind="Yl",
                        phase="liq",
                        cell=il,
                        spec=k_red,
                        name=f"Yl[{name}@{il}]",
                    )
                )
                cursor += 1
        blocks["Yl"] = slice(start, cursor)

    # 5) Interface scalars
    if include_Ts:
        start = cursor
        entries.append(
            VarEntry(
                i=cursor,
                kind="Ts",
                phase="interface",
                cell=None,
                spec=None,
                name="Ts",
            )
        )
        cursor += 1
        blocks["Ts"] = slice(start, cursor)

    if include_mpp:
        start = cursor
        entries.append(
            VarEntry(
                i=cursor,
                kind="mpp",
                phase="interface",
                cell=None,
                spec=None,
                name="mpp",
            )
        )
        cursor += 1
        blocks["mpp"] = slice(start, cursor)

    if include_Rd:
        start = cursor
        entries.append(
            VarEntry(
                i=cursor,
                kind="Rd",
                phase="interface",
                cell=None,
                spec=None,
                name="Rd",
            )
        )
        cursor += 1
        blocks["Rd"] = slice(start, cursor)

    return UnknownLayout(
        size=cursor,
        entries=entries,
        blocks=blocks,
        Ng=grid.Ng,
        Nl=grid.Nl,
        Ns_g_full=len(gas_full),
        Ns_g_eff=Ns_g_eff,
        Ns_l_full=len(liq_full),
        Ns_l_eff=Ns_l_eff,
        gas_species_full=gas_full,
        gas_species_reduced=gas_reduced,
        gas_closure_species=gas_closure,
        gas_full_to_reduced=gas_full_to_reduced,
        gas_reduced_to_full_idx=gas_reduced_to_full_idx,
        gas_closure_index=gas_closure_idx,
        liq_species_full=liq_full,
        liq_species_reduced=liq_reduced,
        liq_closure_species=liq_closure,
        liq_full_to_reduced=liq_full_to_reduced,
        liq_reduced_to_full_idx=liq_reduced_to_full_idx,
        liq_closure_index=liq_closure_idx,
    )


def pack_state(
    state: State,
    layout: UnknownLayout,
    *,
    refs: Optional[Mapping[str, float]] = None,
) -> Tuple[np.ndarray, np.ndarray, List[VarEntry]]:
    """
    Pack State into a 1D unknown vector following the given layout.

    refs:
      - "T_ref": temperature scaling reference (default 298.15)
      - "Rd_ref": radius scaling reference (default max(|Rd|, 1.0))
      - "mpp_ref": evaporation scaling reference (default 1e-3)
    """
    refs = refs or {}
    T_ref = float(refs.get("T_ref", 298.15))
    Rd_ref = refs.get("Rd_ref", None)
    m_ref = float(refs.get("mpp_ref", 1.0e-3))

    u = np.zeros(layout.size, dtype=np.float64)
    scale_u = np.ones(layout.size, dtype=np.float64)

    for entry in layout.entries:
        i = entry.i
        kind = entry.kind

        if kind == "Yg":
            ig = entry.cell
            k_red = entry.spec
            k_full = layout.gas_reduced_to_full_idx[k_red]
            u[i] = state.Yg[k_full, ig]
            scale_u[i] = 1.0
        elif kind == "Tg":
            ig = entry.cell
            val = float(state.Tg[ig])
            u[i] = val
            scale_u[i] = max(abs(val), T_ref)
        elif kind == "Yl":
            il = entry.cell
            k_red = entry.spec
            k_full = layout.liq_reduced_to_full_idx[k_red]
            u[i] = state.Yl[k_full, il]
            scale_u[i] = 1.0
        elif kind == "Tl":
            il = entry.cell
            val = float(state.Tl[il])
            u[i] = val
            scale_u[i] = max(abs(val), T_ref)
        elif kind == "Ts":
            val = float(state.Ts)
            u[i] = val
            scale_u[i] = max(abs(val), T_ref)
        elif kind == "Rd":
            val = float(state.Rd)
            u[i] = val
            if Rd_ref is None:
                scale_u[i] = max(abs(val), 1.0)
            else:
                scale_u[i] = max(abs(val), float(Rd_ref))
        elif kind == "mpp":
            val = float(state.mpp)
            u[i] = val
            scale_u[i] = max(abs(val), m_ref)
        else:
            raise ValueError(f"Unknown variable kind: {kind}")

    return u, scale_u, list(layout.entries)


def _reconstruct_closure(
    Y_full: np.ndarray,
    reduced_to_full_idx: List[int],
    closure_idx: Optional[int],
    *,
    tol: float,
    clip_negative: bool,
    phase: str,
) -> None:
    """Compute closure species from reduced ones; mutate Y_full in place."""
    if closure_idx is None:
        return
    if Y_full.shape[0] <= closure_idx:
        raise ValueError(f"{phase} closure index {closure_idx} out of bounds for shape {Y_full.shape}")
    sum_other = np.sum(Y_full, axis=0) - Y_full[closure_idx, :]
    closure = 1.0 - sum_other

    if np.any(closure < -tol):
        raise ValueError(
            f"{phase} closure species negative beyond tol={tol}: min={float(np.min(closure)):.3e}"
        )
    if np.any(closure > 1.0 + tol):
        raise ValueError(
            f"{phase} closure species exceeds 1 beyond tol={tol}: max={float(np.max(closure)):.3e}"
        )

    if clip_negative:
        closure = np.where((closure < 0.0) & (closure >= -tol), 0.0, closure)

    Y_full[closure_idx, :] = closure


def apply_u_to_state(
    state: State,
    u: np.ndarray,
    layout: UnknownLayout,
    *,
    tol_closure: float = 1e-12,
    clip_negative_closure: bool = False,
) -> State:
    """Apply unknown vector to a State and reconstruct closure species."""
    u = np.asarray(u, dtype=np.float64)
    if u.size < layout.size:
        raise ValueError(f"u size {u.size} is smaller than layout size {layout.size}")

    Tg = np.array(state.Tg, copy=True, dtype=np.float64)
    Yg = np.array(state.Yg, copy=True, dtype=np.float64)
    Tl = np.array(state.Tl, copy=True, dtype=np.float64)
    Yl = np.array(state.Yl, copy=True, dtype=np.float64)
    Ts = float(state.Ts)
    Rd = float(state.Rd)
    mpp = float(state.mpp)

    for entry in layout.entries:
        i = entry.i
        kind = entry.kind
        val = float(u[i])

        if kind == "Yg":
            ig = entry.cell
            k_red = entry.spec
            k_full = layout.gas_reduced_to_full_idx[k_red]
            Yg[k_full, ig] = val
        elif kind == "Tg":
            ig = entry.cell
            Tg[ig] = val
        elif kind == "Yl":
            il = entry.cell
            k_red = entry.spec
            k_full = layout.liq_reduced_to_full_idx[k_red]
            Yl[k_full, il] = val
        elif kind == "Tl":
            il = entry.cell
            Tl[il] = val
        elif kind == "Ts":
            Ts = val
        elif kind == "Rd":
            Rd = val
        elif kind == "mpp":
            mpp = val
        else:
            raise ValueError(f"Unknown variable kind: {kind}")

    if ("Yg" in layout.blocks) and (layout.gas_closure_index is not None):
        _reconstruct_closure(
            Y_full=Yg,
            reduced_to_full_idx=layout.gas_reduced_to_full_idx,
            closure_idx=layout.gas_closure_index,
            tol=tol_closure,
            clip_negative=clip_negative_closure,
            phase="Gas",
        )
    if ("Yl" in layout.blocks) and (layout.liq_closure_index is not None):
        _reconstruct_closure(
            Y_full=Yl,
            reduced_to_full_idx=layout.liq_reduced_to_full_idx,
            closure_idx=layout.liq_closure_index,
            tol=tol_closure,
            clip_negative=clip_negative_closure,
            phase="Liquid",
        )

    return State(Tg=Tg, Yg=Yg, Tl=Tl, Yl=Yl, Ts=Ts, mpp=mpp, Rd=Rd)


def assert_pack_apply_consistency(state: State, layout: UnknownLayout, rtol=1e-12, atol=1e-14) -> None:
    """Pack then apply and assert the state is unchanged (for tests)."""
    u, _, _ = pack_state(state, layout)
    state2 = apply_u_to_state(state, u, layout)

    if "Tg" in layout.blocks:
        np.testing.assert_allclose(state2.Tg, state.Tg, rtol=rtol, atol=atol)
    if "Yg" in layout.blocks:
        np.testing.assert_allclose(state2.Yg, state.Yg, rtol=rtol, atol=atol)
    if "Tl" in layout.blocks:
        np.testing.assert_allclose(state2.Tl, state.Tl, rtol=rtol, atol=atol)
    if "Yl" in layout.blocks:
        np.testing.assert_allclose(state2.Yl, state.Yl, rtol=rtol, atol=atol)
    if "Ts" in layout.blocks:
        np.testing.assert_allclose(state2.Ts, state.Ts, rtol=rtol, atol=atol)
    if "Rd" in layout.blocks:
        np.testing.assert_allclose(state2.Rd, state.Rd, rtol=rtol, atol=atol)
    if "mpp" in layout.blocks:
        np.testing.assert_allclose(state2.mpp, state.mpp, rtol=rtol, atol=atol)
