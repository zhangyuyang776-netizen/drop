from core.layout import build_layout
from core.types import CaseSpecies
from tests.utils_layout import make_case_config, make_simple_grid


def test_all_minus_closure_mode_uses_all_non_closure_species():
    grid = make_simple_grid()
    species = CaseSpecies(
        gas_balance_species="N2",
        gas_species=["O2", "N2", "FUEL"],
        liq_species=["FUEL"],
        liq_balance_species="FUEL",
        liq2gas_map={"FUEL": "FUEL"},
    )
    cfg = make_case_config(grid, species)
    layout = build_layout(cfg, grid)

    assert layout.Ns_g_eff == len(species.gas_species) - 1
    assert layout.gas_species_reduced == ["O2", "FUEL"]
    assert layout.gas_full_to_reduced["N2"] is None


def test_condensables_only_mode_solves_single_condensable():
    grid = make_simple_grid()
    g_name = "FUEL"
    closure = "N2"
    species = CaseSpecies(
        gas_balance_species=closure,
        gas_species=[g_name, closure, "O2"],
        solve_gas_mode="condensables_only",
        liq_species=["FUEL"],
        liq_balance_species="FUEL",
        liq2gas_map={"FUEL": g_name},
    )
    cfg = make_case_config(grid, species)
    layout = build_layout(cfg, grid)

    assert layout.Ns_g_eff == 1
    assert layout.gas_species_reduced == [g_name]
    assert layout.gas_full_to_reduced[g_name] == 0
    assert layout.gas_full_to_reduced[closure] is None


def test_explicit_list_mode_respects_mechanism_order():
    grid = make_simple_grid()
    species = CaseSpecies(
        gas_balance_species="N2",
        gas_species=["O2", "N2", "AR", "NC12H26"],
        solve_gas_mode="explicit_list",
        solve_gas_species=["O2", "NC12H26"],
        liq_species=["NC12H26"],
        liq_balance_species="NC12H26",
        liq2gas_map={"NC12H26": "NC12H26"},
    )
    cfg = make_case_config(grid, species)
    layout = build_layout(cfg, grid)

    assert layout.Ns_g_eff == 2
    assert layout.gas_species_reduced == ["O2", "NC12H26"]
    assert layout.gas_full_to_reduced["N2"] is None
    assert layout.gas_reduced_to_full_idx == [0, 3]
