from core.layout import build_layout
from core.types import CaseSpecies
from tests.utils_layout import make_case_config, make_simple_grid


def test_all_minus_closure_mode_uses_all_non_closure_species():
    grid = make_simple_grid()
    species = CaseSpecies(
        gas_balance_species="N2",
        gas_species_full=["O2", "N2", "FUEL"],
        liq_species=["FUEL"],
        liq_balance_species="FUEL",
        liq2gas_map={"FUEL": "FUEL"},
    )
    cfg = make_case_config(grid, species)
    layout = build_layout(cfg, grid)

    assert layout.Ns_g_eff == len(species.gas_species_full) - 1
    assert layout.gas_species_reduced == ["O2", "FUEL"]
    assert layout.gas_full_to_reduced["N2"] is None


def test_layout_preserves_mechanism_order_when_excluding_closure():
    grid = make_simple_grid()
    species = CaseSpecies(
        gas_balance_species="AR",
        gas_species_full=["O2", "N2", "AR", "NC12H26"],
        liq_species=["NC12H26"],
        liq_balance_species="NC12H26",
        liq2gas_map={"NC12H26": "NC12H26"},
    )
    cfg = make_case_config(grid, species)
    layout = build_layout(cfg, grid)

    assert layout.Ns_g_eff == len(species.gas_species_full) - 1
    assert layout.gas_species_reduced == ["O2", "N2", "NC12H26"]
    assert layout.gas_full_to_reduced["AR"] is None
    assert layout.gas_reduced_to_full_idx == [0, 1, 3]
