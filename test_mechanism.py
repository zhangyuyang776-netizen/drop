#!/usr/bin/env python3
"""Test if the mechanism file loads correctly."""

import sys
from pathlib import Path

try:
    import cantera as ct

    mech_path = Path(__file__).parent / "mechanism" / "mech.yaml"
    print(f"Loading mechanism from: {mech_path}")
    print(f"File exists: {mech_path.exists()}")

    gas = ct.Solution(str(mech_path))

    print(f"\nMechanism loaded successfully!")
    print(f"Number of species: {gas.n_species}")
    print(f"Species names: {list(gas.species_names)}")
    print(f"Number of reactions: {gas.n_reactions}")

    # Test setting TPY
    gas.TPY = 300.0, 101325.0, {"N2": 0.79, "O2": 0.21}
    print(f"\nTest TPY set successfully!")
    print(f"Density: {gas.density:.3f} kg/m3")
    print(f"Cp: {gas.cp_mass:.1f} J/kg/K")
    print(f"Thermal conductivity: {gas.thermal_conductivity:.6f} W/m/K")

except ImportError as e:
    print(f"ERROR: Cannot import cantera: {e}")
    sys.exit(1)
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\nAll tests passed!")
