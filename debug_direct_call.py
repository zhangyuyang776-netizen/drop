#!/usr/bin/env python3
"""
Direct call debug script that bypasses exception handling to show full traceback.
This script directly calls the internal functions without the try-except wrapper.
"""

import sys
from pathlib import Path
import yaml
import tempfile
import traceback

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

def main():
    print("=" * 80)
    print("Direct debug - bypassing exception handling")
    print("=" * 80)

    # Import after adding to path
    from driver.run_scipy_case import (
        _load_case_config,
        _prepare_run_dir,
        build_gas_model,
        _maybe_fill_gas_species,
        build_grid,
        build_layout,
        build_liquid_model,
        _build_initial_state,
    )
    from properties.aggregator import build_props_from_state

    # Load test case YAML
    yaml_path = Path(__file__).parent / "cases" / "step13_A_equilibrium_const_props.yaml"
    print(f"\n1. Loading YAML: {yaml_path}")

    # Create temp directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Load and patch YAML
        with open(yaml_path) as f:
            raw_cfg = yaml.safe_load(f)

        case_id = raw_cfg["case"]["id"]
        raw_cfg["paths"]["output_root"] = str(tmp_path)
        raw_cfg["paths"]["case_dir"] = str(tmp_path / case_id)
        raw_cfg["paths"]["mechanism_dir"] = str(Path(__file__).parent / "mechanism")

        # Write patched YAML
        tmp_yaml = tmp_path / f"{case_id}.yaml"
        with open(tmp_yaml, "w") as f:
            yaml.safe_dump(raw_cfg, f)

        print(f"   Temp YAML: {tmp_yaml}")

        # Now call each step directly to see where it fails
        try:
            print("\n2. Loading case config...")
            cfg = _load_case_config(str(tmp_yaml))
            print(f"   ✓ Config loaded: {cfg.case.id}")

        except Exception as e:
            print(f"\n❌ FAILED at: Loading case config")
            print(f"Exception: {type(e).__name__}: {e}")
            traceback.print_exc()
            return 1

        try:
            print("\n3. Building gas model...")
            gas_model = build_gas_model(cfg)
            print(f"   ✓ Gas model built: {gas_model.gas.n_species} species")
            print(f"   Species: {list(gas_model.gas.species_names)}")

        except Exception as e:
            print(f"\n❌ FAILED at: Building gas model")
            print(f"Exception: {type(e).__name__}: {e}")
            traceback.print_exc()
            return 2

        try:
            print("\n4. Filling gas species...")
            _maybe_fill_gas_species(cfg, gas_model)
            print(f"   ✓ Gas species filled: {cfg.species.gas_species}")

        except Exception as e:
            print(f"\n❌ FAILED at: Filling gas species")
            print(f"Exception: {type(e).__name__}: {e}")
            traceback.print_exc()
            return 3

        try:
            print("\n5. Building grid...")
            grid = build_grid(cfg)
            print(f"   ✓ Grid built: Ng={grid.Ng}, Nl={grid.Nl}")

        except Exception as e:
            print(f"\n❌ FAILED at: Building grid")
            print(f"Exception: {type(e).__name__}: {e}")
            traceback.print_exc()
            return 4

        try:
            print("\n6. Building layout...")
            layout = build_layout(cfg, grid)
            print(f"   ✓ Layout built: {layout.n_dof()} DOFs")

        except Exception as e:
            print(f"\n❌ FAILED at: Building layout")
            print(f"Exception: {type(e).__name__}: {e}")
            traceback.print_exc()
            return 5

        try:
            print("\n7. Building liquid model...")
            liq_model = build_liquid_model(cfg) if cfg.physics.enable_liquid else None
            if liq_model:
                print(f"   ✓ Liquid model built: {len(liq_model.liq_names)} species")
                print(f"   Liq species: {liq_model.liq_names}")
                print(f"   Fluids: {liq_model.fluids}")
            else:
                print(f"   ✓ No liquid model (enable_liquid=False)")

        except Exception as e:
            print(f"\n❌ FAILED at: Building liquid model")
            print(f"Exception: {type(e).__name__}: {e}")
            traceback.print_exc()
            return 6

        try:
            print("\n8. Building initial state...")
            state = _build_initial_state(cfg, grid, gas_model, liq_model)
            print(f"   ✓ Initial state built")
            print(f"   Tg shape: {state.Tg.shape}")
            print(f"   Yg shape: {state.Yg.shape}")
            print(f"   Tl shape: {state.Tl.shape}")
            print(f"   Yl shape: {state.Yl.shape}")
            print(f"   Ts: {state.Ts}")
            print(f"   Rd: {state.Rd}")
            print(f"   mpp: {state.mpp}")

        except Exception as e:
            print(f"\n❌ FAILED at: Building initial state")
            print(f"Exception: {type(e).__name__}: {e}")
            traceback.print_exc()
            return 7

        try:
            print("\n9. Building properties from state...")
            props, extras = build_props_from_state(cfg, grid, state, gas_model, liq_model)
            print(f"   ✓ Properties built")
            print(f"   rho_g shape: {props.rho_g.shape}")
            print(f"   rho_l shape: {props.rho_l.shape}")

        except Exception as e:
            print(f"\n❌ FAILED at: Building properties from state")
            print(f"Exception: {type(e).__name__}: {e}")
            traceback.print_exc()
            return 8

        print("\n" + "=" * 80)
        print("✅ All initialization steps completed successfully!")
        print("=" * 80)
        print("\nThe error must occur during time stepping.")
        print("Run the full test to see time stepping errors.")

        return 0

if __name__ == "__main__":
    try:
        rc = main()
        sys.exit(rc)
    except Exception as e:
        print("\n" + "=" * 80)
        print("UNCAUGHT EXCEPTION:")
        print("=" * 80)
        print(f"Type: {type(e).__name__}")
        print(f"Message: {e}")
        print("\nFull traceback:")
        traceback.print_exc()
        sys.exit(99)
