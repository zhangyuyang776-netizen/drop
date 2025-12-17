#!/usr/bin/env python3
"""
Debug version of test with detailed logging.
Run this to see the full error traceback.
"""

import logging
import sys
from pathlib import Path
import yaml
import tempfile

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

def test_case_debug():
    """Run test case with detailed DEBUG logging to see full error."""
    # Configure detailed logging to see ALL messages
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
        stream=sys.stdout
    )

    print("=" * 80)
    print("Running test with DEBUG logging")
    print("=" * 80)

    from driver.run_scipy_case import run_case

    # Test case A
    yaml_path = Path(__file__).parent / "cases" / "step13_A_equilibrium_const_props.yaml"
    print(f"\nYAML file: {yaml_path}")
    print(f"File exists: {yaml_path.exists()}")

    # Create temp directory for output
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        print(f"Temp output: {tmp_path}")

        # Load and patch YAML
        with open(yaml_path) as f:
            cfg = yaml.safe_load(f)

        case_id = cfg["case"]["id"]
        cfg["paths"]["output_root"] = str(tmp_path)
        cfg["paths"]["case_dir"] = str(tmp_path / case_id)
        cfg["paths"]["mechanism_dir"] = str(Path(__file__).parent / "mechanism")

        # Write patched YAML
        tmp_yaml = tmp_path / f"{case_id}.yaml"
        with open(tmp_yaml, "w") as f:
            yaml.safe_dump(cfg, f)

        print(f"Running test...")
        print("-" * 80)

        # Run with DEBUG level
        rc = run_case(str(tmp_yaml), max_steps=25, log_level=logging.DEBUG)

        print("-" * 80)
        print(f"\nReturn code: {rc}")

        if rc != 0:
            print(f"\n❌ Test FAILED with return code {rc}")
            if rc == 99:
                print("Return code 99 indicates an unhandled exception was caught.")
                print("Check the traceback above for details.")
        else:
            print(f"\n✅ Test PASSED")

        return rc

if __name__ == "__main__":
    try:
        rc = test_case_debug()
        sys.exit(rc)
    except Exception as e:
        print(f"\n{'='*80}")
        print(f"EXCEPTION CAUGHT IN MAIN:")
        print(f"{'='*80}")
        print(f"Type: {type(e).__name__}")
        print(f"Message: {e}")
        print(f"\nFull traceback:")
        import traceback
        traceback.print_exc()
        sys.exit(99)
