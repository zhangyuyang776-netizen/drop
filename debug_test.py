#!/usr/bin/env python3
"""Debug script to run test case with detailed error output."""

import logging
import sys
from pathlib import Path
import yaml

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from driver.run_scipy_case import run_case

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)

def test_case_a():
    print("=" * 80)
    print("Testing case A: step13_A_equilibrium_const_props.yaml")
    print("=" * 80)

    yaml_path = Path(__file__).parent / "cases" / "step13_A_equilibrium_const_props.yaml"

    # Create temporary output directory
    tmp_out = Path(__file__).parent / "test_output_debug"
    tmp_out.mkdir(exist_ok=True)

    # Load and modify YAML to use temp output
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)

    cfg["paths"]["output_root"] = str(tmp_out)
    cfg["paths"]["case_dir"] = str(tmp_out / cfg["case"]["id"])

    # Write modified YAML
    tmp_yaml = tmp_out / "test_A.yaml"
    with open(tmp_yaml, "w") as f:
        yaml.safe_dump(cfg, f)

    rc = run_case(str(tmp_yaml), max_steps=25, log_level=logging.DEBUG)
    print(f"\nReturn code: {rc}")
    return rc

if __name__ == "__main__":
    try:
        rc = test_case_a()
        sys.exit(rc)
    except Exception as e:
        print(f"\nException caught: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(99)
