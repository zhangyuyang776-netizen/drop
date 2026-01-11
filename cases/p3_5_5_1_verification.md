# P3.5-5-1 Verification Guide

## Purpose

Verify that the "少步 SNES" smoke mode infrastructure works correctly:
1. `smoke: true` limits SNES iterations to 1
2. `fieldsplit + mf` auto-promotes to `mfpc_sparse_fd`
3. Diagnostic info is recorded in `result.diag.extra["snes_smoke"]`

## Verification Cases

Two minimal smoke test configurations are provided:

### 1. Additive Fieldsplit
**File**: `p3_5_5_1_smoke_additive.yaml`
- Tests additive fieldsplit path
- Uses default subsolvers (auto-filled)

### 2. Schur Fieldsplit
**File**: `p3_5_5_1_smoke_schur.yaml`
- Tests Schur fieldsplit path
- Uses `schur_fact_type: "lower"` (default)

## Key Configuration Features

Both configs use:
- **Small grid**: `N_liq: 2`, `N_gas: 10` (faster execution)
- **Single timestep**: `max_steps: 1`
- **Smoke mode**: `nonlinear.smoke: true`
- **PETSc MPI backend**: `nonlinear.backend: "petsc_mpi"`
- **Deliberate mf**: `petsc.jacobian_mode: "mf"` (triggers auto-promotion)
- **Fieldsplit PC**: `solver.linear.pc_type: "fieldsplit"`

## Running Verification

### Command (MPI required)

```bash
# Test additive
mpiexec -n 2 python driver/run_droplet.py cases/p3_5_5_1_smoke_additive.yaml

# Test schur
mpiexec -n 2 python driver/run_droplet.py cases/p3_5_5_1_smoke_schur.yaml
```

### Expected Behavior

**Must NOT happen**:
- Hang or segfault
- More than 1 SNES iteration
- Pure `mf` jacobian mode with fieldsplit (should auto-promote)

**Must happen**:
- Returns successfully (even if not converged)
- Exactly 1 SNES iteration (smoke mode limit)
- Auto-promotion: `mf` → `mfpc_sparse_fd`

## Verification Checks

### From Log Output

Look for messages like:
```
pc_type=fieldsplit with jacobian_mode=mf uses identity P;
promoting to mfpc_sparse_fd for a usable preconditioner.
```

### From Diagnostic Data

If you have access to `result.diag.extra`, verify:

```python
# Load result (pseudo-code)
result = run_case("p3_5_5_1_smoke_additive.yaml")

# Check smoke diagnostics
smoke_diag = result.diag.extra["snes_smoke"]
assert smoke_diag["smoke_enabled"] == True
assert smoke_diag["max_outer_iter_effective"] == 1
assert smoke_diag["jacobian_mode_effective"] == "mfpc_sparse_fd"

# Check PC structure
pc_diag = result.diag.extra.get("pc_structured", {})
assert pc_diag.get("uses_amat") == False  # P3.4 constraint
assert pc_diag.get("fieldsplit_type") in ["additive", "schur"]

# Check SNES actually stopped at 1 iteration
assert result.diag.n_iter <= 1
```

### Manual Inspection

For manual verification without code access:

1. **Log messages**: Search for "promoting to mfpc_sparse_fd"
2. **Iteration count**: Check SNES output shows `n_iter=1` or `n_iter=0`
3. **Completion**: Program exits without error
4. **Time**: Should complete in seconds (not minutes)

## Success Criteria

P3.5-5-1 verification passes if:

✓ Both configs run without hang/segfault
✓ SNES iterations limited to 1 (smoke mode active)
✓ Jacobian mode shows `mfpc_sparse_fd` (auto-promoted from `mf`)
✓ `pc_structured` diagnostics present (if debug enabled)
✓ `uses_amat=False` enforced for fieldsplit

## Troubleshooting

### If hang occurs
- Check watchdog is not disabled
- Verify MPI is working: `mpiexec -n 2 hostname`
- Try running with `-x DROPLET_PETSC_DEBUG=1` for more output

### If auto-promotion doesn't happen
- Check log for warning message about promotion
- Verify `petsc.jacobian_mode: "mf"` in config
- Verify `solver.linear.pc_type: "fieldsplit"` in config

### If more than 1 iteration
- Check `nonlinear.smoke: true` is set
- Check `_normalize_cfg_for_snes_smoke()` is called
- Check smoke_diag is recorded

## Next Steps

After P3.5-5-1 verification passes:
- **P3.5-5-2**: Add options prefix isolation
- **P3.5-5-3**: Create parametrized MPI smoke tests
- **P3.5-5-4**: Integrate into regression suite
