# test_timestepper_one_step 修复方案3 - 最终实施计划

## 修改概览

**目标**：在 `build_transport_system` 中实现 `Tg + Tl + Ts + mpp + Rd` 的完全隐式耦合求解。

**核心思路**：
1. 修改 `build_liquid_T_system` 支持"耦合模式"，返回纯局部 Tl 系统
2. 在 `build_transport_system` 中嵌入 Tl 块到全局矩阵
3. 移除 `timestepper` 中的 Stage 2 分裂逻辑
4. 测试配置改回 `solve_Tl=True`

---

## 修改1：`assembly/build_liquid_T_system_SciPy.py`

### 1.1 函数签名添加参数

**位置**：第22行

**修改前**：
```python
def build_liquid_T_system(
    cfg: CaseConfig,
    grid: Grid1D,
    layout: UnknownLayout,
    state_old: State,
    props: Props,
    dt: float,
) -> Tuple[np.ndarray, np.ndarray]:
```

**修改后**：
```python
def build_liquid_T_system(
    cfg: CaseConfig,
    grid: Grid1D,
    layout: UnknownLayout,
    state_old: State,
    props: Props,
    dt: float,
    couple_interface: bool = False,  # 新增参数
) -> Tuple[np.ndarray, np.ndarray]:
```

### 1.2 Docstring 更新

**位置**：第30-37行

**修改后**：
```python
"""
Assemble liquid-phase temperature equation into dense numpy system A x = b (Tl-only).

Scope:
- Only Tl in liquid cells (0..Nl-1); returned system size is (Nl, Nl).
- Implicit conduction with rho_l, cp_l, k_l frozen at state_old.
- Center r=0: symmetry (zero-gradient) via ghost-cell-like stencil.
- Interface r=R_d:
  * If couple_interface=False (Stage 2 mode): Dirichlet T_l(Nl-1) = Ts_given.
  * If couple_interface=True (Stage 1 coupled mode): keep conduction equation,
    coupling provided by Ts energy equation in build_transport_system.

Parameters
----------
couple_interface : bool, default False
    If False: apply Dirichlet BC at interface (for Stage 2 split solve).
    If True: keep conduction equation at interface (for Stage 1 coupled solve).
"""
```

### 1.3 矩阵构建改为纯局部系统

**位置**：第70-71行

**修改前**：
```python
A = np.zeros((Nl, Nl), dtype=np.float64)
b = np.zeros(Nl, dtype=np.float64)
```

**修改后**（保持不变，确认使用局部索引）：
```python
A = np.zeros((Nl, Nl), dtype=np.float64)
b = np.zeros(Nl, dtype=np.float64)
```

**关键**：确保后续所有索引都是 `il`（0..Nl-1），不使用 `layout.idx_Tl(il)`。

检查第78-148行的循环，确认所有 `row` 和列索引都是局部的：
```python
for il in range(Nl):
    row = il  # ← 必须是局部索引，不能是 layout.idx_Tl(il)
    ...
    A[row, row] += aP
    b[row] += b_i
```

### 1.4 界面边界条件分支

**位置**：第150-156行

**修改前**：
```python
# Interface Dirichlet: T_l(Nl-1) = Ts_given (from state_old)
il_bc = Nl - 1
row_bc = il_bc
Ts_given = float(state_old.Ts)
A[row_bc, :] = 0.0
A[row_bc, row_bc] = 1.0
b[row_bc] = Ts_given
```

**修改后**：
```python
# Interface boundary condition
il_bc = Nl - 1
row_bc = il_bc

if not couple_interface:
    # Stage 2 mode: Dirichlet BC (Tl[Nl-1] = Ts_given)
    Ts_given = float(state_old.Ts)
    A[row_bc, :] = 0.0
    A[row_bc, row_bc] = 1.0
    b[row_bc] = Ts_given
else:
    # Stage 1 coupled mode: keep conduction equation at interface
    # Interface energy coupling is handled by _build_Ts_row
    # The conduction equation remains as assembled in the loop above
    pass
```

**说明**：
- `couple_interface=False`（默认）：保持原有 Stage 2 行为
- `couple_interface=True`：保留导热方程，不施加 Dirichlet BC

---

## 修改2：`assembly/build_system_SciPy.py`

### 2.1 在 Tg 块后嵌入 Tl 块

**位置**：在 Tg 方程组装完成后（约第230行，在 "Outer boundary Dirichlet" 之后）

**插入位置**：在 `_apply_outer_dirichlet_Tg(...)` 之后，界面方程之前

**新增代码**：
```python
# --- Liquid temperature equations (Tl block) ---
if layout.has_block("Tl"):
    from assembly.build_liquid_T_system_SciPy import build_liquid_T_system

    A_l, b_l = build_liquid_T_system(
        cfg=cfg,
        grid=grid,
        layout=layout,
        state_old=state_old,
        props=props,
        dt=dt,
        couple_interface=True,  # Coupled mode: no Dirichlet BC at interface
    )

    # Embed local Tl system (Nl×Nl) into global matrix (N×N)
    Nl = grid.Nl
    for il in range(Nl):
        row_global = layout.idx_Tl(il)
        for il2 in range(Nl):
            col_global = layout.idx_Tl(il2)
            A[row_global, col_global] += A_l[il, il2]
        b[row_global] += b_l[il]
```

**说明**：
- 导入 `build_liquid_T_system` 在函数内部（避免循环导入）
- 使用 `couple_interface=True` 获取不含 Dirichlet BC 的纯导热方程
- 将局部 `A_l[il, il2]` 映射到全局 `A[row_global, col_global]`
- 使用 `+=` 以防万一有预先填充的系数（虽然当前没有）

---

## 修改3：`solvers/timestepper.py`

### 3.1 移除 Stage 2 分裂逻辑

**位置**：第173-200行（`_advance_liquid_T_step12` 调用部分）

**修改前**：
```python
# --- Stage 2: liquid temperature update using new Ts as boundary ---
liq_diag: Dict[str, Any] | None = None
liq_lin: Optional[LinearSolveResult] = None
if cfg.physics.solve_Tl and layout.has_block("Tl"):
    try:
        state_new, liq_lin, liq_diag = _advance_liquid_T_step12(
            cfg=cfg,
            grid=grid,
            layout=layout,
            state_tr=state_new,
            props=props_old,
            dt=dt,
        )
    except Exception as exc:
        logger.exception("Liquid temperature solve raised an exception.")
        diag = _build_step_diagnostics_fail(...)
        return StepResult(...)

    if not liq_lin.converged:
        logger.warning("Liquid temperature solve did not converge: %s", liq_lin.message)
        diag = _build_step_diagnostics_fail(...)
        return StepResult(...)
```

**修改后**：
```python
# --- Stage 2 removed: Tl is now solved in Stage 1 (coupled mode) ---
# No separate liquid temperature solve needed
liq_diag: Dict[str, Any] | None = None
liq_lin: Optional[LinearSolveResult] = None
```

**说明**：
- 删除整个 Stage 2 块
- 保留 `liq_diag` 和 `liq_lin` 的初始化（设为 None），避免后续代码报错
- `state_new` 已经包含了 Stage 1 求解的 Tl

### 3.2 更新诊断信息构建

**位置**：第201-220行（`_build_step_diagnostics_success` 调用）

**修改前**：
```python
diag = _build_step_diagnostics_success(
    t_old=t_old,
    t_new=t_new,
    dt=dt,
    linear=lin_result,
    liq_linear=liq_lin,
    diag_sys=diag_sys,
    liq_diag=liq_diag,
    state=state_new,
)
```

**修改后**（保持不变，但确认 `liq_lin=None, liq_diag=None`）：
```python
diag = _build_step_diagnostics_success(
    t_old=t_old,
    t_new=t_new,
    dt=dt,
    linear=lin_result,
    liq_linear=liq_lin,  # None in coupled mode
    diag_sys=diag_sys,
    liq_diag=liq_diag,   # None in coupled mode
    state=state_new,
)
```

**说明**：确保 `_build_step_diagnostics_success` 能正确处理 `liq_lin=None` 的情况。

### 3.3 确保 `_unpack_solution_to_state` 包含 Tl

**位置**：第300-370行（`_unpack_solution_to_state` 函数）

**检查并添加 Tl 部分**（如果没有）：

```python
def _unpack_solution_to_state(
    x: np.ndarray,
    layout: UnknownLayout,
    grid: Grid1D,
    state_ref: State,
    cfg: CaseConfig,
) -> State:
    """Unpack solution vector x into State object."""
    state_new = state_ref.copy()

    # Tg
    if layout.has_block("Tg"):
        for ig in range(grid.Ng):
            idx = layout.idx_Tg(ig)
            state_new.Tg[ig] = float(x[idx])

    # Tl (新增或确认存在)
    if layout.has_block("Tl"):
        for il in range(grid.Nl):
            idx = layout.idx_Tl(il)
            state_new.Tl[il] = float(x[idx])

    # Yg (if present)
    if layout.has_block("Yg"):
        Ns_g_red = layout.n_reduced_gas_species()
        for ig in range(grid.Ng):
            for k in range(Ns_g_red):
                idx = layout.idx_Yg(ig, k)
                state_new.Yg[layout.gas_reduced_to_full_idx[k], ig] = float(x[idx])

    # Ts
    if layout.has_block("Ts"):
        idx = layout.idx_Ts()
        state_new.Ts = float(x[idx])

    # mpp
    if layout.has_block("mpp"):
        idx = layout.idx_mpp()
        state_new.mpp = float(x[idx])

    # Rd
    if layout.has_block("Rd"):
        idx = layout.idx_Rd()
        state_new.Rd = float(x[idx])

    return state_new
```

**说明**：确认 Tl 块存在，如果没有则添加。

---

## 修改4：`physics/interface_bc.py`

### 4.1 确认 `_build_Ts_row` 已支持 Tl 有无的分支

**位置**：第302-313行

**确认代码如下**（应该已经包含，如果没有则添加）：

```python
# Unknown indices near the interface
idx_Tg = layout.idx_Tg(ig_local)

coeff_Tg = -A_if * k_g / dr_g
coeff_Tl = A_if * k_l / dr_l
coeff_Ts = A_if * (k_g / dr_g - k_l / dr_l)
coeff_mpp = -A_if * L_v

# Check if Tl is in layout (coupled) or fixed (explicit)
if layout.has_block("Tl"):
    # Fully coupled: Tl is an unknown in this system
    idx_Tl = layout.idx_Tl(il_local)
    cols = [idx_Tg, idx_Ts, idx_Tl, idx_mpp]
    vals = [coeff_Tg, coeff_Ts, coeff_Tl, coeff_mpp]
    rhs = 0.0
else:
    # Gauss-Seidel split: Tl fixed at old value
    Tl_fixed = float(state.Tl[il_local]) if state.Tl.size > il_local else 0.0
    cols = [idx_Tg, idx_Ts, idx_mpp]
    vals = [coeff_Tg, coeff_Ts, coeff_mpp]
    rhs = -coeff_Tl * Tl_fixed
```

**说明**：
- 在耦合模式下（`solve_Tl=True`），`layout.has_block("Tl")` 为 True
- 此时 Ts 方程包含 Tl 项，实现完全耦合

---

## 修改5：`tests/test_timestepper_one_step.py`

### 5.1 修改测试配置

**位置**：第62-69行

**修改前**：
```python
physics = CasePhysics(
    solve_Tg=True,
    solve_Yg=False,  # Stage 1 only solves Tg, Ts, mpp, Rd (per 12.2)
    solve_Tl=False,  # Tl solved separately in Stage 2 if needed
    solve_Yl=False,
    include_Ts=True,
    include_mpp=True,
    include_Rd=True,
)
```

**修改后**：
```python
physics = CasePhysics(
    solve_Tg=True,
    solve_Yg=False,  # No flux test, Yg not solved
    solve_Tl=True,   # ← 改回 True：Tl coupled in Stage 1
    solve_Yl=False,
    include_Ts=True,
    include_mpp=True,
    include_Rd=True,
)
```

**预期 Layout**：
```
Blocks: {Tg: slice(0,1), Tl: slice(1,2), Ts: slice(2,3), mpp: slice(3,4), Rd: slice(4,5)}
Total DOF: 5
```

---

## 修改总结

| 文件 | 修改内容 | 行数 |
|------|---------|------|
| `build_liquid_T_system_SciPy.py` | 添加 `couple_interface` 参数 + 分支逻辑 | ~20 |
| `build_system_SciPy.py` | 嵌入 Tl 块到全局矩阵 | ~15 |
| `timestepper.py` | 移除 Stage 2 调用 + 确认 unpack Tl | ~10 |
| `interface_bc.py` | 确认 Tl 有无分支（可能已存在） | 0-15 |
| `test_timestepper_one_step.py` | `solve_Tl=False` → `True` | 1 |
| **总计** | | **~60行** |

---

## 实施顺序

建议按以下顺序修改，逐步验证：

### Step 1: 修改 `build_liquid_T_system_SciPy.py`
- 添加 `couple_interface` 参数
- 添加界面边界条件分支
- **验证**：单独测试 `build_liquid_T_system(..., couple_interface=True)` 返回 Nl×Nl 矩阵

### Step 2: 修改 `interface_bc.py`
- 确认 `_build_Ts_row` 支持 Tl 有无分支
- **验证**：单独测试 `build_interface_coeffs` 在 Tl 存在时不报错

### Step 3: 修改 `build_system_SciPy.py`
- 嵌入 Tl 块
- **验证**：检查矩阵结构，确认 Tl 行正确填充

### Step 4: 修改 `timestepper.py`
- 移除 Stage 2
- 确认 unpack 包含 Tl
- **验证**：运行简单算例，检查 state_new.Tl 是否更新

### Step 5: 修改测试配置
- `solve_Tl=True`
- **验证**：运行完整测试

---

## 预期测试结果

修改完成后，运行：
```bash
pytest tests/test_timestepper_one_step.py::test_one_step_no_flux_no_evap_keeps_state_constant -v
```

**预期**：
- ✅ `res.success = True`
- ✅ `linear_converged = True`
- ✅ `matrix_rank = 5`（非奇异）
- ✅ `Tg, Tl, Ts` 保持初值（变化 < 1e-10）
- ✅ `mpp ≈ 0`（< 1e-12）
- ✅ `Rd` 保持初值
- ✅ `energy_balance_if < 1e-8`
- ✅ `mass_balance_rd < 1e-8`

---

## 矩阵结构示意

修改后的全局矩阵结构（5×5）：

```
         Tg   Tl   Ts   mpp  Rd
    Tg [ *    0    *?   *?   0  ]  ← 气相能量 + 对流
    Tl [ 0    *    *    0    0  ]  ← 液相导热（耦合 Ts）
    Ts [ *    *    *    *    0  ]  ← 界面能量跳跃
   mpp [ *    0    0    *    0  ]  ← Stefan 质量平衡
    Rd [ 0    0    0    *    *  ]  ← 半径演化
```

**注意**：
- Tg-Ts, Tg-mpp 的耦合取决于界面能量方程的展开
- Tl-Ts 耦合来自 `_build_Ts_row` 的 `coeff_Tl` 项
- 所有非零元素应该保证矩阵非奇异

---

## 调试检查点

如果修改后测试仍失败，按以下顺序检查：

1. **矩阵奇异性**：
   ```python
   print('Matrix rank:', np.linalg.matrix_rank(A.toarray()))
   print('Expected rank:', layout.n_dof())
   ```

2. **Tl 块填充**：
   ```python
   for il in range(Nl):
       row = layout.idx_Tl(il)
       print(f'Tl[{il}] row {row}:', A[row, :].toarray())
   ```

3. **Ts 行系数**：
   ```python
   idx_Ts = layout.idx_Ts()
   print(f'Ts row {idx_Ts}:', A[idx_Ts, :].toarray())
   ```

4. **解向量**：
   ```python
   print('Solution x:', lin_result.x)
   print('Tl values:', [x[layout.idx_Tl(il)] for il in range(Nl)])
   ```

---

## 注意事项

1. **局部 vs 全局索引**：
   - `build_liquid_T_system` 内部全部用局部索引 `il` (0..Nl-1)
   - 嵌入到全局矩阵时才用 `layout.idx_Tl(il)`

2. **Dirichlet BC 冲突**：
   - 确保 `couple_interface=True` 时不施加 Dirichlet BC
   - 界面能量耦合由 `_build_Ts_row` 提供

3. **`_advance_liquid_T_step12` 函数**：
   - 可以保留函数定义（以备将来测试用）
   - 但不再被 `advance_one_step_scipy` 调用

4. **Stage 2 的其他用途**：
   - 如果项目中其他地方单独使用 `build_liquid_T_system`（例如 `test_step10_scipy_liquid_T.py`）
   - 它们仍然使用 `couple_interface=False`（默认值），行为不变

---

## 成功标志

修改完成并测试通过后，应该看到：

```
tests/test_timestepper_one_step.py::test_one_step_no_flux_no_evap_keeps_state_constant PASSED [100%]
```

并且测试输出中：
- `linear_converged = True`
- `Tg, Tl, Ts` 基本保持初值
- `mpp ≈ 0`, `Rd` 不变
- 能量和质量守恒满足
