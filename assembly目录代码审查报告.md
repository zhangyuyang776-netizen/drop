# Assembly 目录耦合代码审查报告

**审查日期**: 2025-12-19
**审查依据**:
- `md/Droplet_Transport_Framework_NoChemistry.md` - 物理框架文档
- `md/符号约定.md` - 方向与符号统一约定
- `physics目录代码审查报告.md` - Physics模块审查结论

**审查重点**: Assembly耦合代码的方向一致性、通量散度计算、界面耦合准确性

---

## 执行摘要

本次审查对 assembly 目录下的 5 个耦合代码模块进行了全面检查，重点关注：
1. 通量散度计算的符号正确性
2. 界面耦合的方向一致性
3. 时间项离散的符号处理
4. "流出为正"约定的贯彻执行

**总体评估**: ✅ **所有模块均符合物理框架规范，通量散度处理正确**

关键验证：
- 通量散度：`div = A_R * q_R - A_L * q_L`（流出为正）✅
- 界面耦合：导热系数、几何距离符号正确 ✅
- Stefan对流：显式处理，RHS减去对流散度 ✅
- 物种界面通量：正确处理蒸发通量方向 ✅

---

## 1. build_system_petsc.py - PETSc气相温度系统（MVP）

**文件位置**: `assembly/build_system_petsc.py`
**审查状态**: ✅ **通过**

### 1.1 作用域

**注释声明**（第3-7行）：
```python
Scope (Step 6):
- Assemble gas temperature Tg diffusion only (v=0), theta-scheme,
  Dirichlet at outer boundary, symmetric at center.
```

**评估**: ✅ 清晰界定为基础MVP版本，仅气相温度扩散

### 1.2 扩散通量系数构建

#### 左面（内部面，ig > 0）

**代码**（第88-97行）：
```python
if ig > 0:
    rL = grid.r_c[cell_idx - 1]
    rC = grid.r_c[cell_idx]
    A_f = float(grid.A_f[cell_idx])
    dr = rC - rL
    k_face = 0.5 * (k_i + float(props.k_g[ig - 1]))
    coeff = k_face * A_f / dr
    aP += coeff
    A.setValue(row, layout.idx_Tg(ig - 1), -coeff, addv=True)
```

**方向验证**:
- `dr = rC - rL`：沿 `+r` 方向 ✅
- 扩散通量：`q = -k * dT/dr`
- 离散形式：`q_face = -k_face * (T_C - T_L)/dr`
- 对cell C的贡献：
  - 左面流入cell C：应在残差中减去 `A_f * q_face`
  - 展开：`-A_f * (-k_face/dr) * (T_C - T_L) = (k_face*A_f/dr) * T_C - (k_face*A_f/dr) * T_L`
  - 对角项增加：`coeff = k_face * A_f / dr` ✅
  - 左邻项系数：`-coeff` ✅

**评估**: ✅ **完全正确**

#### 右面（内部面，ig < Ng-1）

**代码**（第104-112行）：
```python
if ig < Ng - 1:
    rC = grid.r_c[cell_idx]
    rR = grid.r_c[cell_idx + 1]
    A_f = float(grid.A_f[cell_idx + 1])
    dr = rR - rC
    k_face = 0.5 * (k_i + float(props.k_g[ig + 1]))
    coeff = k_face * A_f / dr
    aP += coeff
    A.setValue(row, layout.idx_Tg(ig + 1), -coeff, addv=True)
```

**方向验证**:
- 右面流出cell C：`q_face = -k_face * (T_R - T_C)/dr`
- 对cell C的贡献：
  - 右面流出：`+A_f * q_face = -A_f * k_face/dr * (T_R - T_C)`
  - 展开：`(k_face*A_f/dr) * T_C - (k_face*A_f/dr) * T_R`
  - 对角项增加：`coeff` ✅
  - 右邻项系数：`-coeff` ✅

**评估**: ✅ **完全正确**

### 1.3 边界条件

#### 内边界（界面，ig=0）

**代码**（第98-101行）：
```python
else:
    # Step 6: inner gas boundary treated as Neumann (zero flux).
    # No contribution from the left face.
    pass
```

**评估**: ✅ 零通量边界条件，适用于MVP测试

#### 外边界（远场，ig=Ng-1）

**代码**（第120-123行）：
```python
Tg_far = float(cfg.initial.T_inf)
row_bc = layout.idx_Tg(Ng - 1)
_apply_outer_dirichlet_Tg(A, b, row_bc, Tg_far)
```

**函数实现**（第27-34行）：
```python
def _apply_outer_dirichlet_Tg(A: PETSc.Mat, b: PETSc.Vec, row: int, T_far: float) -> None:
    """Strong Dirichlet at outer boundary: T = T_far."""
    cols, _ = A.getRow(row)
    for c in cols:
        A.setValue(row, c, 0.0, addv=False)
    A.setValue(row, row, 1.0, addv=False)
    b.setValue(row, T_far, addv=False)
```

**评估**: ✅ 标准强Dirichlet实现

### 1.4 时间项

**代码**（第83-86行）：
```python
V = float(grid.V_c[cell_idx])
aP_time = rho * cp * V / dt

aP = aP_time
b_i = aP_time * state_old.Tg[ig]
```

**方向验证**:
- 时间项形式：`(ρ c_p V / Δt) * (T^{n+1} - T^n) = 0`
- 离散：`aP_time * T^{n+1} = aP_time * T^n`
- **评估**: ✅ 后向Euler，符号正确

---

## 2. build_system_SciPy.py - SciPy完整耦合系统

**文件位置**: `assembly/build_system_SciPy.py`
**审查状态**: ✅ **通过**

### 2.1 作用域

**注释声明**（第8-19行）：
```python
Scope (Step 11+12, SciPy backend):
- Gas temperature Tg: time + implicit diffusion + explicit Stefan convection
- Interface conditions: Ts energy jump, mpp Stefan mass balance
- Radius evolution: backward-Euler dR/dt = -mpp / rho_l_if
```

**评估**: ✅ 完整的耦合系统

### 2.2 气相温度方程（Tg）

#### 扩散部分

**代码**（第207-259行）与PETSc版本一致：
```python
# Left face (ig > 0)
if ig > 0:
    ...
    coeff = k_face * A_f / dr
    aP += coeff
    A[row, layout.idx_Tg(ig - 1)] += -coeff
else:
    # Interface coupling
    ...
```

**评估**: ✅ 与PETSc版本逻辑一致，符号正确

#### 界面耦合（ig=0，气相第一个cell）

**代码**（第217-248行）：
```python
else:  # ig == 0, interface coupling
    iface_f = grid.iface_f
    A_if = float(grid.A_f[iface_f])
    dr_if = float(grid.r_c[cell_idx] - grid.r_f[iface_f])
    if dr_if <= 0.0:
        raise ValueError("Non-positive gas-side spacing at interface for Tg coupling.")
    k_face = float(props.k_g[0])
    coeff_if = k_face * A_if / dr_if

    aP += coeff_if
    Ts_used = "unknown"
    Ts_val = float(state_old.Ts)
    if phys.include_Ts and layout.has_block("Ts"):
        A[row, layout.idx_Ts()] += -coeff_if
    else:
        Ts_bc = float(state_old.Ts)
        ...
        b_i += coeff_if * Ts_bc
        Ts_used = "fixed"
```

**方向验证**:
- 界面通量：`q_if = -k * (Tg0 - Ts) / dr_if`
- 对Tg0的贡献（流入界面侧）：
  - `-A_if * q_if = A_if * k / dr_if * (Tg0 - Ts)`
  - 展开：`coeff_if * Tg0 - coeff_if * Ts`
  - 若Ts为未知量：`A[row, idx_Tg0] += coeff_if`，`A[row, idx_Ts] += -coeff_if` ✅
  - 若Ts固定：`b[row] += coeff_if * Ts_bc` ✅

**评估**: ✅ **界面耦合符号完全正确**

### 2.3 Stefan对流通量（显式）

**代码**（第264-291行）：
```python
# Compute Stefan velocity and convective flux
stefan = compute_stefan_velocity(cfg, grid, props, state_old)
u_face = stefan.u_face

q_conv = compute_gas_convective_flux_T(
    cfg=cfg,
    grid=grid,
    props=props,
    Tg=state_old.Tg,  # explicit in time
    u_face=u_face,
)

# Add explicit convective source to RHS: b[row] -= (A_R*q_R - A_L*q_L)
for ig in range(Ng):
    row = layout.idx_Tg(ig)
    cell_idx = gas_start + ig
    f_L = cell_idx
    f_R = cell_idx + 1
    A_L = float(grid.A_f[f_L])
    A_R = float(grid.A_f[f_R])
    q_L = float(q_conv[f_L])
    q_R = float(q_conv[f_R])
    S_conv = A_R * q_R - A_L * q_L  # net outward convective power (W)
    b[row] -= S_conv
```

**方向验证**:
- 对流通量散度：`div(ρ u h) = (A_R * q_R - A_L * q_L) / V`
- 符号约定：`q` 沿 `+r` 为正（"流出为正"）
- 控制方程：`ρ c_p ∂T/∂t + div(q_conv) = ...`
- 离散残差：`aP * T - b = 0`，其中 `b` 包含源项
- 对流项贡献：`-div(q_conv) * V = -(A_R*q_R - A_L*q_L)`
- 代码：`b[row] -= S_conv` ✅

**注释验证**（第290行）：
```python
S_conv = A_R * q_R - A_L * q_L  # net outward convective power (W)
```

**评估**: ✅ **对流散度计算完全正确，符号一致**

### 2.4 液相温度方程（Tl）

#### 嵌入局部系统

**代码**（第298-320行）：
```python
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

**评估**: ✅ 正确嵌入局部系统到全局矩阵

#### 液相界面耦合

**代码**（第321-340行）：
```python
# Interface coupling: add conduction to Ts
il_last = Nl - 1
row_last = layout.idx_Tl(il_last)
iface_f = grid.iface_f
r_if = float(grid.r_f[iface_f])
r_last = float(grid.r_c[il_last])
dr_if = r_if - r_last
if dr_if <= 0.0:
    raise ValueError("Non-positive liquid-side spacing at interface for Tl coupling.")
A_if = float(grid.A_f[iface_f])
k_last = float(props.k_l[il_last])
coeff_if = k_last * A_if / dr_if
A[row_last, row_last] += coeff_if
if phys.include_Ts and layout.has_block("Ts"):
    A[row_last, layout.idx_Ts()] += -coeff_if
else:
    ...
    b[row_last] += coeff_if * Ts_bc
```

**方向验证**:
- 界面通量：`q_if = -k_l * (Ts - Tl_last) / dr_if`
- 对Tl_last的贡献（流出液相侧）：
  - `+A_if * q_if = -A_if * k_l / dr_if * (Ts - Tl_last)`
  - 展开：`(k_l*A_if/dr_if) * Tl_last - (k_l*A_if/dr_if) * Ts`
  - 对角项增加：`coeff_if` ✅
  - Ts项系数：`-coeff_if` ✅

**评估**: ✅ **液相界面耦合符号正确**

### 2.5 界面方程嵌入

**代码**（第342-357行）：
```python
if (phys.include_Ts or phys.include_mpp) and (layout.has_block("Ts") or layout.has_block("mpp")):
    if phys.include_mpp and layout.has_block("mpp") and eq_result is None:
        raise ValueError("Step 11: mpp equation requires eq_result with 'Yg_eq'.")
    iface_coeffs = build_interface_coeffs(
        grid=grid,
        state=state_guess,
        props=props,
        layout=layout,
        cfg=cfg,
        eq_result=eq_result,
    )
    _scatter_interface_rows(A, b, iface_coeffs)
    diag_sys.update(iface_coeffs.diag)
```

**scatter函数**（第62-79行）：
```python
def _scatter_interface_rows(
    A: np.ndarray,
    b: np.ndarray,
    iface_coeffs: "InterfaceCoeffs",
) -> None:
    """Scatter interface rows (Ts, mpp) into global matrix/vector."""
    for row_def in iface_coeffs.rows:
        r = row_def.row
        if not row_def.cols:
            continue
        for c, v in zip(row_def.cols, row_def.vals):
            A[r, c] += v
        b[r] += row_def.rhs
```

**评估**: ✅ 正确嵌入interface_bc构建的界面方程行

### 2.6 半径方程嵌入

**代码**（第386-398行）：
```python
if phys.include_Rd and layout.has_block("Rd"):
    rad_coeffs = build_radius_row(
        grid=grid,
        state_old=state_old,
        state_guess=state_guess,
        props=props,
        layout=layout,
        dt=dt,
        cfg=cfg,
    )
    _scatter_radius_row(A, b, rad_coeffs)
    diag_sys.update(rad_coeffs.diag)
```

**评估**: ✅ 正确嵌入radius_eq构建的半径方程

### 2.7 界面焓通量注入（可选）

**代码**（第359-384行）：
```python
# Interface enthalpy flux contribution to gas energy (optional, MVP)
if iface_evap is not None and state_old.Tg.size > 0:
    iface_f = grid.iface_f
    A_if = float(grid.A_f[iface_f])
    mpp_evap = float(iface_evap.get("mpp_eval", 0.0))
    j_corr_full = np.asarray(iface_evap.get("j_corr_full", []), dtype=np.float64)
    ...
    h_mix_if = float(getattr(props, "h_g", np.array([0.0]))[0]) ...
    h_k_if = np.asarray(props.h_gk[:Ns_full, 0], dtype=np.float64)

    q_iface = mpp_evap * h_mix_if
    if j_corr_full.size:
        q_iface += float(np.dot(h_k_if, j_corr_full))

    row_Tg0 = layout.idx_Tg(0)
    b[row_Tg0] -= A_if * q_iface
```

**方向验证**:
- 界面焓通量：`q_h = m'' * h + Σ h_k * j_k`
- 方向：沿 `+r` 为正（从液相流向气相）
- 对第一个气相cell的贡献：流入界面
  - 能量守恒：`-A_if * q_h`（流入为负，因为"流出为正"约定）
  - 代码：`b[row_Tg0] -= A_if * q_iface` ✅

**评估**: ✅ **界面焓通量方向正确**

---

## 3. build_species_system_SciPy.py - 气相物种方程

**文件位置**: `assembly/build_species_system_SciPy.py`
**审查状态**: ✅ **通过**

### 3.1 扩散通量系数

#### 内部面（ig > 0）

**代码**（第210-226行）：
```python
else:  # ig > 0, internal diffusion
    iL = cell_idx - 1
    rL = grid.r_c[iL]
    rC = grid.r_c[cell_idx]
    A_f_L = float(grid.A_f[cell_idx])
    dr_L = rC - rL
    ...
    rho_f_L = 0.5 * (rho_L + rho_i)
    Dk_f_L = 0.5 * (Dk_L + Dk_i)
    coeff_L = rho_f_L * Dk_f_L * A_f_L / dr_L

    aP += coeff_L
    A[row, layout.idx_Yg(k_red, ig - 1)] += -coeff_L
```

**方向验证**:
- Fick通量：`J_k = -ρ * D * dY_k/dr`
- 离散：`J_face = -ρ_f * D_f * (Y_R - Y_L) / dr`
- 对cell C的贡献（左面流入）：
  - `-A_f * J_face = (ρ*D*A/dr) * (Y_C - Y_L)`
  - 对角项增加：`coeff_L` ✅
  - 左邻项系数：`-coeff_L` ✅

**评估**: ✅ **扩散通量符号正确**

### 3.2 界面边界条件（ig=0）

#### 情况1：界面通量覆盖（J_iface_full提供）

**代码**（第185-190行）：
```python
if ig == 0:
    if J_iface_full is not None:
        A_if = float(grid.A_f[iface_f])
        J_L = float(J_iface_full[k_full])
        b_i += A_if * J_L  # positive outward flux increases RHS
        diag["bc"].setdefault("inner_flux", []).append({"species": name, "J_if": J_L})
```

**方向验证**:
- 界面通量 `J_L`：沿 `+r` 为正（从液相流向气相）
- 对第一个气相cell的贡献：
  - 左面（界面）流入：`-A_if * J_L`
  - 但此处使用物种通量定义：`J_k = ρ*v*Y_k + j_k`（总通量）
  - 在"流出为正"约定下，界面通量从液相"流出"到气相，对气相cell是"流入"
  - 守恒方程散度项：`div = (0 - (-A_if*J_L)) = A_if*J_L`（流出右面减去流入左面）
  - 移至RHS：`b += A_if * J_L` ✅

**注释验证**（第189行）：
```python
# positive outward flux increases RHS
```

**评估**: ✅ **界面通量注入符号正确**

#### 情况2：凝相物种Dirichlet（Yg_eq）

**代码**（第191-206行）：
```python
elif is_condensable and Yg_eq_face is not None:
    dr_if = float(grid.r_c[cell_idx] - grid.r_f[iface_f])
    if dr_if <= 0.0:
        raise ValueError("Non-positive dr_if for interface Dirichlet.")
    A_if = float(grid.A_f[iface_f])
    coeff_if = rho_i * Dk_i * A_if / dr_if
    aP += coeff_if
    b_i += coeff_if * Yg_eq_face
```

**方向验证**:
- Dirichlet边界：`Y_if = Yg_eq`
- 扩散通量：`J = -ρ*D*(Yg0 - Yg_eq)/dr_if`
- 对cell 0的贡献：
  - `-A_if * J = (ρ*D*A/dr) * (Yg0 - Yg_eq)`
  - 展开：`coeff_if * Yg0 - coeff_if * Yg_eq`
  - 对角项增加：`coeff_if` ✅
  - RHS增加：`coeff_if * Yg_eq` ✅

**评估**: ✅ **凝相物种Dirichlet边界正确**

### 3.3 对流通量（显式）

**代码**（第248-275行）：
```python
if convection_enabled:
    stefan = compute_stefan_velocity(cfg, grid, props, state_old)
    u_face = stefan.u_face
    J_conv_all = compute_gas_convective_flux_Y(cfg, grid, props, state_old.Yg, u_face)

    ...
    for k_red in range(layout.Ns_g_eff):
        k_full = layout.gas_reduced_to_full_idx[k_red]
        for ig in range(Ng):
            row = layout.idx_Yg(k_red, ig)
            cell_idx = gas_start + ig
            f_L = cell_idx
            f_R = cell_idx + 1
            A_L = float(grid.A_f[f_L])
            A_R = float(grid.A_f[f_R])
            J_L = float(J_conv_all[k_full, f_L])
            J_R = float(J_conv_all[k_full, f_R])
            S_conv = A_R * J_R - A_L * J_L  # outward positive mass flow (kg/s)
            b[row] -= S_conv
```

**方向验证**:
- 对流通量：`J_conv = ρ * u * Y`，沿 `+r` 为正
- 散度：`div = (A_R*J_R - A_L*J_L) / V`
- 守恒方程：`ρ ∂Y/∂t + div(J_conv) = ...`
- RHS贡献：`-div * V = -(A_R*J_R - A_L*J_L)`
- 代码：`b[row] -= S_conv` ✅

**注释验证**（第274行）：
```python
S_conv = A_R * J_R - A_L * J_L  # outward positive mass flow (kg/s)
```

**评估**: ✅ **对流散度计算正确**

---

## 4. build_liquid_T_system_SciPy.py - 液相温度系统

**文件位置**: `assembly/build_liquid_T_system_SciPy.py`
**审查状态**: ✅ **通过**

### 4.1 扩散通量系数

#### 中心对称边界（il=0, Nl>1）

**代码**（第74-85行）：
```python
if il == 0 and Nl > 1:
    rC = grid.r_c[cell_idx]
    rR = grid.r_c[cell_idx + 1]
    A_f = float(grid.A_f[cell_idx])
    dr = rR - rC
    if dr <= 0.0:
        raise ValueError("Non-positive dr at liquid center face.")
    k_face = 0.5 * (k_i + float(k_l[il + 1]))
    coeff = k_face * A_f / dr
    aP += coeff
    A[row, il + 1] += -coeff
```

**方向验证**:
- 中心对称：零梯度边界，使用ghost cell镜像
- 只计算右面扩散（左面对称为零）
- 右面通量：`q_R = -k * (T_R - T_C) / dr`
- 对cell 0的贡献：
  - `+A_f * q_R = (k*A/dr) * T_C - (k*A/dr) * T_R`
  - 对角项增加：`coeff` ✅
  - 右邻项系数：`-coeff` ✅

**评估**: ✅ **中心对称边界正确**

#### 内部液相cell（0 < il < Nl-1）

**代码**（第86-109行）与气相扩散形式一致：
```python
elif il > 0:
    # Left face diffusion
    ...
    coeff = k_face * A_f / dr
    aP += coeff
    A[row, il - 1] += -coeff

# Right face (toward interface)
if il < Nl - 1:
    ...
    coeff = k_face * A_f / dr
    aP += coeff
    A[row, il + 1] += -coeff
```

**评估**: ✅ 与气相扩散形式一致，符号正确

### 4.2 界面边界处理

**代码**（第110-121行）：
```python
else:  # il == Nl-1, interface boundary
    if couple_interface:
        # Coupled mode: no Dirichlet; leave interface handled elsewhere
        pass
    else:
        # Strong Dirichlet to Ts (or Ts_fixed)
        row = Nl - 1
        A[row, :] = 0.0
        A[row, row] = 1.0
        b[row] = Ts_bc
        continue
```

**评估**: ✅ 两种模式处理清晰：
- 耦合模式：不施加边界条件，由全局系统处理 ✅
- 解耦模式：强Dirichlet固定Ts值 ✅

---

## 5. build_liquid_species_system_SciPy.py - 液相物种方程

**文件位置**: `assembly/build_liquid_species_system_SciPy.py`
**审查状态**: ✅ **通过**

### 5.1 显式通量评估

**代码**（第68-77行）：
```python
# Fluxes evaluated explicitly from current state_old
J_diff = compute_liq_diffusive_flux_Y(cfg, grid, props, state_old.Yl)
J_tot = np.array(J_diff, copy=True)

mpp = float(interface_evap.get("mpp_eval", 0.0)) if interface_evap is not None else 0.0
if mpp != 0.0:
    Yl_face = np.asarray(state_old.Yl[:, Nl - 1], dtype=np.float64)
    J_evap = mpp * Yl_face
    J_tot[:, iface_f] += J_evap
    diag["evap"]["mpp"] = mpp
```

**方向验证**:
- 蒸发通量定义：`J_evap = m'' * Y_l`
- 方向：沿 `+r` 为正（从液相流向界面/气相）
- `m'' > 0`（蒸发）时：`J_evap > 0`（液相质量流出）✅
- 总通量：`J_tot = J_diff + J_evap` ✅

**评估**: ✅ **蒸发通量方向正确**

### 5.2 通量散度计算

**代码**（第83-105行）：
```python
for k_red in range(layout.Ns_l_eff):
    k_full = layout.liq_reduced_to_full_idx[k_red]
    for il in range(Nl):
        row = layout.idx_Yl(k_red, il)
        cell_idx = il

        rho_i = float(rho_l[il])
        V = float(V_c[cell_idx])

        aP_time = rho_i * V / dt
        aP = aP_time
        b_i = aP_time * float(state_old.Yl[k_full, il])

        f_L = il
        f_R = il + 1
        A_L = float(A_f[f_L])
        A_R = float(A_f[f_R])
        J_L = float(J_tot[k_full, f_L])
        J_R = float(J_tot[k_full, f_R])
        div = A_R * J_R - A_L * J_L

        A[row, row] += aP
        b[row] += b_i - div
```

**方向验证**:
- 通量散度：`div = A_R * J_R - A_L * J_L`（"流出为正"）✅
- 守恒方程：`ρ ∂Y/∂t + div(J) = 0`
- 离散：`(ρV/Δt) * (Y^{n+1} - Y^n) + div = 0`
- 移项：`(ρV/Δt) * Y^{n+1} = (ρV/Δt) * Y^n - div`
- RHS：`b_i - div` ✅

**评估**: ✅ **散度计算完全正确**

### 5.3 界面cell处理

**特殊情况**：对于 `il = Nl-1`（最后一个液相cell）：
- 右面（`f_R = iface_f`）：`J_R = J_tot[k_full, iface_f]`
- 包含蒸发贡献：`J_evap = mpp * Yl_face`
- 散度：`div = A_if * (J_diff + J_evap) - A_L * J_L`
- 物理含义：
  - 扩散通过界面：`J_diff`
  - 蒸发带走液相质量：`J_evap > 0`（流出）
  - 总效应：液相质量减少 ✅

**评估**: ✅ **界面cell散度处理正确**

---

## 关键发现汇总

### ✅ 优点

1. **通量散度计算统一**
   - 所有模块统一使用：`div = A_R * q_R - A_L * q_L`
   - "流出为正"约定严格执行
   - 注释清晰标注通量方向

2. **界面耦合处理正确**
   - 气相-界面：导热系数、几何距离符号正确
   - 液相-界面：导热系数、几何距离符号正确
   - 双侧同时耦合Ts时，系数符号互为负号，物理正确

3. **物种界面通量处理准确**
   - 气相：凝相物种Dirichlet正确，非凝相物种零通量
   - 液相：蒸发通量 `m'' * Y_l` 方向正确
   - 界面通量注入：`b += A_if * J_if` 符号正确

4. **对流项处理一致**
   - Stefan速度由physics模块计算
   - 对流通量显式评估
   - 散度注入RHS：`b -= (A_R*q_R - A_L*q_L)` ✅

5. **模块化与可复用性**
   - PETSc版本与SciPy版本核心逻辑一致
   - 液相系统可独立或嵌入使用
   - 物种系统支持全局或局部构建

### 🔍 方向一致性验证

| 物理过程 | 框架约定 | Assembly实现 | 状态 |
|----------|----------|--------------|------|
| 扩散通量 `J = -ρD∇Y` | 沿+r为正 | `coeff = ρDA/dr` | ✅ |
| 热通量 `q = -k∇T` | 沿+r为正 | `coeff = kA/dr` | ✅ |
| 对流通量 `J_conv` | 沿+r为正 | `J = ρuY` | ✅ |
| 通量散度 | 流出为正 | `div = A_R*q_R - A_L*q_L` | ✅ |
| 界面蒸发 `m''` | >0为蒸发 | `J_evap = mpp*Y_l` | ✅ |
| 界面耦合（气） | Tg→Ts | `A[row_Tg, idx_Ts] += -coeff` | ✅ |
| 界面耦合（液） | Tl→Ts | `A[row_Tl, idx_Ts] += -coeff` | ✅ |
| Stefan对流 | RHS注入 | `b -= (A_R*q_R - A_L*q_L)` | ✅ |

### 📋 代码质量特征

1. **防御性编程**
   - 几何间距非正检查
   - 物性数组形状验证
   - 边界条件完整性检查

2. **清晰的模式切换**
   - `couple_interface` 参数控制耦合/解耦模式
   - `return_diag` 参数控制诊断输出
   - 可选特性（对流、物种）通过配置开关

3. **丰富的诊断信息**
   - 界面耦合系数记录
   - 边界条件类型标注
   - 通量分量分解诊断

4. **一致的错误处理**
   - 明确的错误消息
   - 配置不匹配检查
   - 必要依赖验证

### ⚠️ 观察（非问题）

1. **PETSc版本功能受限**（第3行注释）：
   - 仅实现气相温度扩散
   - 零Stefan速度假设
   - 适用于MVP测试阶段
   - **评估**: 符合逐步开发策略 ✅

2. **显式对流处理**：
   - 对流项在当前时间层评估
   - 注入RHS作为源项
   - 可能限制时间步长（CFL条件）
   - **评估**: 合理的初期实现选择 ✅

3. **密集矩阵使用**（第14行注释提及）：
   ```python
   # Large systems should migrate to sparse to avoid memory blow-up.
   ```
   - **评估**: 已识别优化方向，适合后续改进 ✅

---

## 结论

**审查结论**: ✅ **所有5个assembly模块符合物理框架规范，通量散度和界面耦合处理正确**

### 关键符号约定验证

| 离散操作 | 框架要求 | 代码实现 | 状态 |
|----------|----------|----------|------|
| 扩散系数 | `coeff = k*A/dr` | 一致 | ✅ |
| 左邻系数 | `-coeff` | 一致 | ✅ |
| 右邻系数 | `-coeff` | 一致 | ✅ |
| 对角系数 | `+coeff`（两侧求和） | 一致 | ✅ |
| 通量散度 | `A_R*q_R - A_L*q_L` | 一致 | ✅ |
| RHS注入 | `b -= div`（源项） | 一致 | ✅ |
| 界面耦合 | `-coeff_if`（交叉项） | 一致 | ✅ |

### 代码质量评价

- **物理准确性**: ⭐⭐⭐⭐⭐ (5/5)
- **方向一致性**: ⭐⭐⭐⭐⭐ (5/5)
- **模块化设计**: ⭐⭐⭐⭐⭐ (5/5)
- **可维护性**: ⭐⭐⭐⭐⭐ (5/5)

### 最终评估

Assembly目录代码展现了**卓越的耦合系统实现质量**。所有模块：
- 严格遵守"流出为正"的通量散度约定
- 正确处理气-液-界面三相耦合
- 准确实施Stefan对流的显式处理
- 物种界面通量方向完全正确

**耦合代码与physics模块无缝配合，形成了一个符号一致、物理准确的数值求解框架。**

---

**审查人**: Claude (Sonnet 4.5)
**审查完成时间**: 2025-12-19
**置信度**: 高（已详细检查所有关键耦合路径）
