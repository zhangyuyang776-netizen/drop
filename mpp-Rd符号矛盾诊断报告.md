# mpp与Rd符号矛盾问题诊断报告

**问题描述**: 运行结果显示mpp为正（应为蒸发），但Rd逐渐增加（与物理矛盾）

**日期**: 2025-12-19
**严重程度**: 🔴 **严重** - 违反基本物理守恒

---

## 问题分析

### 预期物理行为

根据框架约定（`md/符号约定.md` 第227-246行）：

```
dR_d/dt = -m''/ρ_l

若 m'' > 0（蒸发）：
  → dR_d/dt < 0（半径减小）
若 m'' < 0（凝结）：
  → dR_d/dt > 0（半径膨胀）
```

**当前异常**: `mpp > 0` 但 `Rd 增加` ❌

---

## 代码审查结果

### 1. radius_eq.py 方程构建 ✅

**位置**: `physics/radius_eq.py` 第66-67行，114-116行

**代码**:
```python
# 方程形式注释
(Rd^{n+1} - Rd^{n}) / dt + mpp^{n+1} / rho_l_if = 0

# 系数实现
coeff_Rd = 1.0 / dt
coeff_mpp = 1.0 / rho_l_if
rhs = Rd_old / dt
```

**推导验证**:
```
线性系统: (1/dt) * Rd^{n+1} + (1/rho_l) * mpp^{n+1} = Rd^n / dt

解得: Rd^{n+1} = Rd^n - (dt/rho_l) * mpp^{n+1}

若 mpp > 0: Rd^{n+1} = Rd^n - positive_value  →  Rd减小 ✅
若 mpp < 0: Rd^{n+1} = Rd^n - negative_value  →  Rd增大 ✅
```

**结论**: ✅ **radius_eq.py 方程符号完全正确**

---

### 2. assembly 系统耦合 ✅

**位置**: `assembly/build_system_SciPy.py` 第386-398行

**代码**:
```python
if phys.include_Rd and layout.has_block("Rd"):
    rad_coeffs = build_radius_row(...)
    _scatter_radius_row(A, b, rad_coeffs)
```

**scatter函数** (第82-91行):
```python
def _scatter_radius_row(A: np.ndarray, b: np.ndarray, rad_coeffs: "RadiusCoeffs"):
    r = rad_coeffs.row
    for c, v in zip(rad_coeffs.cols, rad_coeffs.vals):
        A[r, c] += v      # 正确累加系数
    b[r] += rad_coeffs.rhs
```

**结论**: ✅ **assembly嵌入正确，无符号错误**

---

### 3. 状态更新 (apply_u_to_state) ✅

**位置**: `core/layout.py` 第537-538行

**代码**:
```python
elif kind == "Rd":
    Rd = val  # 直接使用解向量u[i]的值，无符号变换
```

**结论**: ✅ **状态更新无符号反转**

---

## 🔍 可能的根本原因

### 假设1: 方程未被求解

**检查点**:
```python
# 检查配置是否启用Rd求解
cfg.physics.include_Rd = ?

# 检查layout是否包含Rd块
layout.has_block("Rd") = ?

# 检查线性系统是否包含Rd方程
A.shape = ?  # 应该包含Rd对应的行
```

**诊断方法**:
```python
# 在timestepper.py第181行之后添加诊断
print(f"[DEBUG] Rd方程诊断:")
print(f"  include_Rd: {cfg.physics.include_Rd}")
print(f"  layout.has_block('Rd'): {layout.has_block('Rd')}")
if layout.has_block("Rd"):
    idx_Rd = layout.idx_Rd()
    print(f"  idx_Rd: {idx_Rd}")
    print(f"  A[idx_Rd, :] 非零元素数: {np.count_nonzero(A[idx_Rd, :])}")
    print(f"  b[idx_Rd]: {b[idx_Rd]}")
```

---

### 假设2: mpp方程求解错误（实际为负）

**检查点**:
```python
# 检查mpp方程残差
diag_sys["evaporation"]["mpp_eval"]  # 界面BC计算的mpp
state_new.mpp  # 求解器返回的mpp
```

**可能问题**: mpp方程的残差形式可能有符号错误

**检查**: `physics/interface_bc.py` 第597-600行
```python
# Residual row: delta_Y_eff * mpp = j_corr_b
cols: List[int] = [idx_mpp]
vals: List[float] = [delta_Y_eff]   # ← 检查这里！
rhs = float(j_corr[k_b_full])
```

**物理含义**:
- `j_corr_b`: 凝相物种的修正扩散通量（沿+r为正）
- `delta_Y_eff = Yl_b - Yg_eq_b`: 液相减去气相平衡值
- 蒸发时: `Yl_b > Yg_eq_b` → `delta_Y_eff > 0`
- 蒸发时: `j_corr_b` 应该向外（正）

**诊断方法**:
```python
# 在timestepper.py添加
if "evaporation" in diag_sys:
    evap = diag_sys["evaporation"]
    print(f"[DEBUG] mpp方程诊断:")
    print(f"  delta_Y_eff: {evap['DeltaY_eff']}")
    print(f"  j_corr_b: {evap['j_corr_full'][evap['k_b_full']]}")
    print(f"  mpp_unconstrained: {evap['mpp_unconstrained']}")
    print(f"  mpp_state: {state_new.mpp}")
```

---

### 假设3: 半径方程系数符号错误（最可能）

**⚠️ 关键发现**: 让我重新检查方程推导！

框架文档（`md/Droplet_Transport_Framework_NoChemistry.md` 第238-246行）:
```
dM_l/dt = - 4 π R_d^2 m''

d/dt [ (4/3) π R_d^3 ρ̄_l ] = - 4 π R_d^2 m''

展开（假设ρ_l常数）:
(4/3) π * 3 * R_d^2 * dR_d/dt * ρ̄_l = - 4 π R_d^2 m''

简化:
4 π R_d^2 ρ̄_l * dR_d/dt = - 4 π R_d^2 m''

消去 4πR_d^2:
ρ̄_l * dR_d/dt = - m''

因此:
dR_d/dt = - m'' / ρ̄_l
```

**后向Euler离散**:
```
(Rd^{n+1} - Rd^n) / Δt = - m''^{n+1} / ρ_l

整理:
(Rd^{n+1} - Rd^n) / Δt + m''^{n+1} / ρ_l = 0  ✅
```

**等等！让我检查radius_eq.py中是否使用了正确的符号...**

查看 `physics/radius_eq.py`:
- 第66行注释：`(Rd^{n+1} - Rd^{n}) / dt + mpp^{n+1} / rho_l_if = 0`
- 第115行代码：`coeff_mpp = 1.0 / rho_l_if`

这看起来是正确的！

---

## 🎯 诊断步骤（按优先级）

### 步骤1: 检查Rd是否被求解

在运行代码中添加以下诊断（建议在 `solvers/timestepper.py` 第180行之后）:

```python
# === 添加诊断代码 ===
print("="*60)
print("[DIAG] 半径方程诊断")
print("="*60)
print(f"cfg.physics.include_Rd: {cfg.physics.include_Rd}")
print(f"layout.has_block('Rd'): {layout.has_block('Rd')}")

if layout.has_block("Rd"):
    idx_Rd = layout.idx_Rd()
    idx_mpp = layout.idx_mpp() if layout.has_block("mpp") else None

    print(f"\n[矩阵结构]")
    print(f"  系统大小: {A.shape}")
    print(f"  Rd方程行号: {idx_Rd}")
    print(f"  mpp未知量索引: {idx_mpp}")

    print(f"\n[Rd方程系数]")
    row_Rd = A[idx_Rd, :]
    nonzero = np.nonzero(row_Rd)[0]
    print(f"  非零系数位置: {nonzero}")
    print(f"  非零系数值: {row_Rd[nonzero]}")
    print(f"  RHS值: {b[idx_Rd]}")

    if idx_mpp is not None:
        print(f"\n[Rd-mpp耦合]")
        print(f"  A[Rd, Rd] = {A[idx_Rd, idx_Rd]}")
        print(f"  A[Rd, mpp] = {A[idx_Rd, idx_mpp]}")
        print(f"  预期: A[Rd,Rd]=(1/dt), A[Rd,mpp]=(1/rho_l)")

    print(f"\n[物性参数]")
    if "radius_eq" in diag_sys:
        rd_diag = diag_sys["radius_eq"]
        print(f"  dt = {rd_diag.get('dt')}")
        print(f"  rho_l_if = {rd_diag.get('rho_l_if')}")
        print(f"  Rd_old = {rd_diag.get('Rd_old')}")
        print(f"  Rd_guess = {rd_diag.get('Rd_guess')}")
        print(f"  mpp_guess = {rd_diag.get('mpp_guess')}")

print(f"\n[求解前]")
print(f"  Rd_old = {state_old.Rd}")
print(f"  mpp_old = {state_old.mpp}")

# 继续原有的solve代码...
lin_result: LinearSolveResult = solve_linear_system_scipy(...)

print(f"\n[求解后]")
print(f"  Rd_new = {state_new.Rd}")
print(f"  mpp_new = {state_new.mpp}")
print(f"  ΔRd = {state_new.Rd - state_old.Rd}")
print(f"  预期ΔRd = {-(dt / rd_diag.get('rho_l_if', 1.0)) * state_new.mpp if 'radius_eq' in diag_sys else 'N/A'}")
print("="*60)
```

---

### 步骤2: 检查求解器是否正常工作

在 `solvers/scipy_linear.py` 中（如果可访问）添加残差检查:

```python
# 求解后验证
residual = A @ x - b
print(f"[求解器诊断]")
print(f"  ||Ax - b|| = {np.linalg.norm(residual)}")
print(f"  max|Ax - b| = {np.max(np.abs(residual))}")

# 特别检查Rd方程残差
if layout.has_block("Rd"):
    idx_Rd = layout.idx_Rd()
    print(f"  Rd方程残差 = {residual[idx_Rd]}")
```

---

### 步骤3: 手动计算验证

添加手动计算来验证符号:

```python
# 在state_new获得之后
if layout.has_block("Rd") and layout.has_block("mpp"):
    Rd_old_val = float(state_old.Rd)
    Rd_new_val = float(state_new.Rd)
    mpp_new_val = float(state_new.mpp)
    rho_l_val = float(props.rho_l[-1])  # 界面液相密度
    dt_val = float(dt)

    # 手动计算预期的Rd
    Rd_expected = Rd_old_val - (dt_val / rho_l_val) * mpp_new_val

    print(f"[手动验证]")
    print(f"  Rd_old = {Rd_old_val:.6e}")
    print(f"  mpp_new = {mpp_new_val:.6e}")
    print(f"  dt = {dt_val:.6e}")
    print(f"  rho_l_if = {rho_l_val:.6e}")
    print(f"  Rd_expected = Rd_old - (dt/rho_l)*mpp = {Rd_expected:.6e}")
    print(f"  Rd_actual = {Rd_new_val:.6e}")
    print(f"  差异 = {Rd_new_val - Rd_expected:.6e}")

    if mpp_new_val > 0:
        if Rd_new_val > Rd_old_val:
            print(f"  ❌ 错误！mpp>0（蒸发）但Rd增加")
        else:
            print(f"  ✅ 正确：mpp>0（蒸发），Rd减小")
```

---

## 🔧 可能的修复方案

### 如果问题是"Rd方程未被求解"

**修复**: 确保配置文件中启用Rd:
```python
cfg.physics.include_Rd = True
```

---

### 如果问题是"符号确实错误"

**🚨 警告**: 根据代码审查，符号应该是正确的！但如果诊断确认符号错误，需要：

**错误修复方案A** - 如果radius_eq.py的coeff_mpp符号错了:
```python
# physics/radius_eq.py 第115行
# 错误（如果是这样）:
coeff_mpp = 1.0 / rho_l_if

# 修正为:
coeff_mpp = -1.0 / rho_l_if  # 注意负号！
```

**但根据审查，这应该不是问题！当前的`+1.0/rho_l_if`是正确的。**

---

### 如果问题是"rho_l为负"

**检查**: `props.rho_l` 的符号
```python
print(f"[密度检查]")
print(f"  rho_l_min = {np.min(props.rho_l)}")
print(f"  rho_l_max = {np.max(props.rho_l)}")
if np.any(props.rho_l <= 0):
    print(f"  ❌ 错误：液相密度非正！")
```

---

## 📋 推荐行动

1. **立即**: 运行步骤1的诊断代码，检查Rd方程是否被包含在系统中
2. **如果Rd方程存在**: 运行步骤3的手动计算验证
3. **检查输出文件**: 确认诊断输出中的mpp符号是否与内部状态一致
4. **提供诊断输出**: 将打印的诊断信息发送给我进一步分析

---

## 📖 参考文档

- **物理框架**: `md/Droplet_Transport_Framework_NoChemistry.md` 第227-253行
- **符号约定**: `md/符号约定.md` 第217-254行
- **Radius方程**: `physics/radius_eq.py` 第1-161行
- **Assembly耦合**: `assembly/build_system_SciPy.py` 第386-398行

---

**诊断报告生成时间**: 2025-12-19
**需要用户提供**: 运行诊断代码后的输出结果
