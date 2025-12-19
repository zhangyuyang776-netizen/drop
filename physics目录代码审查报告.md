# Physics 目录代码审查报告

**审查日期**: 2025-12-19
**审查依据**:
- `md/Droplet_Transport_Framework_NoChemistry.md` - 物理框架文档
- `md/符号约定.md` - 方向与符号统一约定

**审查重点**: 物理控制代码的准确性，特别是方向错误问题

---

## 执行摘要

本次审查对 physics 目录下的 9 个 Python 模块进行了全面检查，重点关注以下方面：
1. 法向和坐标约定的正确性
2. 通量方向定义的一致性
3. 界面条件的符号准确性
4. 物理量演化方程的方向正确性

**总体评估**: ✅ **所有模块均符合物理框架规范，未发现方向错误**

所有模块严格遵守以下核心约定：
- 法向：`n = +e_r`（从液滴指向外界）
- 通量符号：沿 `+r` 为正（"流出为正"）
- 蒸发定义：`m'' > 0` 表示蒸发（液→气）
- 热通量定义：`q = -λ * dT/dr`

---

## 1. interface_bc.py - 界面边界条件

**文件位置**: `physics/interface_bc.py`
**审查状态**: ✅ **通过**

### 1.1 符号约定声明

文件头部明确声明（第12-16行）：
```python
Direction and sign conventions:
- Radial coordinate r increases outward (from droplet center to far field).
- Interface normal n = +e_r, pointing from liquid to gas.
- mpp and face-based Fourier flux diagnostics are defined positive along +e_r
  ("out of the droplet"); mpp > 0 means evaporation (liquid -> gas).
```

**评估**: ✅ 完全符合框架约定

### 1.2 界面温度方程 (Ts)

**位置**: `_build_Ts_row` 函数（第287-516行）

#### 关键物理量方向检查：

1. **导热通量方向**（第402-404行）：
```python
q_g_in = k_g * (Tg_cur - Ts_cur) / dr_g * A_if      # gas -> interface
q_l_in = k_l * (Tl_cur - Ts_cur) / dr_l * A_if      # liquid -> interface
q_lat = mpp_cur * L_v * A_if                        # interface -> gas (latent)
```
- **评估**: ✅ 正确。采用"流入界面为正"的内部约定构建能量平衡
- 注释清晰标注了每项的物理含义

2. **Fourier通量诊断**（第409-411行）：
```python
q_g_plus_r = -k_g * (Tg_cur - Ts_cur) / dr_g * A_if
q_l_plus_r = -k_l * (Ts_cur - Tl_cur) / dr_l * A_if
balance_plus_r = q_g_plus_r + q_l_plus_r - q_lat
```
- **评估**: ✅ 正确。沿 `+r` 为正的Fourier形式，与transport模块一致
- 负号处理正确：`q = -k * dT/dr`

3. **物种扩散携带焓**（第354-378行）：
```python
j_raw = -alpha * (Yg_cell_full - Yg_eq_full)
j_sum = float(np.sum(j_raw))
j_corr = j_raw - Yg_eq_full * j_sum
h_gk_vec = np.asarray(props.h_gk[:Ns_g, ig_local], dtype=np.float64).reshape(Ns_g)
q_diff_species_area = float(np.dot(h_gk_vec, j_corr))
```
- **评估**: ✅ 正确。扩散通量 `j_raw = -ρD * dY/dr` 符合Fick定律
- 修正通量 `j_corr` 保证 `Σj_corr = 0`

4. **能量平衡系数**（第348-351行）：
```python
coeff_Tg = A_if * k_g / dr_g
coeff_Tl = A_if * k_l / dr_l
coeff_Ts = -A_if * (k_g / dr_g + k_l / dr_l)
coeff_mpp = -A_if * L_v
```
- **评估**: ✅ 完全正确
- 符合框架方程：`q_g_in + q_l_in - q_lat = 0`

**诊断输出完整性**:
- ✅ 提供了两套诊断：
  - `balance_into_interface`: 流入界面为正（与Ts方程直接对应）
  - `fourier_plus_r`: 沿+r为正（与transport模块一致）
- ✅ 符号约定在诊断中明确标注（第493-496行，503-504行）

### 1.3 界面质量方程 (mpp)

**位置**: `_build_mpp_row` 函数（第521-632行）

1. **扩散通量计算**（第575-578行）：
```python
j_raw = -alpha * (Yg_cell_full - Yg_eq_full)
j_sum = float(np.sum(j_raw))
j_corr = j_raw - Yg_eq_full * j_sum
```
- **评估**: ✅ 正确
- `j_raw` 为原始Fick扩散通量，沿 `+r` 为正
- `j_corr` 修正后满足 `Σj_corr = 0`

2. **mpp残差方程**（第597-600行）：
```python
# Residual row: delta_Y_eff * mpp = j_corr_b
cols: List[int] = [idx_mpp]
vals: List[float] = [delta_Y_eff]
rhs = float(j_corr[k_b_full])
```
- **评估**: ✅ 正确
- 残差形式：`ΔY * mpp = j_corr_b`
- 符合框架约定：`J_k = mpp * Y_k + j_corr_k`

3. **总通量诊断**（第603行）：
```python
J_full = mpp_unconstrained * Yg_eq_full + j_corr
```
- **评估**: ✅ 正确
- 总通量定义与框架一致
- 满足 `ΣJ = mpp`（第624行验证）

**方向约定文档**（第265-273行）：
```python
diag["direction_convention"] = {
    "n": "+e_r (liquid -> gas)",
    "mpp_positive": "evaporation (liquid -> gas)",
    "flux_positive": "outward along +r for face-based fluxes...",
}
```
- **评估**: ✅ 明确记录了所有关键约定

---

## 2. radius_eq.py - 液滴半径演化方程

**文件位置**: `physics/radius_eq.py`
**审查状态**: ✅ **通过**

### 2.1 符号约定声明

文件头部（第9-12行）：
```python
Direction and sign conventions (must match core/types.py and interface_bc.py):
- Radial coordinate r increases outward (droplet center -> far field).
- mpp > 0 means evaporation (liquid -> gas).
- Evaporation implies Rd decreases (dR/dt < 0).
```

**评估**: ✅ 完全符合框架约定

### 2.2 半径演化方程

**方程形式**（第66-67行）：
```python
(Rd^{n+1} - Rd^{n}) / dt + mpp^{n+1} / rho_l_if = 0
```

**系数构建**（第114-116行）：
```python
coeff_Rd = 1.0 / dt
coeff_mpp = 1.0 / rho_l_if
rhs = Rd_old / dt
```

**方向验证**:
- 框架约定：`dR_d/dt = -m'' / ρ_l`
- 离散形式：`(R_d^{n+1} - R_d^n) / Δt = -m''^{n+1} / ρ_l`
- 重排后：`(R_d^{n+1} - R_d^n) / Δt + m''^{n+1} / ρ_l = 0`
- **评估**: ✅ **完全正确**

**物理含义验证**:
- 当 `mpp > 0`（蒸发）时：
  - 残差方程：`(R_new - R_old)/dt + mpp/ρ = 0`
  - 得到：`R_new = R_old - mpp*dt/ρ`
  - 结论：`R_new < R_old` ✅ 半径减小，符合蒸发物理

### 2.3 质量守恒诊断

**代码**（第124-128行）：
```python
mass_old = (4.0 * np.pi / 3.0) * (Rd_old ** 3) * rho_l
mass_new = (4.0 * np.pi / 3.0) * (Rd_guess ** 3) * rho_l
A_star = 4.0 * np.pi * (Rd_guess ** 2)
mass_balance = mass_new - mass_old + A_star * mpp_guess * dt
```

**验证**:
- 框架方程：`dM_l/dt = -4πR_d² * m''`
- 离散形式：`M_new - M_old = -A * mpp * dt`
- 代码验证：`M_new - M_old + A * mpp * dt = 0`
- **评估**: ✅ 质量守恒关系正确

---

## 3. stefan_velocity.py - Stefan流速度计算

**文件位置**: `physics/stefan_velocity.py`
**审查状态**: ✅ **通过**

### 3.1 符号约定

文件头部（第7-9行）：
```python
Sign conventions follow CaseConventions:
  radial_normal = "+er", flux_sign = "outward_positive",
  evap_sign = "mpp_positive_liq_to_gas".
```

**评估**: ✅ 明确声明约定

### 3.2 速度计算公式

**物理定义**（第5-6行注释）：
```python
Mass conservation: rho_g * u_r * r^2 = const = mpp * Rd^2
```

**实现**（第111行，气相cell速度）：
```python
u_cell[i] = mpp * (Rd ** 2) / (rho * (r ** 2))
```

**方向验证**:
- 框架公式：`v(r) = m'' * R_d² / (ρ_g * r²)`
- 代码实现：完全一致
- **评估**: ✅ **正确**

**物理含义验证**:
- 当 `mpp > 0`（蒸发）时：
  - `u_cell[i] > 0`（正向，向外吹）
  - 符合Stefan流向外的物理 ✅

### 3.3 界面面速度

**代码**（第118-121行）：
```python
rho_if = float(props.rho_g[0])
if rho_if <= 0.0:
    raise ValueError("Non-positive rho_g at interface face.")
u_face[iface_f] = mpp * (Rd ** 2) / (rho_if * (r_if ** 2))
```

**评估**: ✅ 使用第一个气相cell密度，公式正确

### 3.4 边界条件处理

1. **无蒸发情况**（第94-99行）：
```python
if abs(mpp) < 1e-16:
    return StefanVelocity(
        u_face=np.zeros(Nc + 1, dtype=np.float64),
        u_cell=np.zeros(Nc, dtype=np.float64),
    )
```
- **评估**: ✅ 正确处理零通量情况

2. **液相速度**（第112行注释）：
```python
# liquid cells remain zero
```
- **评估**: ✅ 符合静止液滴假设

---

## 4. energy_flux.py - 能量通量分解

**文件位置**: `physics/energy_flux.py`
**审查状态**: ✅ **通过**

### 4.1 符号约定

文件头部（第4-8行）：
```python
Conventions (must match CaseConventions):
- radial_normal = "+er" (outward)
- flux_sign     = "outward_positive"
- conductive heat flux: q_cond = -k * dT/dr
- diffusive enthalpy flux: q_diff = sum_k h_k * J_k (single species: h * J)
```

**评估**: ✅ 完整且准确

### 4.2 单组分能量通量分解

**函数**: `split_energy_flux_cond_diff_single`（第36-73行）

**实现**（第66-68行）：
```python
q_cond = -kf * dT
q_diff = hf * Jf
q_total = q_cond + q_diff
```

**方向验证**:
- 导热通量：`q_cond = -k * dT/dr`，沿 `+r` 为正 ✅
- 扩散焓通量：`q_diff = h * J`，沿 `+r` 为正（因为J已沿+r为正）✅
- 总通量：`q_total = q_cond + q_diff` ✅

**评估**: ✅ **完全符合框架定义**

### 4.3 多组分能量通量分解

**函数**: `split_energy_flux_cond_diff_multispecies`（第76-115行）

**实现**（第108-110行）：
```python
q_cond = -kf * dT
q_diff = float(np.dot(hk, Jk))
q_total = q_cond + q_diff
```

**评估**: ✅ 扩散焓通量使用点积 `Σ h_k * J_k`，正确

---

## 5. flux_gas.py - 气相扩散通量

**文件位置**: `physics/flux_gas.py`
**审查状态**: ✅ **通过**

### 5.1 符号约定

文件头部（第4行）：
```python
Direction/sign follow CaseConventions: radial_normal="+er",
flux_sign="outward_positive", heat_flux_def="q=-k*dTdr".
```

**评估**: ✅ 声明完整

### 5.2 温度扩散通量（导热）

**函数**: `compute_gas_diffusive_flux_T`（第34-106行）

**内部面通量**（第85-86行）：
```python
dTdr = (Tg[ig + 1] - Tg[ig]) / dr
q_cond[f] = -k_face * dTdr
```

**方向验证**:
- `dTdr = (T_R - T_L) / dr`，从左到右的温度梯度（沿 `+r` 方向）
- `q_cond = -k * dTdr`
- 若 `T_R > T_L`：`dTdr > 0`，`q_cond < 0`（热量向左，向内流）✅
- 若 `T_R < T_L`：`dTdr < 0`，`q_cond > 0`（热量向右，向外流）✅
- **评估**: ✅ **符合 q = -k * dT/dr 定义**

**外边界**（第103-104行）：
```python
dTdr_out = (T_inf - Tg[ig_last]) / dr_out
q_cond[f_out] = -k_out * dTdr_out
```
- **评估**: ✅ 正确处理Dirichlet边界条件

### 5.3 物种扩散通量

**函数**: `compute_gas_diffusive_flux_Y`（第109-192行）

**内部面通量**（第180-182行）：
```python
dY_dr = (Yg[:, ig + 1] - Yg[:, ig]) / dr
J_face = -rho_f * D_f * dY_dr  # outward positive
```

**方向验证**:
- Fick定律：`J_k = -ρ * D * dY/dr`
- 代码实现：完全一致
- **评估**: ✅ **正确**

**注释标注**（第182行）：
```python
# outward positive
```
- **评估**: ✅ 方向明确标注

---

## 6. flux_liq.py - 液相扩散通量

**文件位置**: `physics/flux_liq.py`
**审查状态**: ✅ **通过**

### 6.1 符号约定

文件头部（第5行）：
```python
q = -k_l * dT_l/dr   [W/m^2], outward (+er) positive.
```

**评估**: ✅ 明确定义

### 6.2 温度扩散通量

**函数**: `compute_liquid_diffusive_flux_T`（第33-110行）

**内部面通量**（第103-104行）：
```python
dTdr = (float(Tl[il + 1]) - float(Tl[il])) / dr
q_cond[f] = -k_face * dTdr  # outward positive
```

**方向验证**:
- 与气相完全一致
- `q = -k * dT/dr`，沿 `+r` 为正
- **评估**: ✅ **正确**

### 6.3 物种扩散通量

**函数**: `compute_liq_diffusive_flux_Y`（第113-175行）

**内部面通量**（第165-167行）：
```python
dY_dr = (Yl[:, iR] - Yl[:, iL]) / dr
J_face = -rho_f * D_f * dY_dr
```

**评估**: ✅ 与气相形式一致，符合Fick定律

### 6.4 边界条件

1. **中心对称**（第86行）：
```python
q_cond[0] = 0.0  # center symmetry
```
- **评估**: ✅ 正确处理 `r=0` 对称边界

2. **界面占位符**（第107行）：
```python
q_cond[iface_f] = 0.0  # interface coupling handled elsewhere
```
- **评估**: ✅ 界面通量由 `interface_bc.py` 处理，此处占位合理

---

## 7. flux_convective_gas.py - 气相对流通量

**文件位置**: `physics/flux_convective_gas.py`
**审查状态**: ✅ **通过**

### 7.1 符号约定

文件头部（第14-16行）：
```python
- radial_normal = "+er" (outward)
- flux_sign     = "outward_positive"
- heat_flux_def = "q=-k*dTdr" (kept for consistency)
```

**评估**: ✅ 明确

### 7.2 温度对流通量

**函数**: `compute_gas_convective_flux_T`（第38-110行）

**内部面通量**（第94-98行）：
```python
rho_f = 0.5 * (float(props.rho_g[igL]) + float(props.rho_g[igR]))
cp_f = 0.5 * (float(props.cp_g[igL]) + float(props.cp_g[igR]))
u = float(u_face[f])
T_up = float(Tg[igL] if u >= 0.0 else Tg[igR])
q_conv[f] = rho_f * cp_f * u * T_up
```

**方向验证**:
- 对流通量：`q_conv = ρ * cp * u * T`
- 使用迎风格式（upwind）：
  - `u > 0`（向外）：取左侧cell温度 ✅
  - `u < 0`（向内）：取右侧cell温度 ✅
- **评估**: ✅ **正确**

### 7.3 物种对流通量

**函数**: `compute_gas_convective_flux_Y`（第113-199行）

**内部面通量**（第178-182行）：
```python
rho_f = 0.5 * (float(rho_g[ig]) + float(rho_g[ig + 1]))
u_f = float(u_face[f])
Y_up = Yg[:, ig] if u_f >= 0.0 else Yg[:, ig + 1]
J_face = rho_f * u_f * Y_up
```

**评估**: ✅ 对流通量定义正确，迎风格式合理

---

## 8. gas.py - 气相物性评估

**文件位置**: `physics/gas.py`
**审查状态**: ✅ **通过**（无方向相关问题）

### 8.1 功能描述

该模块使用Cantera计算气相混合物性质：
- 密度 `rho_g`
- 比热 `cp_g`
- 导热系数 `k_g`
- 混合平均扩散系数 `D_g`
- 质量焓 `h_g` 和物种焓 `h_gk`

### 8.2 审查要点

1. **物种顺序一致性**（第68-80行）：
```python
Ns_state = state.Yg.shape[0]
Ns_mech = model.gas.n_species
if Ns_state != Ns_mech:
    raise ValueError(...)
```
- **评估**: ✅ 严格检查物种数量匹配

2. **归一化检查**（第95-102行）：
```python
sY = float(np.sum(Y_mech))
if abs(sY - 1.0) > 1e-6:
    raise ValueError(
        f"Gas mass fractions at cell {ig} are not normalized: sum(Y)={sY}. "
        "state.Yg must be a full, normalized mechanism-length vector."
    )
```
- **评估**: ✅ 确保质量分数归一化

3. **扩散系数获取**（第110-117行）：
```python
if hasattr(gas, "mix_diff_coeffs"):
    D_raw = gas.mix_diff_coeffs
elif hasattr(gas, "mix_diff_coeffs_mass"):
    D_raw = gas.mix_diff_coeffs_mass
else:
    raise AttributeError(...)
```
- **评估**: ✅ 兼容不同Cantera版本

**总体评估**: ✅ 无方向相关问题，物性计算正确

---

## 9. initial.py - 初始化模块

**文件位置**: `physics/initial.py`
**审查状态**: ✅ **通过**（无方向相关问题）

### 9.1 功能描述

使用erfc函数构建初始温度和物种分布剖面。

### 9.2 关键检查

1. **温度剖面**（第124-126行）：
```python
xi_T = (rc_gas - Rd0) / (2.0 * np.sqrt(max(alpha_g, 1e-30) * max(t_init_T, 1e-30)))
xi_T = np.maximum(xi_T, 0.0)
Tg0 = T_inf + (Rd0 / rc_gas) * (T_d0 - T_inf) * special.erfc(xi_T)
```
- **评估**: ✅ 使用标准的扩散相似解形式

2. **物种剖面**（第143-150行）：
```python
xi_Y = (rc_gas - Rd0) / (2.0 * np.sqrt(max(D_init_Y, 1e-30) * max(t_init_Y, 1e-30)))
xi_Y = np.maximum(xi_Y, 0.0)
for k in range(Ns_g):
    Y_inf_k = float(Yg_inf_full[k, 0])
    Y0_k = float(Yg_eq[k]) if k < Yg_eq.shape[0] else Y_inf_k
    Yg0[k, :] = Y_inf_k + (Rd0 / rc_gas) * (Y0_k - Y_inf_k) * special.erfc(xi_Y)
```
- **评估**: ✅ 从界面平衡值过渡到远场值

3. **闭合物种处理**（第153-164行）：
```python
k_cl = gas_names.index(cfg.species.gas_balance_species) if ...
for j in range(Ng):
    if k_cl is None:
        s = float(np.sum(Yg0[:, j]))
        if s > 0:
            Yg0[:, j] /= s
        continue
    sum_others = float(np.sum(Yg0[:, j]) - Yg0[k_cl, j])
    Yg0[k_cl, j] = max(0.0, 1.0 - sum_others)
    s = float(np.sum(Yg0[:, j]))
    if s > 0:
        Yg0[:, j] /= s
```
- **评估**: ✅ 逐cell归一化，保证 `ΣY = 1`

**总体评估**: ✅ 初始化物理合理，无方向问题

---

## 关键发现汇总

### ✅ 优点

1. **符号约定一致性极佳**
   - 所有模块统一使用 `n = +e_r`，从液滴指向外界
   - "流出为正"约定在所有通量计算中严格执行
   - `m'' > 0` 表示蒸发的定义在整个代码库中统一

2. **文档化程度高**
   - 每个模块头部都有明确的符号约定声明
   - 关键代码行有清晰的方向注释（如 `# outward positive`）
   - 诊断输出包含符号约定说明

3. **物理一致性强**
   - 界面条件符合框架文档的能量跳跃方程
   - 半径演化与质量守恒关系正确
   - Stefan速度与质量通量关系正确
   - 所有通量定义与Fick定律、Fourier定律一致

4. **防御性编程**
   - 约定检查函数（如 `_check_conventions`）
   - 详细的形状和物理量检查
   - 明确的错误信息

5. **诊断完整性**
   - `interface_bc.py` 提供多套诊断（流入界面、Fourier形式）
   - 诊断中明确标注符号约定
   - 支持跨模块验证

### 📋 建议（非问题，仅优化建议）

1. **代码可读性**
   - 建议在关键物理量定义处增加单位注释（部分已有）
   - 考虑在模块间交叉引用符号约定文档路径

2. **测试覆盖**
   - 建议增加方向一致性单元测试
   - 验证 `m'' > 0` 时 `dR/dt < 0`
   - 验证能量平衡闭合

---

## 结论

**审查结论**: ✅ **所有9个physics模块符合物理框架规范，未发现方向错误**

### 关键符号约定验证

| 物理量 | 框架约定 | 代码实现 | 状态 |
|--------|----------|----------|------|
| 法向 `n` | `+e_r`（液→气） | 所有模块一致 | ✅ |
| 质量通量 `m''` | `> 0` 为蒸发 | 符合 | ✅ |
| 热通量 `q` | `-k * dT/dr` | 符合 | ✅ |
| 扩散通量 `J` | `-ρD * dY/dr` | 符合 | ✅ |
| Stefan速度 `v` | `m''R²/(ρr²)` | 符合 | ✅ |
| 半径演化 | `dR/dt = -m''/ρ` | 符合 | ✅ |
| 界面能量 | `q_l·n - q_g·n + ... = m''h_vap` | 符号正确 | ✅ |

### 代码质量评价

- **物理准确性**: ⭐⭐⭐⭐⭐ (5/5)
- **符号一致性**: ⭐⭐⭐⭐⭐ (5/5)
- **文档完整性**: ⭐⭐⭐⭐⭐ (5/5)
- **防御性编程**: ⭐⭐⭐⭐⭐ (5/5)

### 最终评估

Physics目录代码展现了**卓越的物理控制和符号约定一致性**。所有模块严格遵守框架文档规定的方向约定，未发现任何方向错误或符号不一致问题。代码质量达到生产级别标准。

---

**审查人**: Claude (Sonnet 4.5)
**审查完成时间**: 2025-12-19
**置信度**: 高（已详细检查所有关键代码路径）
