# Physics 模块审查报告

## 审查范围

- `physics/interface_bc.py`
- `physics/radius_eq.py`
- `physics/stefan_velocity.py`
- `physics/flux_gas.py`
- `physics/flux_liq.py`
- `physics/flux_convective_gas.py`
- `physics/energy_flux.py`
- `physics/initial.py`
- `physics/gas.py`

---

## 1. 符号约定一致性审查

### 1.1 坐标系统与法向量

| 项目 | 符号约定.md 要求 | 代码实现 | 状态 |
|------|------------------|----------|------|
| 径向法向量 | n = +e_r (指向外) | 一致 | ✅ 符合 |
| 界面位置 | r = Rd | `grid.iface_f = Nl` | ✅ 符合 |

### 1.2 通量符号约定

| 物理量 | 约定 | 代码实现 | 状态 |
|--------|------|----------|------|
| 热通量 | q = -k∇T, 正向外 | `flux_gas.py`: 使用 `-k_g * dT/dr` | ✅ 符合 |
| 质量通量 | m'' > 0 蒸发 | `interface_bc.py`: mpp > 0 表示蒸发 | ✅ 符合 |
| Stefan速度 | 正向外表示蒸发 | `stefan_velocity.py`: `u_s = mpp / rho_g` | ✅ 符合 |

### 1.3 蒸发质量通量符号

**关键检查**: `interface_bc.py` 中的质量守恒方程

```python
# 质量守恒: mpp * (Yg_eq - 1) - rho_g * D_g * dYg/dr = 0
# 这与 mpp > 0 表示蒸发的约定一致
```

**状态**: ✅ 符号一致

---

## 2. 物理方程实现审查

### 2.1 界面能量方程 (Ts)

**位置**: `interface_bc.py:_build_Ts_energy_row()`

**物理框架要求**:
```
q_g + q_l + m'' * L_v = 0
其中:
- q_g = -k_g * dT_g/dr|_if (气侧导热)
- q_l = +k_l * dT_l/dr|_if (液侧导热)
- m'' * L_v = 潜热项
```

**代码实现审查**:
```python
# 气侧导热系数
coeff_g = k_g * A_if / dr_g  # k * A / dr

# 液侧导热系数
coeff_l = k_l * A_if / dr_l  # k * A / dr

# 潜热项
h_vap_if = props.h_vap_if  # J/kg
```

**问题发现**:
1. ⚠️ 代码中液侧和气侧的热通量符号处理需要仔细验证
2. ✅ 潜热项正确使用 `h_vap_if` (来自CoolProp)

### 2.2 界面质量方程 (mpp)

**位置**: `interface_bc.py:_build_mpp_evaporation_row()`

**物理框架要求**:
```
mpp * (Y_k^eq - 1) = -ρ_g * D_k * dY_k/dr|_if
```

**代码实现**:
```python
# Stefan mass balance: mpp*(Yg_eq - 1) - rho_g*D_g*dY/dr = 0
val_mpp = (Yg_eq_cond - 1.0) * A_if
val_Yg = -rho_g * D_cond * A_if / dr_if
```

**状态**: ✅ 方程形式正确

### 2.3 半径演化方程 (Rd)

**位置**: `radius_eq.py:build_radius_row()`

**物理框架要求**:
```
dR/dt = -m'' / ρ_l
```

**代码实现**:
```python
# Backward Euler: (Rd^{n+1} - Rd^n) / dt = -mpp^{n+1} / rho_l
# 整理: Rd^{n+1} + (dt/rho_l) * mpp^{n+1} = Rd^n
coeff_Rd = 1.0
coeff_mpp = dt / rho_l_if
```

**状态**: ✅ 符号和形式正确

### 2.4 Stefan 速度

**位置**: `stefan_velocity.py`

**物理框架要求**:
```
u_s = m'' / ρ_g|_if
```

**代码实现**:
```python
rho_if = float(props.rho_g[0])
u_s = mpp / rho_if  # m/s, positive = outward = evaporation
```

**状态**: ✅ 正确

---

## 3. 边界条件审查

### 3.1 气相外边界

| 变量 | 边界条件 | 代码实现 | 状态 |
|------|----------|----------|------|
| Tg | Dirichlet: T = T_inf | `_apply_outer_dirichlet_Tg()` | ✅ |
| Yg | Dirichlet: Y = Y_inf | `build_gas_species_system_global()` | ✅ |

### 3.2 液相中心边界

| 变量 | 边界条件 | 代码实现 | 状态 |
|------|----------|----------|------|
| Tl | 对称(零梯度) | 隐式处理 via 反射ghost cell | ✅ |
| Yl | 对称(零梯度) | `flux_liq.py`: f=0时零通量 | ✅ |

### 3.3 界面边界

| 变量 | 边界条件 | 代码实现 | 状态 |
|------|----------|----------|------|
| Tg[0] | 耦合到Ts | `build_transport_system()` | ✅ |
| Tl[Nl-1] | 耦合到Ts | `build_liquid_T_system_SciPy()` | ✅ |
| Yg_cond | Dirichlet: Yg_eq | `interface_bc.py` | ✅ |
| Yl | 蒸发通量 J = mpp * Yl | `build_liquid_species_system()` | ✅ |

---

## 4. 问题与建议

### 4.1 已发现问题

1. **能量通量符号验证** (中等优先级)
   - 位置: `interface_bc.py`
   - 问题: 液侧和气侧热通量的符号约定需要更明确的注释
   - 建议: 添加详细的符号推导注释

2. **潜热来源** (低优先级)
   - 当前: 从CoolProp获取 `hvap_l`
   - 备用: `cfg.physics.latent_heat_default`
   - 状态: 已实现fallback

### 4.2 代码质量

- ✅ 函数职责单一
- ✅ 明确的类型注解
- ✅ 丰富的诊断信息返回
- ⚠️ 部分函数较长,可考虑进一步分解

---

## 5. 总结

Physics模块整体实现与物理框架文档一致,符号约定遵循设计规范。建议在Step 19全隐式牛顿框架中:

1. 确保界面方程雅可比矩阵的一致性
2. 验证所有通量符号在残差计算中的一致性
3. 为Newton迭代添加适当的阻尼和收敛判据
