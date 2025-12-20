# Assembly 模块审查报告

## 审查范围

- `assembly/build_system_SciPy.py`
- `assembly/build_species_system_SciPy.py`
- `assembly/build_liquid_species_system_SciPy.py`
- `assembly/build_liquid_T_system_SciPy.py`

---

## 1. 系统组装结构审查

### 1.1 全局未知量布局

**要求** (符号约定.md):
- 残差块顺序: Tg → Yg → Tl → Yl → Ts → mpp → Rd

**代码实现** (`core/layout.py`):
```python
# Block order:
# 1) Tg (Ng cells)
# 2) Yg (Ns_g_eff × Ng, reduced species)
# 3) Tl (Nl cells)
# 4) Yl (Ns_l_eff × Nl, reduced species)
# 5) Ts (1 scalar, if include_Ts)
# 6) mpp (1 scalar, if include_mpp)
# 7) Rd (1 scalar, if include_Rd)
```

**状态**: ✅ 完全符合

### 1.2 索引访问

| 索引函数 | 用途 | 状态 |
|----------|------|------|
| `idx_Tg(ig)` | 气温单元ig | ✅ |
| `idx_Yg(k_red, ig)` | 气相组分k在单元ig | ✅ |
| `idx_Tl(il)` | 液温单元il | ✅ |
| `idx_Yl(k_red, il)` | 液相组分k在单元il | ✅ |
| `idx_Ts()` | 界面温度 | ✅ |
| `idx_mpp()` | 蒸发通量 | ✅ |
| `idx_Rd()` | 液滴半径 | ✅ |

---

## 2. 气相温度方程组装

**位置**: `build_system_SciPy.py:build_transport_system()`

### 2.1 时间项

```python
# 全隐式 (theta = 1.0)
aP_time = rho * cp * V / dt
```

**符号约定检查**:
- ✅ 使用单元体积 V_c
- ✅ 时间项加到对角线

### 2.2 扩散项

```python
# 左邻面 (ig-1/2)
k_face = 0.5 * (k_i + k_{i-1})
coeff = k_face * A_f / dr
A[row, idx_Tg(ig-1)] += -coeff  # 邻居
A[row, row] += coeff            # 对角
```

**状态**: ✅ 正确的FVM离散

### 2.3 Stefan对流 (显式)

```python
# 对流通量加到RHS
S_conv = A_R * q_R - A_L * q_L
b[row] -= S_conv
```

**状态**: ✅ 显式处理,符合当前线性化策略

### 2.4 界面耦合

```python
# 气相首个单元与Ts的耦合
if phys.include_Ts and layout.has_block("Ts"):
    A[row, layout.idx_Ts()] += -coeff_if
else:
    # 固定Ts边界
    b_i += coeff_if * Ts_bc
```

**状态**: ✅ 支持耦合/解耦模式

---

## 3. 气相组分方程组装

**位置**: `build_species_system_SciPy.py:build_gas_species_system_global()`

### 3.1 时间项 + 扩散项

```python
aP_time = rho_i * V / dt  # 注意: 组分方程不含cp

# 扩散: rho * D * A / dr
coeff = rho_f * D_f * A_f / dr
```

**状态**: ✅ 正确

### 3.2 界面边界条件

| 组分类型 | 边界条件 | 实现 |
|----------|----------|------|
| 可凝结组分 | Dirichlet: Y = Yg_eq | ✅ |
| 非凝结组分 | Neumann: dY/dr = 0 | ✅ |

### 3.3 外边界条件

```python
# 强Dirichlet
A[row_bc, :] = 0.0
A[row_bc, row_bc] = 1.0
b[row_bc] = Y_far
```

**状态**: ✅ 正确

---

## 4. 液相温度方程组装

**位置**: `build_liquid_T_system_SciPy.py`

### 4.1 中心对称边界

```python
# il=0时的左边界处理
# 使用反射ghost cell实现零梯度
```

**状态**: ✅ 正确

### 4.2 界面耦合

```python
if couple_interface:
    # 耦合模式: 不施加Dirichlet
    pass
else:
    # 强Dirichlet到Ts
    A[row, row] = 1.0
    b[row] = Ts_bc
```

**状态**: ✅ 支持两种模式

---

## 5. 液相组分方程组装

**位置**: `build_liquid_species_system_SciPy.py`

### 5.1 蒸发通量边界

```python
# 界面蒸发通量
mpp = interface_evap.get("mpp_eval", 0.0)
J_evap = mpp * Yl_face  # 蒸发带走的组分
J_tot[:, iface_f] += J_evap
```

**状态**: ✅ 正确的蒸发通量处理

### 5.2 通量散度

```python
div = A_R * J_R - A_L * J_L
b[row] += b_i - div  # 注意负号
```

**状态**: ✅ 符号正确

---

## 6. 界面方程组装

**位置**: `physics/interface_bc.py:build_interface_coeffs()`

### 6.1 Ts能量方程

矩阵行结构:
```
[coeff_Ts, coeff_Tg0, coeff_Tl_last, coeff_mpp] * [Ts, Tg[0], Tl[-1], mpp]^T = rhs
```

### 6.2 mpp质量方程

矩阵行结构:
```
[coeff_mpp, coeff_Yg_cond] * [mpp, Yg_cond[0]]^T = rhs
```

**状态**: ✅ 线性化形式正确

---

## 7. 半径方程组装

**位置**: `physics/radius_eq.py:build_radius_row()`

```python
# 后向Euler: Rd^{n+1} + (dt/rho_l) * mpp^{n+1} = Rd^n
RadiusCoeffs(
    row=layout.idx_Rd(),
    cols=[layout.idx_Rd(), layout.idx_mpp()],
    vals=[1.0, dt / rho_l_if],
    rhs=Rd_old
)
```

**状态**: ✅ 正确

---

## 8. 问题与建议

### 8.1 发现的问题

1. **稀疏矩阵效率** (中等优先级)
   - 当前使用dense numpy数组
   - 对于大网格效率较低
   - 建议: Step 19可考虑迁移到scipy.sparse

2. **对流项处理** (Step 19相关)
   - 当前Stefan对流是显式的
   - 全隐式Newton需要将其隐式化
   - 需要添加对流项的雅可比贡献

3. **界面方程线性化** (Step 19关键)
   - 当前是准线性化 (在state_guess处评估)
   - 全隐式Newton需要完整的雅可比矩阵
   - 需要添加∂F/∂Ts, ∂F/∂mpp等导数

### 8.2 Newton框架准备状态

| 功能 | 当前状态 | Step 19需求 |
|------|----------|-------------|
| 残差计算 | ✅ 可用 | 需要 |
| 雅可比矩阵 | ⚠️ 部分线性化 | 需完善 |
| 收敛判据 | ❌ 未实现 | 需添加 |
| 阻尼策略 | ❌ 未实现 | 需添加 |

---

## 9. 总结

Assembly模块实现了完整的线性系统组装,未知量布局与符号约定一致。主要改进方向:

1. **Step 19优先任务**:
   - 实现完整的雅可比矩阵计算
   - 添加Newton迭代收敛判据
   - 实现线搜索/信赖域阻尼

2. **性能优化** (可选):
   - 迁移到稀疏矩阵格式
   - 缓存重复计算的物性
