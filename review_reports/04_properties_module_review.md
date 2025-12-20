# Properties 模块审查报告

## 审查范围

- `properties/gas.py`
- `properties/liquid.py`
- `properties/aggregator.py`
- `properties/equilibrium.py`
- `properties/compute_props.py`

---

## 1. 气相物性计算

**位置**: `properties/gas.py`

### 1.1 模型构建

```python
@dataclass
class GasPropertiesModel:
    gas: ct.Solution      # Cantera气相对象
    P_ref: float          # 参考压力 (P_inf)
    gas_names: Tuple[str, ...]  # 组分名列表
    name_to_idx: Dict[str, int]  # 名称→索引映射
```

### 1.2 物性计算

| 物性 | 来源 | 输出形状 |
|------|------|----------|
| rho_g | Cantera density | (Ng,) |
| cp_g | Cantera cp_mass | (Ng,) |
| k_g | Cantera thermal_conductivity | (Ng,) |
| D_g | Cantera mix_diff_coeffs | (Ns_g, Ng) |
| h_g | Cantera enthalpy_mass | (Ng,) |
| h_gk | partial_molar_enthalpies / MW | (Ns_g, Ng) |

### 1.3 验证检查

```python
# 每个单元格都检查:
if rho <= 0: raise ValueError
if cp <= 0: raise ValueError
if k <= 0: raise ValueError
if D < 1e-20: raise ValueError  # 防止扩散系数过小
```

**状态**: ✅ 完整的健壮性检查

---

## 2. 液相物性计算

**位置**: `properties/liquid.py`

### 2.1 模型构建

```python
@dataclass
class LiquidPropertiesModel:
    backend: str              # CoolProp后端 (HEOS)
    fluids: Tuple[str, ...]   # CoolProp流体名
    liq_names: Tuple[str, ...] # 液相组分名
    P_inf: float              # 参考压力
```

### 2.2 物性计算

| 物性 | 来源 | 混合规则 |
|------|------|----------|
| rho_l | CoolProp PropsSI("D") | 调和平均 |
| cp_l | CoolProp PropsSI("Cpmass") | 质量加权 |
| k_l | CoolProp PropsSI("L") | 质量加权 |
| psat_l | CoolProp PropsSI("P", Q=0) | 每组分 |
| hvap_l | H(Q=1) - H(Q=0) | 每组分 |

### 2.3 验证检查

```python
# 检查归一化
if abs(sum(Y) - 1.0) > 1e-6:
    raise ValueError("Liquid mass fractions not normalized")
```

**状态**: ✅ 正确

---

## 3. 物性聚合器

**位置**: `properties/aggregator.py`

### 3.1 主函数

```python
def build_props_from_state(cfg, grid, state, gas_model, liq_model):
    # 1. 计算气相物性
    gas_core, gas_extra = compute_gas_props(gas_model, state, grid)

    # 2. 计算液相物性
    liq_core, liq_extra = compute_liquid_props(liq_model, state, grid)

    # 3. 构建液相扩散系数
    D_l = _build_liq_diffusivity(cfg, liq_names, Nl)

    # 4. 提取界面潜热
    h_vap_if = hvap_l[idx_balance]

    # 5. 验证形状
    props.validate_shapes(grid, Ns_g, Ns_l)

    return props, extras
```

### 3.2 液相扩散系数

```python
# 使用cfg.transport配置
D_default = cfg.transport.D_l_const  # 默认值
D_l_species = cfg.transport.D_l_species  # 组分特定值
```

**状态**: ✅ 支持配置覆盖

---

## 4. 界面平衡模型

**位置**: `properties/equilibrium.py`

### 4.1 平衡模型

```python
@dataclass
class EquilibriumModel:
    method: str = "raoult_psat"   # Raoult定律
    psat_model: str               # coolprop | clausius | auto
    idx_cond_l: np.ndarray       # 可凝结组分液相索引
    idx_cond_g: np.ndarray       # 可凝结组分气相索引
    M_g, M_l: np.ndarray         # 摩尔质量
    Yg_farfield: np.ndarray      # 远场气相组成
    ...
```

### 4.2 平衡计算步骤

```
1. 液相质量分数 → 摩尔分数 X_liq
2. 计算饱和蒸汽压 psat(Ts)
3. Raoult定律: p_k = x_k * psat_k
4. 可凝结组分气相摩尔分数: y_k = p_k / P_g
5. 限制总分压 ≤ 0.995 * P_g (防止过饱和)
6. 非凝结背景气体填充剩余摩尔分数
7. 摩尔分数 → 质量分数 Yg_eq
```

### 4.3 饱和蒸汽压计算

| 方法 | 来源 | 备用 |
|------|------|------|
| coolprop | CoolProp PropsSI | Clausius |
| clausius | 简化Clausius-Clapeyron | - |
| auto | 先尝试CoolProp | Clausius |

**状态**: ✅ 物理模型正确

---

## 5. 物性缓存机制

**位置**: `properties/compute_props.py`

```python
_MODEL_CACHE: Dict[_ModelCacheKey, Tuple[GasModel, LiqModel]]

def get_or_build_models(cfg):
    key = _make_model_cache_key(cfg)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    # 构建并缓存
    gas_model = build_gas_model(cfg)
    liq_model = build_liquid_model(cfg)
    _MODEL_CACHE[key] = (gas_model, liq_model)
    return gas_model, liq_model
```

**优势**: 避免每步重建Cantera/CoolProp对象

**状态**: ✅ 正确实现

---

## 6. 符号约定一致性

### 6.1 潜热符号

| 定义 | 代码实现 | 状态 |
|------|----------|------|
| L_v = h_v - h_l > 0 | hvap = H(Q=1) - H(Q=0) | ✅ |
| 蒸发吸热 | q_lat = mpp * L_v | ✅ |

### 6.2 组分顺序

| 要求 | 实现 | 状态 |
|------|------|------|
| Yg按机理顺序 | Cantera species_names | ✅ |
| Yl按配置顺序 | cfg.species.liq_species | ✅ |

---

## 7. 问题与建议

### 7.1 发现的问题

1. **Clausius-Clapeyron备用**
   - 当前使用固定B=2000常数
   - 建议: 可以从Antoine方程参数计算

2. **混合规则简化**
   - 当前: 简单质量加权
   - 未来: 可考虑非理想混合 (活度系数)

### 7.2 Newton框架相关

对于全隐式Newton:
- 物性依赖于 T, Y
- 需要计算 ∂ρ/∂T, ∂cp/∂T 等导数
- Cantera支持有限差分导数,但可能影响性能

### 7.3 建议

1. **当前可用**: 物性在每步开始时评估 (准定常)
2. **Step 19**: 可以先保持准定常物性,后续迭代收敛后更新
3. **高级**: 实现完整的物性雅可比 (可选)

---

## 8. 总结

Properties模块实现了完整的气液两相物性计算:
- ✅ 基于Cantera/CoolProp的准确物性
- ✅ 正确的界面平衡计算
- ✅ 良好的缓存机制
- ✅ 完整的健壮性检查

Step 19建议:
- 保持现有物性准定常评估策略
- Newton迭代中在外层更新物性
- 收敛后重新计算物性并验证一致性
