# Cases/Tests 模块审查报告

## 审查范围

- `cases/*.yaml` 配置文件
- `tests/test_*.py` 测试文件

---

## 1. 配置文件审查

### 1.1 可用配置

| 文件 | 用途 | 状态 |
|------|------|------|
| case_001.yaml | 完整参考配置 | ✅ 主配置 |
| case_evap_single.yaml | 单组分蒸发 | ✅ |
| runA_evap.yaml | 蒸发运行A | ✅ |
| runB_evap_Rd.yaml | 蒸发+半径 | ✅ |
| runC_evap_Ts.yaml | 蒸发+界面温度 | ✅ |
| runD_evap_multiliq.yaml | 多组分液相 | ✅ |
| step13_A_equilibrium_const_props.yaml | Step13测试A | ✅ |
| step13_B_shrink_const_props.yaml | Step13测试B | ✅ |

### 1.2 case_001.yaml 配置完整性

**约定块**:
```yaml
conventions:
  radial_normal: "+er"
  flux_sign: "outward_positive"
  heat_flux_def: "q=-k*dTdr"
  evap_sign: "mpp_positive_evaporation"
  gas_closure_species: "N2"
  index_source: "unknown_layout_only"
  assembly_pure: true
  grid_state_props_split: true
```

**物理块**:
```yaml
physics:
  model: "droplet_1d_spherical_noChem"
  enable_liquid: true
  include_chemistry: false

  solve_Tg: true
  solve_Yg: true
  solve_Tl: true
  solve_Yl: false

  include_Ts: false   # 当前bc_mode=Ts_fixed
  include_mpp: true
  include_Rd: true
  stefan_velocity: true

  interface:
    type: "no_condensation"
    bc_mode: "Ts_fixed"
    Ts_fixed: 300.0
    equilibrium:
      method: "raoult_psat"
      psat_model: "coolprop"
```

**状态**: ✅ 完整且与代码一致

### 1.3 全局残差未知量配置

根据Step 19需求 (Tg/Tl/Yg/Yl/mpp/Rd/Ts, 无化学):

| 未知量 | 当前配置 | Step 19需求 |
|--------|----------|-------------|
| Tg | solve_Tg: true | ✅ |
| Tl | solve_Tl: true | ✅ |
| Yg | solve_Yg: true | ✅ |
| Yl | solve_Yl: false | ⚠️ 需开启 |
| mpp | include_mpp: true | ✅ |
| Rd | include_Rd: true | ✅ |
| Ts | include_Ts: false | ⚠️ 需开启 |

**Step 19建议配置**:
```yaml
physics:
  solve_Tg: true
  solve_Yg: true
  solve_Tl: true
  solve_Yl: true    # 开启液相组分求解
  include_Ts: true  # 开启界面温度求解
  include_mpp: true
  include_Rd: true

  interface:
    bc_mode: "coupled"  # 或新增模式
```

---

## 2. 测试覆盖审查

### 2.1 测试文件列表

共39个测试文件,覆盖多个模块:

| 类别 | 测试文件数 | 覆盖模块 |
|------|------------|----------|
| 界面/平衡 | 8 | interface_bc, equilibrium |
| 通量计算 | 5 | flux_gas, flux_liq, flux_convective |
| 组分求解 | 6 | species, layout, closure |
| 物性计算 | 4 | gas, liquid, aggregator |
| 时间步进 | 4 | timestepper, scipy_transport |
| 网格/初始化 | 3 | grid, initial |
| 集成测试 | 4 | evap_end_to_end, run_scipy |

### 2.2 关键测试覆盖

| 功能 | 测试文件 | 状态 |
|------|----------|------|
| Layout pack/apply | test_layout_closure_reconstruction_with_inactive_species.py | ✅ |
| 界面能量平衡 | test_interface_balance.py | ✅ |
| 界面质量平衡 | test_interface_multicomponent_mass_balance.py | ✅ |
| Stefan速度 | test_flux_convective_gas.py | ✅ |
| 气相扩散 | test_flux_gas_species_diff.py | ✅ |
| 液相蒸发 | test_liquid_species_evaporation.py | ✅ |
| 半径方程 | test_radius_eq.py | ✅ |
| 平衡计算 | test_equilibrium.py, test_equilibrium_raoult_multicomp.py | ✅ |
| 物性计算 | test_gas_properties.py, test_liquid_properties.py | ✅ |
| 端到端蒸发 | test_evap_end_to_end_smoke.py | ✅ |
| 物性重计算 | test_step17_props_recompute_is_called.py | ✅ |

### 2.3 测试质量评估

**优点**:
- ✅ 覆盖面广
- ✅ 单元测试与集成测试结合
- ✅ 包含物理一致性检查
- ✅ Step-by-step测试 (test_step*)

**不足**:
- ⚠️ 缺少Newton迭代测试 (Step 19需添加)
- ⚠️ 缺少收敛性测试
- ⚠️ 缺少大时间步稳定性测试

---

## 3. Step 19 测试需求

### 3.1 需要新增的测试

1. **Newton迭代收敛测试**
   ```python
   def test_newton_converges_to_tolerance():
       """验证Newton迭代达到指定容差"""
   ```

2. **雅可比矩阵验证测试**
   ```python
   def test_jacobian_vs_finite_difference():
       """验证解析雅可比与数值雅可比一致"""
   ```

3. **全耦合系统测试**
   ```python
   def test_full_coupled_Tg_Tl_Yg_Yl_Ts_mpp_Rd():
       """验证7个未知量块的耦合求解"""
   ```

4. **非线性残差测试**
   ```python
   def test_residual_at_exact_solution_is_zero():
       """验证精确解处残差为零"""
   ```

### 3.2 回归测试保持

现有测试应继续通过:
- test_evap_end_to_end_smoke.py
- test_timestepper_one_step.py
- test_interface_balance.py

---

## 4. 配置建议

### 4.1 Step 19 专用配置

建议创建 `cases/step19_newton_full_coupled.yaml`:

```yaml
case:
  id: step19_newton
  title: "Step 19: Full Newton coupled system"

physics:
  solve_Tg: true
  solve_Yg: true
  solve_Tl: true
  solve_Yl: true
  include_Ts: true
  include_mpp: true
  include_Rd: true

  interface:
    bc_mode: "coupled_newton"

newton:
  max_iter: 10
  rtol: 1e-8
  atol: 1e-12
  line_search: "backtracking"
  damping: 1.0
```

---

## 5. 总结

Cases/Tests模块状态良好:

- ✅ 配置文件完整
- ✅ 测试覆盖广泛
- ✅ 遵循符号约定

Step 19需要:
1. 更新配置开启 solve_Yl 和 include_Ts
2. 添加Newton相关测试
3. 创建全耦合专用配置
