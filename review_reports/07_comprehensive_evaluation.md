# 数值框架综合评价报告

## 审查概述

本报告对多组分液滴无化学数值框架进行全面审查，重点检查代码是否符合`Droplet_Transport_Framework_NoChemistry.md`物理框架和`符号约定.md`的设计规范。

**审查范围**: physics → assembly → solvers → properties → core → cases/tests

**审查日期**: 2024

---

## 一、整体评价

### 1.1 框架完成度

| 模块 | 完成度 | 质量评分 |
|------|--------|----------|
| core (types, layout) | 100% | ★★★★★ |
| physics | 95% | ★★★★☆ |
| assembly | 90% | ★★★★☆ |
| solvers | 75% | ★★★☆☆ |
| properties | 100% | ★★★★★ |
| cases/tests | 90% | ★★★★☆ |

**总体评分**: ★★★★☆ (4/5)

### 1.2 符号约定一致性

| 约定类别 | 一致性 | 说明 |
|----------|--------|------|
| 坐标系统 | ✅ 100% | r增加向外, n=+e_r |
| 通量符号 | ✅ 100% | 正向外, mpp>0蒸发 |
| 热通量定义 | ✅ 100% | q = -k∇T |
| 未知量布局 | ✅ 100% | Tg→Yg→Tl→Yl→Ts→mpp→Rd |
| 组分索引 | ✅ 100% | 仅通过layout访问 |
| Closure处理 | ✅ 100% | 不在未知量中 |

---

## 二、已完成功能 (Step 1-17)

### 2.1 核心框架

| Step | 功能 | 状态 | 文件位置 |
|------|------|------|----------|
| 1 | 项目结构 | ✅ | 目录布局 |
| 2 | 类型定义 | ✅ | core/types.py |
| 3 | 未知量布局 | ✅ | core/layout.py |
| 4 | 网格生成 | ✅ | (grid module) |
| 5 | 初始化 | ✅ | physics/initial.py |
| 6 | 气相温度求解 | ✅ | assembly/build_system_SciPy.py |
| 7 | Stefan速度 | ✅ | physics/stefan_velocity.py |
| 8 | 气相对流 | ✅ | physics/flux_convective_gas.py |
| 9 | 气相组分求解 | ✅ | assembly/build_species_system_SciPy.py |
| 10 | 液相温度求解 | ✅ | assembly/build_liquid_T_system_SciPy.py |
| 11 | 界面条件 (Ts/mpp) | ✅ | physics/interface_bc.py |
| 12 | 半径演化 | ✅ | physics/radius_eq.py |
| 13 | 界面平衡 | ✅ | properties/equilibrium.py |
| 14 | 物性计算 | ✅ | properties/*.py |
| 15 | 组分耦合 | ✅ | assembly/*.py |
| 16 | 液相组分 | ✅ | assembly/build_liquid_species_system_SciPy.py |
| 17 | 物性重计算 | ✅ | solvers/timestepper.py |

### 2.2 物理模型实现

| 方程 | 实现状态 | 验证状态 |
|------|----------|----------|
| 气相能量方程 | ✅ 全隐式扩散 + 显式对流 | ✅ 测试通过 |
| 气相组分方程 | ✅ 全隐式扩散 | ✅ 测试通过 |
| 液相能量方程 | ✅ 全隐式扩散 | ✅ 测试通过 |
| 液相组分方程 | ✅ 显式通量 | ⚠️ 部分测试 |
| 界面能量平衡 | ✅ 线性化 | ✅ 测试通过 |
| 界面质量平衡 | ✅ 线性化 | ✅ 测试通过 |
| 半径演化 | ✅ 后向Euler | ✅ 测试通过 |
| Raoult平衡 | ✅ CoolProp/Clausius | ✅ 测试通过 |

---

## 三、未完成功能

### 3.1 Step 18 (跳过)

- **化学反应耦合**: 按计划跳过

### 3.2 Step 19 需要完成

**全隐式非线性牛顿架构**:

| 功能 | 当前状态 | 需要实现 |
|------|----------|----------|
| 非线性残差F(u) | ❌ 未实现 | 高优先级 |
| 雅可比矩阵J=∂F/∂u | ❌ 未实现 | 高优先级 |
| Newton迭代循环 | ❌ 未实现 | 高优先级 |
| 收敛判据 | ❌ 未实现 | 高优先级 |
| 线搜索/阻尼 | ❌ 未实现 | 中优先级 |
| 自适应时间步 | ❌ 未实现 | 低优先级 |

### 3.3 配置更新需求

当前配置 `case_001.yaml`:
```yaml
solve_Yl: false     # 需要改为 true
include_Ts: false   # 需要改为 true
```

---

## 四、代码质量评估

### 4.1 优点

1. **架构清晰**
   - 模块职责分明
   - 依赖关系合理
   - 接口设计良好

2. **类型安全**
   - 广泛使用dataclass
   - 完整的类型注解
   - 运行时验证

3. **符号约定严格执行**
   - 所有模块遵循统一约定
   - 无手工索引计算
   - 完整的文档注释

4. **测试覆盖**
   - 39个测试文件
   - 单元+集成测试
   - Step-by-step验证

5. **物性准确**
   - Cantera气相物性
   - CoolProp液相物性
   - 正确的平衡计算

### 4.2 改进建议

1. **代码组织**
   - `timestepper.py` 过长 (600+行), 建议拆分
   - 考虑将Newton框架放入新模块

2. **性能**
   - 当前使用dense矩阵, 大网格效率低
   - 建议Step 19使用scipy.sparse

3. **文档**
   - 部分函数缺少数学公式注释
   - 建议添加∂F/∂u推导

---

## 五、Step 19 实施建议

### 5.1 全局残差定义

```
F(u) = [F_Tg, F_Yg, F_Tl, F_Yl, F_Ts, F_mpp, F_Rd]^T

其中:
- F_Tg: 气相能量方程残差
- F_Yg: 气相组分方程残差
- F_Tl: 液相能量方程残差
- F_Yl: 液相组分方程残差
- F_Ts: 界面能量平衡残差
- F_mpp: 界面质量平衡残差
- F_Rd: 半径演化方程残差
```

### 5.2 推荐实施步骤

1. **第一阶段**: 残差函数
   ```
   新建 solvers/newton.py
   - compute_residual(state, state_old, props, grid, layout, cfg, dt) -> F
   ```

2. **第二阶段**: 雅可比矩阵
   ```
   - compute_jacobian_numerical(state, ...) -> J (有限差分)
   - 可选: compute_jacobian_analytical(...) -> J (解析)
   ```

3. **第三阶段**: Newton迭代
   ```
   def newton_solve(state_old, props_old, ...):
       u = pack_state(state_old, layout)
       for iter in range(max_iter):
           state = apply_u_to_state(u, layout)
           props = compute_props(state)
           F = compute_residual(state, state_old, props, ...)
           if ||F|| < tol:
               return state, CONVERGED
           J = compute_jacobian(state, ...)
           du = solve(J, -F)
           u = u + alpha * du  # with line search
       return state, FAILED
   ```

4. **第四阶段**: 收敛控制
   ```
   - 相对残差: ||F|| / ||F_0|| < rtol
   - 绝对残差: ||F|| < atol
   - 最大迭代: iter < max_iter
   - 线搜索: Armijo条件
   ```

### 5.3 文件结构建议

```
solvers/
├── scipy_linear.py      # 现有线性求解器
├── timestepper.py       # 现有时间步进
└── newton.py            # 新增Newton框架
    ├── compute_residual()
    ├── compute_jacobian()
    ├── newton_solve()
    └── NewtonResult dataclass
```

---

## 六、风险与注意事项

### 6.1 技术风险

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| 雅可比矩阵不准确 | 收敛失败 | 与数值雅可比对比验证 |
| 界面奇异性 | 迭代发散 | 添加阻尼, 限制mpp范围 |
| 物性强非线性 | 收敛慢 | 准定常物性更新 |
| 大时间步不稳定 | 发散 | 自适应时间步 |

### 6.2 测试策略

1. **单元测试**: 验证残差和雅可比计算正确性
2. **收敛测试**: 验证Newton迭代达到指定容差
3. **稳定性测试**: 验证大时间步不发散
4. **回归测试**: 确保现有测试继续通过

---

## 七、总结

### 7.1 当前状态

多组分液滴无化学数值框架已完成Step 1-17, 具备:
- ✅ 完整的类型系统和未知量布局
- ✅ 正确的物理方程离散
- ✅ 一致的符号约定
- ✅ 准确的物性计算
- ✅ 广泛的测试覆盖

### 7.2 Step 19 准备状态

| 组件 | 可复用性 | 需要改动 |
|------|----------|----------|
| core/types.py | 100% | 无 |
| core/layout.py | 100% | 无 |
| physics/*.py | 95% | 微调界面方程 |
| assembly/*.py | 80% | 重构为残差形式 |
| properties/*.py | 100% | 无 |
| solvers/*.py | 50% | 新增newton.py |

### 7.3 下一步行动

1. **立即**: 创建 `solvers/newton.py` 骨架
2. **短期**: 实现残差计算和数值雅可比
3. **中期**: 实现Newton迭代和收敛控制
4. **验证**: 添加Newton相关测试
5. **优化**: 解析雅可比 (可选)

---

**报告结束**

*本报告由代码审查生成, 作为Step 19全隐式Newton框架构建的基础参考。*
