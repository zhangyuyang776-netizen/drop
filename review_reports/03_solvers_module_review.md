# Solvers 模块审查报告

## 审查范围

- `solvers/scipy_linear.py`
- `solvers/timestepper.py`

---

## 1. 线性求解器审查

**位置**: `solvers/scipy_linear.py`

### 1.1 求解器接口

```python
def solve_linear_system_scipy(
    A,                    # 系数矩阵
    b: np.ndarray,        # 右端项
    cfg: CaseConfig,      # 配置
    x0: np.ndarray = None,  # 初始猜测(直接法忽略)
    method: str = "direct"  # 求解方法
) -> LinearSolveResult
```

### 1.2 支持的方法

| 方法 | 实现状态 | 备注 |
|------|----------|------|
| direct (spsolve) | ✅ 已实现 | 默认方法 |
| gmres | ❌ 未实现 | 抛出NotImplementedError |
| cg | ❌ 未实现 | 抛出NotImplementedError |

### 1.3 收敛判据

```python
rtol = cfg.petsc.rtol  # 默认 1e-8
atol = cfg.petsc.atol  # 默认 1e-12

r = b - A @ x
res_norm = ||r||_2
rel_res = res_norm / (||b||_2 + 1e-30)
converged = res_norm <= max(atol, rtol * ||b||_2)
```

**状态**: ✅ 正确

### 1.4 返回结构

```python
@dataclass
class LinearSolveResult:
    x: np.ndarray          # 解向量
    converged: bool        # 收敛标志
    n_iter: int           # 迭代次数(直接法为1)
    residual_norm: float  # 绝对残差
    rel_residual: float   # 相对残差
    method: str           # 使用的方法
    message: str = None   # 错误/警告消息
```

**状态**: ✅ 信息完整

---

## 2. Timestepper 审查

**位置**: `solvers/timestepper.py`

### 2.1 单步推进函数

```python
def advance_one_step_scipy(
    cfg: CaseConfig,
    grid: Grid1D,
    layout: UnknownLayout,
    state: State,
    props: Props,
    t: float,
) -> StepResult
```

### 2.2 执行流程

```
1. state_old = state.copy()
2. state_guess = state_old (MVP初始猜测)
3. A, b, diag = _assemble_transport_system_step12(...)
4. u0 = pack_state(state_old, layout)
5. lin_result = solve_linear_system_scipy(A, b, cfg, u0)
6. state_new = apply_u_to_state(state_old, lin_result.x, layout)
7. _postprocess_species_bounds(cfg, layout, state_new)
8. props_new = compute_props(cfg, grid, state_new)
9. sanity_check(...)
10. return StepResult(state_new, props_new, diag, success)
```

### 2.3 后处理

```python
def _postprocess_species_bounds(cfg, layout, state):
    # 1. 可选clamp负值到min_Y
    # 2. 重建closure species: Y_closure = 1 - sum(Y_solved)
    # 3. 最终clip到[0, 1]
```

**状态**: ✅ 实现正确

### 2.4 健壮性检查

```python
def _sanity_check_step(cfg, grid, state_new, props_new, diag):
    # 检查项目:
    # - Rd > 0 且有限
    # - Ts 有限
    # - mpp 有限
    # - Tg/Tl 有限
    # - 温度边界 T_min < T < T_max
    # - sum(Y) = 1 (在容差内)
    # - 线性求解收敛
    # - 时间一致性
```

**状态**: ✅ 全面的检查

---

## 3. 问题与建议

### 3.1 当前timestepper的限制

1. **单步线性求解**
   - 当前: 每步只做一次线性求解
   - Newton需求: 需要迭代直到残差收敛

2. **无Newton迭代**
   - 当前: `state_guess = state_old` (无更新)
   - Newton需求: 迭代更新 `state_guess`

3. **残差评估**
   - 当前: 只计算线性系统残差
   - Newton需求: 计算非线性残差 F(u)

### 3.2 Step 19 Newton框架需求

```
Newton迭代伪代码:
u = u_old
for iter in range(max_newton_iter):
    F = compute_residual(u)        # 非线性残差
    J = compute_jacobian(u)        # 雅可比矩阵

    if ||F|| < tol:
        break

    du = solve(J, -F)              # Newton步
    u = u + alpha * du             # 阻尼更新
```

### 3.3 需要新增的组件

| 组件 | 描述 | 优先级 |
|------|------|--------|
| `compute_residual()` | 计算非线性残差F(u) | 高 |
| `compute_jacobian()` | 计算雅可比J = ∂F/∂u | 高 |
| Newton迭代循环 | 外层迭代管理 | 高 |
| 收敛判据 | ||F|| < tol | 高 |
| 线搜索/阻尼 | 保证下降 | 中 |
| 自适应时间步 | 失败重试 | 低 |

---

## 4. 代码质量评估

### 4.1 优点

- ✅ 清晰的模块分离
- ✅ 完整的诊断信息
- ✅ 良好的错误处理
- ✅ 类型注解完整

### 4.2 改进建议

- ⚠️ timestepper.py过长(600+行),建议拆分
- ⚠️ 缺少Newton迭代的基础设施
- 💡 考虑将线性/非线性求解抽象为Strategy模式

---

## 5. 与Newton框架的兼容性

| 现有组件 | 可复用性 | 改动需求 |
|----------|----------|----------|
| LinearSolveResult | ✅ 完全复用 | 无 |
| solve_linear_system_scipy | ✅ 复用 | 无 |
| StepDiagnostics | ⚠️ 需扩展 | 添加newton_iter字段 |
| advance_one_step_scipy | ⚠️ 需重构 | 添加Newton循环 |
| pack_state / apply_u_to_state | ✅ 完全复用 | 无 |

---

## 6. 总结

Solvers模块当前实现了线性求解和单步时间推进,代码质量良好。
Step 19需要在此基础上:

1. 添加非线性残差计算函数
2. 添加雅可比矩阵计算(解析或数值)
3. 实现Newton迭代外循环
4. 添加收敛判据和阻尼策略

建议创建新模块 `solvers/newton.py` 来实现Newton框架,保持现有代码不变。
