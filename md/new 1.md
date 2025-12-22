简单讲：你现在的框架已经是“半隐式 + 线性化”，我们要做的是在不自虐的前提下，把它升级成**真正的全隐式非线性 Newton**，但**Newton 主循环和线性求解交给 SciPy / PETSc**，自己只负责物理 + 残差包装。

下面是一个“可以直接照着干”的**Step 19 完整工作路线**，已经把“把显式都改成隐式”这个目标塞进去了。

---

## Step 19：全隐式非线性 Newton 架构（无化学）

### 19.0 收尾 Step17，冻结基线

**目标**：留住一个“还能跑的线性半隐式版本”作为对照基线。

* 保留当前：

  * `build_system_SciPy.py`
  * `build_species_system_SciPy.py`
  * `build_liquid_T_system_SciPy.py`
  * `build_liquid_species_system_SciPy.py`
  * `timestepper.py` 中的线性推进流程
* 确认当前 2.2 表里的状态对应代码逻辑：

  * 气相能量：扩散隐式，对流显式（用 `state_old`）
  * 气相组分：扩散隐式，对流可选显式
  * 液相能量：扩散隐式
  * 液相组分：通量显式（用 lagged Yl）
  * 界面 Ts/m''：线性化塞进 A,b
  * Rd：后向 Euler（但目前在“单独模块+线性化”里）

> 这一阶段只做确认，不改东西，保证你随时能退回“老版本线性框架”。

---

## 19.1 搭建全局未知向量 & Nonlinear Context（不改物理）

**目标**：把所有未知量统一打包成 `u`，为 F(u) 做准备。

1. **扩展 UnknownLayout / layout.py**

   * 确认当前 layout 已经包含：

     * `Tg[Ng], Yg[Nspec_red, Ng], Tl[Nl], Yl[Nspec_red, Nl], Ts, mpp, Rd`
   * 若有“按阶段开关”的逻辑（比如 include_Rd / include_mpp），保留，只是 layout 上要有统一接口：
     `pack_state(state, layout) -> u`, `apply_u_to_state(state, u, layout) -> state_new`

2. **新增 NonlinearContext（比如 solvers/nonlinear_context.py）**

   * 内容建议：

     ```python
     @dataclass
     class NonlinearContext:
         cfg: CaseConfig
         layout: UnknownLayout
         grid_ref: Grid1D      # t^n 的参考网格
         state_old: State      # t^n 的状态
         gas_model: GasPropertiesModel
         liq_model: LiquidPropertiesModel
         eq_model: Optional[EquilibriumModel]
         dt: float
     ```

> 到这里还没开始“全隐式化”，只是为 F(u) 铺路。

---

## 19.2 定义全局残差 F(u)（第一版：结构打通，离散仍是现在的“半隐式”）

**目标**：在**不动现有半隐式离散的前提下**，先搭起 F(u) → SciPy Newton 的框架，保证流程通了再谈“全隐式”。

1. 新建 `assembly/residual_global.py`：

   ```python
   def build_global_residual(u: np.ndarray, ctx: NonlinearContext) -> tuple[np.ndarray, dict]:
       """
       用当前的离散方式（扩散隐式 + 对流/液相通量显式）构造 F(u)。
       返回 (residual_vector, diag_info)
       """
   ```

2. 内部流程（**第一版**）：

   1. `state_guess = apply_u_to_state(state_template, u, layout, closure=True)`
   2. 网格：**第一版先用固定网格**，不引入动网格非线性

      * 即：`grid = ctx.grid_ref`，先不随 Rd 变。
   3. 物性：沿用当前 `compute_props(cfg, grid, state_guess)`，但可以先只用 `state_old` 的温度做个对比，确保接口通。
   4. 界面平衡：保持**现在的线性化接口**，不要改：

      * 调用你现有的 `equilibrium` / `interface_bc`，用 `state_guess / state_old` 的组合，输出线性化系数，交给 `build_system_SciPy`。
   5. 线性系统装配：

      ```python
      A, b, diag = build_transport_system(
          cfg=ctx.cfg,
          grid=grid,
          layout=ctx.layout,
          state_old=ctx.state_old,  # 这里仍用旧状态，保持和线性版本一致
          props=props_from_old,     # 第一版可以偷懒用 old props
          dt=ctx.dt,
      )
      ```
   6. 残差：

      ```python
      res = A @ u - b
      return res, {"diag": diag}
      ```

> 这一版 F(u) 实际上是在“老的半隐式线性离散”外边套了一个非线性壳，主要目的是：
> **验证 pack/apply + build_system_SciPy + SciPy Newton 这一条流水线是跑得通的。**

---

## 19.3 SciPy 非线性求解封装（完全交给库做牛顿 / JFNK）

**目标**：不自己写牛顿主循环，用 SciPy 现成的非线性 solver。

1. 新建 `solvers/newton_scipy.py`，核心函数：

   ```python
   def solve_one_step_nonlinear_scipy(...):
       # 构造 u0, ctx
       # 定义 F(u) = build_global_residual(u, ctx)[0]
       sol = scipy.optimize.newton_krylov(F, u0, ...)
       # 或 scipy.optimize.root(F, u0, method="krylov")
       # 解出来的 sol -> u_new -> state_new
   ```

2. 只保留**最少的自编逻辑**：

   * 日志打印（residual norm, iter）
   * 出错时返回失败状态，不要自己写线搜索、Jacobians，先完全交给 SciPy。

3. 在 `timestepper.py / run_scipy_case.py` 里加入选择：

   * `solver.mode = "linear"` 用旧套路
   * `solver.mode = "nonlinear_scipy"` 用新套路调用 `solve_one_step_nonlinear_scipy`

> 到目前为止：
> **物理离散还没变，但全局 F(u) + SciPy 非线性求解框架已经搭起来了。**

---

## 19.4 真·全隐式改造：把所有“显式”项改成“用 state_guess 计算”

这一块才是你说的“最终改为全隐式”。分块来，避免一次性改爆。

### 19.4.1 气相能量：对流从显式 → 隐式

**目标**：`q_conv` 用 `state_guess`，将 Stefan 对流完全纳入 F(u)(u^{n+1})。

修改点：

1. 在 `build_global_residual` 中，**重算 props & Stefan 用 state_guess**：

   ```python
   props_new = compute_props(cfg, grid, state_guess)
   stefan = compute_stefan_velocity(cfg, grid, props_new, state_guess)
   ```

2. `compute_gas_convective_flux_T` 调用中：

   * 现在是 `Tg=state_old.Tg`，改成 `Tg=state_guess.Tg`。
   * u_face 用上一步的 `stefan.u_face`（这本身已随 state_guess 变化）。

3. 装配对流源项时仍然写入 b，但是 b 现在是 `b(u)`，F(u)=A(u)u−b(u) 就是非线性。
   数学上仍然是全隐式 BE，因为对流项依赖的是 u^{n+1}。

> 此时气相能量方程在时间上已经全隐式。

---

### 19.4.2 气相组分：对流从显式 → 隐式

同样操作：

1. 在 species 系统装配里：

   * `compute_gas_convective_flux_Y` 改用 `state_guess.Yg`。
   * u_face = Stefan(state_guess)。

2. 对流通量进 b(u)。

> 气相 Yg 的对流现在也完全随 u^{n+1} 更新，时间上全隐式。

---

### 19.4.3 液相组分：扩散通量从显式 → 隐式

你表里这一行是唯一标成“显式通量”的地方，专门处理一下。

1. 在 `build_liquid_species_system_SciPy.py` 中，找到类似：

   ```python
   J_diff = compute_liquid_diff_flux_Y(cfg, grid, props, state_old.Yl or lagged_Yl)
   b[row] -= (A_R * J_R - A_L * J_L)
   ```

2. 改成：

   ```python
   J_diff = compute_liquid_diff_flux_Y(cfg, grid, props_new, state_guess.Yl)
   ```

3. 同样，只动通量的“输入 state”，装配形式保持不变。

> 到这一步，所有质量/能量方程里的输运项（扩散 + 对流）都用 u^{n+1}，从时间离散角度已经是后向 Euler 全隐式。

---

### 19.4.4 界面 Ts / mpp：从“线性化”变成“完全依赖 state_guess 的非线性项”

这部分你现在是典型的“线性化 BC”：
`interface_bc` 给出 a_P, a_N, b，把它拼进 A,b 里。

全隐式 Newton 里，不要求你完全扔掉线性化，但要保证：

* BC 所有系数都用 **state_guess** 计算；
* 最终 F(u) 中这一块的残差真的是 `F_if(u)=0` 的离散形式，而不是“仅仅在 old state 上线性化一次”。

最小改法（不自虐版）：

1. 在 `build_global_residual` 里，把 **接口相关的 state 输入改为 state_guess**：

   * 传给 `interface_bc` / `energy_flux` 的 Ts、Tg0、TlIf、Yg0、YlIf 全来自 `state_guess`。
   * `Raoult` / `CoolProp` 计算饱和压、Yg_eq 也用 `state_guess.Ts` 和 `state_guess.Yl_if`。

2. 让 interface BC 给出的 a/b 矩阵系数，每次 F(u) 调用都重算一次。
   这样 F(u) 完整依赖 u（Ts,mpp,Yg,Yl,…），Newton/JFNK 会自动做非线性逼近。

若你以后想更“纯粹”，可以再往前走一步：

* 在 F(u) 中直接写接口方程：

  * 能量：`q_g(Tg0(u),Ts(u)) + q_l(Tl_if(u),Ts(u)) + q_lat(mpp(u),Ts(u)) = 0`
  * 质量：`Stefan_balance(mpp(u), Yg0(u), Yg_eq(Ts,Yl_if)) = 0`
* 减少 interface 里的“预线性化”，但这是下一轮的精细化，可以晚点做。

---

### 19.4.5 动网格 & Rd：从“外部更新”到“真正在 F(u) 里耦合”

现在 Rd 是“后向 Euler + 单独处理”的状态。要全隐式，需要：

1. 在 `build_global_residual` 里，每次调用 F(u) 时：

   * 从 `state_guess.Rd` 拿到当前猜测半径；
   * 调 `rebuild_grid_with_Rd(cfg, Rd_guess, grid_ref)` 生成本次迭代的 grid；
   * 把 `state_old` 用当前 grid 做一次 remap，用于时间项；
   * 在这个 grid 上装配所有方程。

2. 半径方程残差：

   * 现在你在 `radius_eq.py` 里已经有后向 Euler 形式，可以让 `build_transport_system` 生成对应矩阵行；
   * 或者在 `build_global_residual` 里额外拼一行 `F_Rd(u)`，跟其它残差一起返回。

重点：**网格与 Rd 的关系也变成 F(u) 的一部分**。
只要在评估 F(u) 时用的是 `state_guess.Rd` 这一个值，Newton 就会自动处理 “T,Y,mpp,Rd” 的耦合。

---

## 19.5 测试路线（保证每一步不把自己搞死）

每做完一个 19.4.x 子步骤，都配一个小 test：

1. **“伪线性一致性测试”**

   * 锁死物性 & Stefan & interface，使它们只依赖 `state_old`（即强制“线性”），
   * 用线性 solver vs 非线性 solver 对比：两者结果应在数值误差内一致。

2. **“显式→隐式改造后的稳定性测试”**

   * 适当放大 dt，看全隐式版本是否比老的半隐式在收敛性、稳定性上更好（至少不更差，且不炸）。

3. **界面局部测试**

   * 重用现有 `test_interface_*` 和 `test_radius_eq_*`，只把底层调用换成“走 F(u) + SciPy Newton 一步”。

---

## 19.6 PETSc 版本（Linux）：把 F(u) 接到 SNES 上

这一步基本是“后台替换”，逻辑就一句话：

> F(u) 还是 `build_global_residual(u, ctx)`，
> 把 SciPy 的 `newton_krylov` 替换成 PETSc 的 SNES（矩阵自由 / FD Jacobian）。

工作内容：

1. 新建 `solvers/newton_petsc.py`：

   * 实现一个 `solve_one_step_nonlinear_petsc(...)`，接口和 SciPy 版尽量一致。
   * 定义 SNES FormFunction：从 Vec → numpy → F(u) → Vec。

2. 在 `cfg.solver.backend` 中增加选项：

   * `"scipy"` / `"petsc"`

3. `timestepper` 中调用统一入口：

   ```python
   if cfg.solver.backend == "scipy":
       solve_one_step_nonlinear_scipy(...)
   elif cfg.solver.backend == "petsc":
       solve_one_step_nonlinear_petsc(...)
   ```

> 非线性主循环 & 线性求解、预条件全交给 PETSc，通过命令行 / cfg 调 KSP / PC 类型。

---

## 19.7（可选加分项）：解析雅可比 / Jacobian 重用

这个阶段纯属锦上添花，不是全隐式的硬需求：

1. 实现 `jacobian_fd.py` 或 `jacobian_analytic.py`：

   * 提供 `build_jacobian(u, ctx)`，给 PETSc SNES 或你自己的 debug 用。

2. 加上 Jacobian 重用策略：

   * 当 `||F||` 下降得很顺利时，不每步重建 J；
   * 当 `||F||` 降不动或上升时，强制刷新 J。

3. 只在需要性能 / 诊断的时候打开这个分支。

---

## 总结一句话版本

* 2.2 里“隐式/显式”说的是**当前线性离散里用谁算通量、进 A 还是进 b**；
* Step 19 的目标是：
  **构造一个 F(u)，让所有通量、界面条件、动网格都用 u^{n+1} 的 state_guess 计算，再把 F(u) 丢给 SciPy / PETSc 做非线性 Newton。**
* 工作路线就是：

  1. 搭 UnknownLayout + NonlinearContext；
  2. 搭第一版 F(u)（沿用半隐式离散，只是包起来）；
  3. 接 SciPy 的 newton_krylov（不用自己写牛顿）；
  4. 逐块把显式通量 / 对流 / 界面 / Rd 改成用 state_guess 的全隐式；
  5. 最后把同一个 F(u) 接到 PETSc SNES 上。

接下来你要是真按这条路线走，我们就可以从 19.1 开始，文件一个个拆着改了。
