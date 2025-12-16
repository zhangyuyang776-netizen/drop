行，你说得对：**接口格式就是“隐性规范”里最重要的那一条**。因为它一旦乱了，后面就会出现你熟悉的剧情：数值没问题，物理也没问题，最后死在“某个模块偷偷改了某个字段”的人祸上。

我这边现在没法给你生成可下载链接（生成文件的环境一直报错），所以先把 **完整 md 内容直接贴出来**。你复制保存为 `NewProject_Guide_NoChem_PETSc.md` 就行。

````md
# 新代码项目说明文件（无化学源项版本）
基于：`Droplet_Transport_Framework_NoChemistry.md`（多组分液滴自发点火数值框架：无化学源项）

> 本文件目的：总结新代码项目的整体框架、模块职责、接口格式与隐性规范要求，并给出旧项目可迁移模块与迁移约束。
> 当前阶段目标：**无化学源项 + 静网格** MVP，先跑通 `A x = b` 的隐式传输步，并可稳定接入 petsc4py。

---

## 0. MVP 版本边界

### 0.1 当前版本必须包含
- 1D 球对称（气相域 + 液相域）
- 多组分框架（允许先单组分液滴测试）
- 气相：Stefan 对流（派生速度） + 扩散 + 导热（或焓形式）
- 液相：导热（可选再加液相扩散）
- 界面：无凝结，Stefan 质量条件 + 能量跳跃 + 温度连续
- 液滴半径：与总质量通量一致的 Rd 更新（与 m'' 耦合）
- 线性系统：组装 PETSc `Mat A / Vec b`，KSP 解 `A x = b`，解包成 `State^{n+1}`

### 0.2 当前版本明确不做
- 化学源项进入残差/矩阵（完全移除）
- 动网格/ALE-Rezone（先静网格；只预留接口）
- 自适应 dt（可留接口，先固定 dt）
- 高级预条件器（先 GMRES+ILU/ASM 能跑即可）

---

## 1. 总体架构与“隐性规范”总则

### 1.1 三个核心对象（唯一事实来源）
- `Grid1D`：几何与离散（r、V、A、cell/face 映射、界面索引）
- `State`：未知量主存储（Tg/Yg/Tl/Yl + Ts/m''/Rd），必要派生量可缓存
- `Props`：物性与传输系数主存储（ρ、cp、λ、D、h、psat…），由 State 更新得到

**强制规则：**
1) 同一物理量只能有一个“主存储位置”（不要复制成多份真相）。  
2) 模块之间只通过参数传递对象，不允许全局单例 data 污染。  
3) 组装阶段只写 `A,b`，严禁偷偷改 `State`。

### 1.2 接入 petsc4py 的硬接口（项目的“编写规范核心”）
新项目最重要的可执行接口：

```python
A, b, layout = build_transport_system_petsc(grid, props, state_old, dt, comm)
x           = petsc_solve(A, b, cfg, comm)
state_new   = layout.unpack(x, state_template=state_old)
````

**隐性规范：**

* `layout` 是唯一合法的索引来源，禁止散落手写 offset。
* `build_transport_system_petsc` 只依赖冻结的 `(grid, props, state_old, dt, cfg)`。
* PETSc 求解模块不允许包含任何物理公式与 state 写回。

### 1.3 方向、符号、单位（必须全链路一致）

* 法向：`n = +e_r`（从液相指向气相）
* `m'' > 0` 表示蒸发（液滴 → 气相）
* 残差约定：**“流出控制体为正”**，每个控制体方程写成
  `积累项 + Σ(流出通量) = 0`
* 导热通量：`q = -λ ∂T/∂r`（沿 +r 的分量）
* 单位：r[m], t[s], T[K], ρ[kg/m^3], m''[kg/m^2/s]

---

## 2. 推荐目录结构与模块职责

```
droplet2/
  run.py
  cases/
    case_001.yaml
  core/
    types.py
    layout.py
    grid.py
    state.py
  properties/
    gas.py
    liquid.py
    equilibrium.py
  physics/
    stefan_velocity.py
    flux_gas.py
    flux_liq.py
    interface_bc.py
    radius_eq.py
  assembly/
    build_system.py
    bc_center.py
    bc_outer.py
  solvers/
    petsc_ksp.py
    timestepper.py
  parallel/
    case_mpi.py
    kernels.py
  io/
    log.py
    writers.py
    checkpoints.py
  tests/
    test_layout.py
    test_scalar_diffusion.py
    test_interface_balance.py
```

---

## 3. 各模块功能、接口、编写/迁移要求

### 3.1 core/types.py（新“data 模块”的正确替代）

**功能：**

* dataclass：`CaseConfig, Grid1D, State, Props, Diagnostics`
* 轻量校验：ΣY、非负性、数量级、单位一致性

**隐性规范：**

* 只存数据与无副作用检查，不写求解、不写物性、不写装配。
* 所有字段名和 shape 在这里定死，别在别处“发明新字段”。

**旧项目迁移：**

* 旧 `data.py` 的字段清单可迁移为 dataclass 字段
* 旧 `data.py` 的写回/计算逻辑必须拆走

---

### 3.2 core/layout.py（UnknownLayout：索引即规范）

**功能：**

* 统一管理向量索引与 pack/unpack（State ↔ PETSc Vec）

**接口建议：**

```python
class UnknownLayout:
    def size(self) -> int: ...
    def idx_Tg(self, i): ...
    def idx_Yg(self, k, i): ...
    def idx_Tl(self, j): ...
    def idx_Yl(self, m, j): ...
    def idx_Ts(self): ...
    def idx_m_evap(self): ...
    def idx_Rd(self): ...
    def pack(self, state) -> PETSc.Vec: ...
    def unpack(self, x, state_template) -> State: ...
```

**隐性规范：**

* 禁止散落的“闭合物种/归一化”逻辑：必须统一在 `layout.pack/unpack` 或 `state.enforce_constraints()`。
* pack/unpack 必须可逆并用单测锁死。

---

### 3.3 core/grid.py（几何地基）

**功能：**

* 构造 r 网格（cell/face）
* 预计算体积/面积/距离：`V[i], A_face[f], dr`
* 界面定位：界面 face index、两侧 cell index
* 边界标记：center / farfield

**隐性规范：**

* 所有几何量预计算并缓存，装配阶段不重复算几何。
* Grid 不存状态/物性，避免耦合。

**旧项目迁移：**

* geometry 的体积面积、索引映射可迁移
* ALE/Rezone 暂不迁移（只留接口占位）

---

### 3.4 properties/gas.py & liquid.py（物性：可并行、尽量纯函数）

**功能：**

* `Props.update_from_state(state, grid, cfg)`
* 气相：Cantera 计算 ρ/cp/λ/D/h（按 cell）
* 液相：CoolProp 计算 ρ/cp/λ/hvap/psat（按 cell 或界面）

**隐性规范：**

* Props 是唯一物性主存储；State 不允许保存“物性真相”。
* 质量/摩尔分数换算必须依赖可靠分子量来源（Cantera 或固定表），避免“先用后算”的旧坑。
* 物性更新只写 Props，不写 State。

---

### 3.5 properties/equilibrium.py（界面平衡：Raoult + psat + 背景气填充）

**功能：**

* 给定界面液相组成与 Ts：计算界面饱和/平衡气相组成 `Yg_s`
* 明确背景气填充策略（远场背景气 vs 界面原有非凝相）：二选一写死进 cfg

**隐性规范：**

* 这部分逻辑只能在一个模块里存在，禁止 interface_bc 里再写一份。

---

### 3.6 physics/stefan_velocity.py（速度派生，不进未知量）

**功能：**

* `v(r) = m'' * Rd^2 / (ρ_g(r) * r^2)`
* 提供 `v_face`（用于对流通量）

**隐性规范：**

* v 不进入 UnknownLayout，不作为未知量，不进残差。
* 插值策略（ρ_face、T_face、Y_face）要统一：一个函数管到底，别每个模块各插各的。

---

### 3.7 physics/flux_gas.py / flux_liq.py（通量离散元件）

**功能：**

* 提供局部 stencil/系数，供 assembly 写入 A,b
  （时间项、对流项、扩散项、导热项）

**隐性规范：**

* 采用有限体积写法：积累 + Σ(流出通量)=0
* 通量方向、符号链必须与 interface_bc、assembly 完全一致

---

### 3.8 physics/interface_bc.py（界面条件：最容易出符号灾难的地方）

**功能：**

* 温度连续：Tg(Rd+)=Tl(Rd-)=Ts
* Stefan 质量条件：基于界面平衡组成与扩散通量求 m''
* 能量跳跃：导热 + 扩散焓 + 潜热（无凝结）

**输出形式（推荐）：**

* 返回“用于装配的系数包”，而不是直接写矩阵：

```python
InterfaceCoeffs = {
  "rows": [
     {"row": idx_Ts, "cols": [...], "vals": [...], "rhs": ...},
     {"row": idx_m,  "cols": [...], "vals": [...], "rhs": ...},
     {"row": idx_Rd, "cols": [...], "vals": [...], "rhs": ...},
  ],
  "diag": {...}
}
```

**隐性规范：**

* interface_bc 不能做物性计算（psat/Raoult 在 equilibrium 里）。
* interface_bc 不允许偷偷归一化 Y（归一化策略统一在 state/layout）。

---

### 3.9 physics/radius_eq.py（Rd 方程）

**功能：**

* 提供 Rd 与 m'' 的耦合方程离散（装配行）
* 目标：与总质量守恒一致，符号与 m'' 一致

---

### 3.10 assembly/build_system.py（核心：把物理翻译成 PETSc）

**功能：**

* 创建 PETSc `Mat A`、`Vec b`
* 遍历所有未知量行：Tg、Yg、Tl、Yl、Ts、m''、Rd
* 应用中心与远场边界（bc_center / bc_outer）
* 写入界面方程行（来自 interface_bc / radius_eq）

**petsc4py 规范（隐性但必须）**

* 预分配：AIJ matrix 必须预估每行非零数，否则性能和收敛都会发疯
* 统一填值模式：`setValue(addv=True)`，最后 `assemble()`
* build_system 只写 A,b，不改 State

---

### 3.11 solvers/petsc_ksp.py（求解器只做求解）

**功能：**

* 封装 KSP/PC 参数（cfg + PETSc options）
* `x = solve(A,b)`

**隐性规范：**

* 这里不允许出现任何物理公式、任何 state 写回。

---

## 4. “隐性迁移与规范要求”清单（重点）

### 4.1 归一化与闭合物种策略（必须唯一）

* ΣY=1 的闭合策略必须写死在一个地方（推荐：layout 或 state）
* 任何模块不得重复归一化（否则会产生你旧项目那种 0.2% 差别和“数据污染”）

### 4.2 插值规则必须统一（face 值怎么来）

* `T_face, Y_face, ρ_face` 的插值/上风策略要统一封装
* 否则同一物理量在不同模块得到不同 face 值，残差会“看起来合理但永远不收敛”

### 4.3 “禁止隐式写回”

* Props 更新只写 Props
* 组装只写 A,b
* interface 只返回系数/诊断，不直接改 State（除非显式在主流程调用）

### 4.4 诊断与硬报错（不符合常识就别继续跑）

每步建议至少检查：

* ΣY（气相/液相）是否接近 1
* 最小 Y 是否出现大负数（超过阈值就报错/回退）
* T、ρ、cp、λ、D 是否为正
* 界面能量/质量符号是否一致（m'' > 0 时应对应蒸发）

---

## 5. 旧项目迁移清单（可迁移 vs 必须重写）

### 5.1 可迁移（逻辑保留，接口适配）

* geometry → grid（网格、体积面积、索引映射）
* initial → state init
* Cantera 气相物性 → properties/gas
* CoolProp 液相物性 → properties/liquid
* psat + Raoult + 背景气填充 → properties/equilibrium
* 输出/日志 → io 模块（按新数据结构改字段）

### 5.2 必须重写

* Residuals_assembly / Newton / JFNK / 自研 GMRES / search / solver_data
  → 全部替换为 PETSc KSP + build_system(A,b)
* 旧 pack/unpack（尤其含“偷偷归一化/转换”）
  → 用 UnknownLayout 重写并单测锁死

---

## 6. 并行策略（从一开始预留）

* 多 case 并行：mpi4py（rank → case）
* case 内并行：物性与通量 cell 级 kernels（先留接口）
* PETSc 并行：1D 通常不必，但 comm 必须贯穿接口，未来可扩展

---

## 7. 里程碑（建议按此顺序推进）

* M1：layout + pack/unpack（单测通过）
* M2：标量扩散（无界面，v=0）跑通
* M3：加入 Stefan 对流（派生 v），气相多组分无界面跑通
* M4：加入界面 + Ts/m''/Rd（单组分液滴）跑通
* M5：扩展到多组分液滴（逐步加 Yl、复杂物性）

---

## 8. 总流程图（模块/数据流）

```mermaid
flowchart TD
  cfg[CaseConfig] --> grid[Grid1D]
  cfg --> layout[UnknownLayout]
  cfg --> stateN[State^n]

  stateN --> props[Props.update_from_state]
  props --> assemble[build_system -> A,b]
  grid --> assemble
  stateN --> assemble

  assemble --> ksp[PETSc KSP solve]
  ksp --> x[Vec x]
  x --> stateNp1[State^{n+1} unpack]
  stateNp1 --> diag[Diagnostics/Checks]
  stateNp1 --> out[IO]
  out --> next[进入下一步]
```

---

## 9. 硬要求总结（写代码时当“宪法”）

1. 索引唯一：UnknownLayout 是唯一索引来源
2. 数据分层：State/Props/Grid 各司其职
3. 装配纯净：build_system 只写 A,b
4. 方向统一：n=+e_r、流出为正、q=-λ dT/dr、m''>0 蒸发
5. 可测试：先跑标量扩散，再逐步加复杂物理

```

如果你愿意，我下一步可以把这份说明再“压成执行清单”，按模块列出 **必须实现的函数名、参数、返回值、最小单测**（那种拿去直接开工不容易跑偏的版本）。
::contentReference[oaicite:0]{index=0}
```
