# 注意事项：case_001.yaml 与 core/types.py（先写着，别忘）

> 目的：不是宏观设计文档，只是把“想到的联动点/容易忘的坑/修改时要同步哪里”记下来。  
> 原则：**case_001.yaml 是配置事实来源**；**types.py 是数据结构与约束的立法层**。两者必须长期一致。

---

## 0. 两条总规则（写给未来的自己）

1. **改 YAML 的字段名/结构 = 必须联动改 types + loader（或 config_reader）**  
   YAML 只是文本；如果 types/loader 不改，等于你“在纸上改了参数”，程序完全不知道。

2. **types.py 只立法，不干活**  
   - ✅ 定义 dataclass、字段名、shape/单位约定、强制约束（__post_init__）、轻量检查函数  
   - ❌ 不做计算（梯度、通量、界面、装配、求解都不放进 types）

---

## 1. case_001.yaml 注意事项（冻结结构后，后面模块围着它转）

### 1.1 paths：路径策略（当前约定）
当前只保留这几项（不引入 output_dir）：

- `paths.output_root`：所有输出根目录（如 `out`）
- `paths.case_dir`：本算例输出目录（如 `out/case_001`）
- `paths.mechanism_dir`：机理目录（如 `mechanism`）
- `paths.gas_mech`：机理文件名或相对 mechanism_dir 的路径（如 `gas_minimal.yaml`）

**联动点**
- 改 paths 的字段名/新增字段：同步改 `core/types.py::CasePaths` 和 YAML 读取器（loader）。

**建议（非强制）**
- 统一把 YAML 中所有路径当作“相对项目根目录”的相对路径，由 loader 统一 resolve 成绝对路径，避免模块各自解释相对路径。

---

### 1.2 species：气相物种从机理读取，闭合物种在 YAML 指定
当前约定：
- `species` 不写 `gas_species`（气相物种列表由机理读取）
- `species.gas_balance_species`：气相闭合物种（如 `N2`），用于 reduced formulation
- `species.liq_species`：液相物种列表（必须显式写）
- `species.liq_balance_species`：液相闭合物种（**必须非 null，且必须在 liq_species 内**）
- 多组分映射与参数：
  - `liq2gas_map`
  - `mw_kg_per_mol`
  - `molar_volume_cm3_per_mol`

**联动点**
- 改 species 结构：同步改 `core/types.py::CaseSpecies` 及其 __post_init__ 约束；同步改 layout/预处理模块里闭合物种读取位置。

**常见坑**
- `liq_balance_species` 写成 null 会在后续“液相组成/界面平衡/物性”路径中埋雷，即使 `solve_Yl=false` 也会踩到。

---

### 1.3 conventions：只放“不可违反的规范”，避免重复真相
conventions 里建议只放：
- 符号/方向约定：`radial_normal / flux_sign / heat_flux_def / evap_sign`
- 项目约束开关（执法性质）：`index_source / assembly_pure / grid_state_props_split`

**联动点**
- 如果 conventions 内出现与 species/physics 重复的“真相”（比如又写一份闭合物种），必须在 loader 或校验函数中强制一致，否则后面肯定对不上。

---

### 1.4 physics：unknown toggles 是 layout 的唯一依据（命名要稳）
physics 内的开关（例）：
- `solve_Tg, solve_Yg, solve_Tl, solve_Yl`
- `include_Ts, include_mpp, include_Rd`
- `enable_liquid, include_chemistry`

**联动点**
- 改这些字段名：同步改 `core/types.py::CasePhysics`、`core/layout.py::build_layout`、以及所有 pack/unpack 相关测试。

---

## 2. core/types.py 注意事项（立法层，不要变成 data.py 2.0）

### 2.1 types.py 负责的东西（现在就写死）
- Case 层（来自 YAML 的“配置契约”）：`CaseConfig` + 各子 dataclass
- 运行时三大对象（跨模块长期存在）：
  - `Grid1D`
  - `State`
  - `Props`
- 轻量执法：shape/ΣY/非负/正性等检查函数

### 2.2 types.py 不负责的东西（禁止迁入）
- 梯度（dT/dr, dY/dr）、面通量（q_face, J_face）、残差/Jacobian、中间缓存  
  这些属于“施工现场”，应该在各自模块内部生成与使用。

> 解释：把中间量塞进 types/State/Props，会快速变成旧项目那种“谁都往里塞”的 Data 神对象。

---

### 2.3 “Yl must exist even if not solved” 的含义（防误解）
- **must exist**：`State.Yl` 字段永远存在，且 shape 合法（`(Nspec_l, Nl)`）
- **not solved**：当 `CasePhysics.solve_Yl == False` 时，`layout` 不把 Yl 打包进未知向量，也不求解液相组分方程

**联动点（关键）**
- `layout.apply_u_to_state()` 必须做到：**只有当 "Yl" block 存在时才允许闭合重建/写回 Yl**  
  否则会出现“solve_Yl=false 但 unpack 顺手改了 Yl”的隐性污染。

---

### 2.4 CaseConfig 必须与 case_001.yaml 同构（不要让 YAML 迁就 types）
当前冻结版 YAML 顶层块：
`case / paths / conventions / physics / species / geometry / time / discretization / initial / petsc / io / checks`

**要求**
- `CaseConfig` 必须覆盖这些块（对应 dataclass），不要只留旧的 `grid/numerics` 合并块。

**联动点**
- 一旦 YAML 新增顶层块或改结构：
  - 同步新增/修改 types 中 dataclass
  - 同步修改 loader/config_reader 的映射
  - 同步更新最小加载测试（建议保留一个最小的 “load case 成功” 测试）

---

## 3. 防遗忘机制（不引入 schema 版本号）

你不想用 schema_version（理解，版本号容易乱），那就用更朴素、更可执行的方式：

### 3.1 在仓库放一个 “契约映射表”（推荐）
文件：`docs/注意事项_case001_types.md`（就是本文件）持续更新即可。  
每次你改 YAML/新增 dataclass，就在这里补一条“联动点”。

### 3.2 保留一个最小的 “case → types → layout” 自检脚本（推荐）
后续可以加 `tests/test_case_load_and_layout.py`，只做三件事：
1. 读 case_001.yaml 构造 `CaseConfig`
2. 用 `CaseConfig` 构造 `UnknownLayout`
3. 运行一次 shape/闭合一致性检查（不做物理计算）

这比“靠记忆”可靠得多。

---

## 4. 追加区：后续想到的新坑就写这里（留空等你补）

- [ ] 新增 grad/flux 模块后，字段命名统一策略（是否引入 `core/fields.py` 常量表）
- [ ] 机理读取后，gas_species 的运行时存放位置（SpeciesInfo vs cfg_runtime）
- [ ] io.fields 的字段名与 State/Props 的字段名绑定策略（避免拼写漂移）
- [ ] PETSc options 映射策略（字符串直接透传 vs 强类型字段）

