# Core 模块审查报告

## 审查范围

- `core/types.py`
- `core/layout.py`

---

## 1. 类型定义审查

**位置**: `core/types.py`

### 1.1 主要数据容器

| 类 | 用途 | 关键字段 |
|----|------|----------|
| CaseConfig | 顶层配置 | case, paths, physics, species, ... |
| Grid1D | 1D球坐标网格 | Nl, Ng, Nc, r_c, r_f, V_c, A_f, iface_f |
| State | 状态变量 | Tg, Yg, Tl, Yl, Ts, mpp, Rd |
| Props | 物性数据 | rho_g, cp_g, k_g, D_g, rho_l, cp_l, k_l, D_l, ... |

### 1.2 符号约定遵守

**State定义**:
```python
@dataclass
class State:
    Tg: FloatArray      # (Ng,) K
    Yg: FloatArray      # (Nspec_g, Ng) 质量分数
    Tl: FloatArray      # (Nl,) K
    Yl: FloatArray      # (Nspec_l, Nl) 质量分数
    Ts: float           # K, 界面温度
    mpp: float          # kg/m²/s, >0蒸发
    Rd: float           # m, 液滴半径
```

**与符号约定.md对比**:
| 约定 | 实现 | 状态 |
|------|------|------|
| mpp > 0 = 蒸发 | ✅ 文档注释正确 | ✅ |
| Yg形状 (Ns_g, Ng) | ✅ 列是空间 | ✅ |
| Yl形状 (Ns_l, Nl) | ✅ 列是空间 | ✅ |

### 1.3 Grid1D约束

```python
def __post_init__(self):
    # Nc = Nl + Ng
    assert self.Nc == self.Nl + self.Ng

    # iface_f = Nl (界面面索引)
    assert self.iface_f == self.Nl

    # r_f严格递增
    assert np.all(np.diff(self.r_f) > 0)

    # r_c在相邻面之间
    assert r_f[i] < r_c[i] < r_f[i+1]

    # V_c > 0, A_f >= 0
    assert np.all(self.V_c > 0)
    assert np.all(self.A_f >= 0)
```

**状态**: ✅ 完整的构造后验证

### 1.4 验证函数

```python
def check_sumY(state, tol=1e-10):
    """验证质量分数归一化"""

def check_nonneg(state, tol=-1e-14):
    """验证质量分数非负"""

def check_positive_props(props, tol=0.0):
    """验证物性正值"""

def check_state_shapes(state, grid, Ns_g, Ns_l):
    """验证State数组形状"""
```

**状态**: ✅ 完整的运行时检查

---

## 2. UnknownLayout 审查

**位置**: `core/layout.py`

### 2.1 布局结构

```python
@dataclass
class UnknownLayout:
    size: int                       # 总自由度
    entries: List[VarEntry]         # 变量条目列表
    blocks: Dict[str, slice]        # 块名→切片映射

    Ng, Nl: int                     # 网格尺寸
    Ns_g_full, Ns_g_eff: int       # 气相组分数 (全/有效)
    Ns_l_full, Ns_l_eff: int       # 液相组分数 (全/有效)

    # 组分映射
    gas_species_full: List[str]
    gas_species_reduced: List[str]
    gas_closure_species: Optional[str]
    gas_full_to_reduced: Dict[str, Optional[int]]
    gas_reduced_to_full_idx: List[int]
    gas_closure_index: Optional[int]
    # 类似的液相映射...
```

### 2.2 块顺序

```python
# build_layout() 中的块顺序:
# 1) Tg: if solve_Tg and Ng > 0
# 2) Yg: if solve_Yg and Ns_g_eff > 0 and Ng > 0
# 3) Tl: if solve_Tl and Nl > 0
# 4) Yl: if solve_Yl and Ns_l_eff > 0 and Nl > 0
# 5) Ts: if include_Ts
# 6) mpp: if include_mpp
# 7) Rd: if include_Rd
```

**与符号约定对比**: ✅ 完全一致

### 2.3 索引函数

```python
def idx_Tg(self, ig: int) -> int:
    """Tg[ig] 的全局索引"""
    return self.blocks["Tg"].start + ig

def idx_Yg(self, k_red: int, ig: int) -> int:
    """Yg[k_red, ig] 的全局索引"""
    # 注意: 单元外层,组分内层
    return self.blocks["Yg"].start + ig * self.Ns_g_eff + k_red

def idx_Tl(self, il: int) -> int
def idx_Yl(self, k_red: int, il: int) -> int
def idx_Ts(self) -> int
def idx_mpp(self) -> int
def idx_Rd(self) -> int
```

**边界检查**: ✅ 每个函数都有范围验证

### 2.4 状态打包/解包

```python
def pack_state(state, layout) -> (u, scale_u, entries):
    """State → 1D向量"""
    # 按layout.entries顺序填充u

def apply_u_to_state(state, u, layout) -> State:
    """1D向量 → State, 重建closure species"""
```

### 2.5 Closure Species重建

```python
def _reconstruct_closure(Y_full, reduced_to_full_idx, closure_idx, ...):
    """从reduced species重建closure species"""
    sum_other = np.sum(Y_full) - Y_full[closure_idx]
    closure = 1.0 - sum_other

    if clip_negative:
        # 容忍小的数值误差
        closure = np.clip(closure, 0.0, 1.0)
    else:
        # 严格检查
        if closure < -tol: raise ValueError
```

**状态**: ✅ 正确处理数值精度

---

## 3. 符号约定一致性总结

### 3.1 已验证的约定

| 约定 | 代码位置 | 状态 |
|------|----------|------|
| 未知量顺序: Tg→Yg→Tl→Yl→Ts→mpp→Rd | layout.py:build_layout | ✅ |
| mpp > 0 = 蒸发 | types.py:State | ✅ |
| 径向坐标增加向外 | Grid1D约束 | ✅ |
| iface_f = Nl | Grid1D约束 | ✅ |
| 组分索引仅通过layout | idx_Yg(), idx_Yl() | ✅ |
| closure species不在未知量 | build_layout逻辑 | ✅ |
| 状态变量形状 (Ns, Ncell) | types.py注释 | ✅ |

### 3.2 代码注释与文档

- ✅ types.py顶部有全局约定文档
- ✅ layout.py顶部有原则声明
- ✅ 每个dataclass有字段文档

---

## 4. 问题与建议

### 4.1 发现的问题

1. **无问题**: Core模块设计良好

### 4.2 Step 19相关建议

1. **残差索引一致性**
   - 残差向量F的索引必须与layout一致
   - 建议添加 `layout.idx_residual_*()` 别名确保一致

2. **雅可比矩阵结构**
   - J[i,j] = ∂F[i]/∂u[j]
   - 可添加 `layout.jacobian_structure()` 返回稀疏结构

---

## 5. 总结

Core模块是整个框架的基础,设计优秀:

- ✅ 清晰的类型定义
- ✅ 完整的约束验证
- ✅ 一致的符号约定
- ✅ 灵活的组分处理 (reduced/full/closure)

Step 19可以直接复用,无需修改。
