# NPZ文件读取器使用说明

## 简介

`read_npz.py` 是一个用于读取droplet模拟生成的NPZ文件的工具。该工具可以自动识别并读取spatial目录下的所有NPZ文件。

## 安装依赖

在使用之前，请确保安装了必要的Python包：

```bash
pip install numpy
```

如果你已经能够运行droplet模拟，那么这些依赖应该已经安装好了。

## 基本用法

### 1. 列出NPZ文件

查看spatial目录下有哪些NPZ文件：

```bash
python read_npz.py /path/to/spatial --list
```

示例：
```bash
python read_npz.py ../out/step4_2/spatial --list
```

输出示例：
```
在 ../out/step4_2/spatial 中找到 200 个npz文件:

    1. snapshot_000000.npz                 (    12.34 KB)
    2. snapshot_000001.npz                 (    12.45 KB)
    3. snapshot_000002.npz                 (    12.56 KB)
    ...
```

### 2. 读取所有NPZ文件的摘要

显示所有NPZ文件的数据摘要（默认模式）：

```bash
python read_npz.py /path/to/spatial
```

示例：
```bash
python read_npz.py ../out/step4_2/spatial
```

### 3. 读取特定NPZ文件

只读取一个特定的NPZ文件：

```bash
python read_npz.py /path/to/spatial --file snapshot_000000.npz
```

示例：
```bash
python read_npz.py ../out/step4_2/spatial --file snapshot_000000.npz
```

### 4. 详细模式

显示详细信息，包括数组的具体数值：

```bash
python read_npz.py /path/to/spatial --file snapshot_000000.npz --verbose
```

或简写：
```bash
python read_npz.py /path/to/spatial -f snapshot_000000.npz -v
```

### 5. 导出到CSV

将所有NPZ文件导出为CSV格式：

```bash
python read_npz.py /path/to/spatial --export csv
```

这会在spatial目录下创建一个`csv_export`子目录，包含所有导出的CSV文件。

## 输出说明

### 摘要模式（默认）

在摘要模式下，工具会显示：

```
================================================================================
文件: snapshot_000000.npz
路径: /home/user/drop/out/step4_2/spatial/snapshot_000000.npz
================================================================================

元数据:
  step_id: 0
  t: 0.0

网格信息:
  r_c:
    形状: (45,)
    类型: float64
    范围: [2.000000e-05, 9.900000e-03]
    均值: 4.950000e-03

  r_f:
    形状: (46,)
    类型: float64
    范围: [0.000000e+00, 1.000000e-02]
    均值: 5.000000e-03

  r_index:
    形状: (45,)
    类型: int64
    范围: [0, 44]
    均值: 22

场变量:
  Tg:
    形状: (40,)
    类型: float64
    范围: [3.000000e+02, 1.000000e+03]
    均值: 6.500000e+02

  Yg:
    形状: (3, 40)
    类型: float64
    范围: [0.000000e+00, 7.900000e-01]
    均值: 2.633333e-01
```

### 详细模式（--verbose）

在详细模式下，还会显示数组的具体数值：

- 对于小数组（≤20个元素），显示全部内容
- 对于大数组，显示前5个和后5个元素
- 对于数值型数组，还会显示统计信息（最小值、最大值、均值、标准差）

## NPZ文件内容说明

根据配置文件 `cases/step4_2_evap_withYg.yaml`，NPZ文件包含以下内容：

### 元数据
- `step_id`: 时间步编号
- `t`: 当前时间（秒）

### 网格信息
- `r_c`: 网格单元中心的径向坐标
- `r_f`: 网格单元界面的径向坐标
- `r_index`: 网格单元索引

### 场变量

根据配置文件中的 `io.fields` 设置：

**气相场（gas）:**
- `Tg`: 气相温度场
- `Yg`: 气相组分质量分数（多组分数组）

**液相场（liquid）:**
- `Tl`: 液相温度场
- `Yl`: 液相组分质量分数（如果启用）

**标量场（scalars）:**
- `Ts`: 界面温度
- `mpp`: 质量蒸发速率
- `Rd`: 液滴半径

## 工作流程示例

### 完整的分析流程

```bash
# 1. 运行模拟（根据你的项目设置）
python driver/run_scipy_case.py cases/step4_2_evap_withYg.yaml

# 2. 查看生成了多少个NPZ文件
python read_npz.py ../out/step4_2/spatial --list

# 3. 查看第一个快照的摘要
python read_npz.py ../out/step4_2/spatial --file snapshot_000000.npz

# 4. 查看最后一个快照的详细信息
python read_npz.py ../out/step4_2/spatial --file snapshot_000199.npz --verbose

# 5. 导出所有数据到CSV用于进一步分析
python read_npz.py ../out/step4_2/spatial --export csv
```

## 命令行选项参考

| 选项 | 简写 | 说明 |
|------|------|------|
| `spatial_dir` | - | spatial目录的路径（必需） |
| `--list` | `-l` | 仅列出NPZ文件，不读取内容 |
| `--file <filename>` | `-f` | 只读取指定的文件 |
| `--verbose` | `-v` | 显示详细信息（包括数组内容） |
| `--summary` | `-s` | 显示数据摘要（默认） |
| `--export <format>` | - | 导出数据到指定格式（csv, txt） |
| `--help` | `-h` | 显示帮助信息 |

## 注意事项

1. **路径可以是相对路径或绝对路径**：工具会自动解析并转换为绝对路径
2. **大型数组处理**：在详细模式下，大型数组只会显示部分内容以避免输出过长
3. **CSV导出**：导出的CSV文件会将每个数组分别保存，带有适当的标题和索引
4. **内存使用**：如果NPZ文件很多且很大，一次性读取所有文件可能会占用较多内存

## 故障排除

### 问题：找不到numpy模块

```bash
ModuleNotFoundError: No module named 'numpy'
```

**解决方案**：安装numpy
```bash
pip install numpy
```

### 问题：目录不存在

```bash
FileNotFoundError: 目录不存在: /path/to/spatial
```

**解决方案**：
1. 检查路径是否正确
2. 确保已经运行过模拟生成了输出文件
3. 检查配置文件中的 `paths.case_dir` 设置

### 问题：没有找到NPZ文件

```
在目录 /path/to/spatial 中没有找到npz文件
```

**解决方案**：
1. 确认模拟已经运行并写入了输出文件
2. 检查配置文件中的 `io.write_every` 设置
3. 确认指向的是 `spatial` 子目录，而不是 `case_dir` 根目录

## 进阶用法

### 在Python脚本中使用

你也可以在自己的Python脚本中导入和使用这些功能：

```python
from pathlib import Path
import sys
sys.path.insert(0, '/home/user/drop')
from read_npz import find_npz_files, read_npz_file

# 查找NPZ文件
spatial_dir = Path("../out/step4_2/spatial")
npz_files = find_npz_files(spatial_dir)

# 读取第一个文件
data = read_npz_file(npz_files[0])

# 访问数据
print(f"时间步: {data['step_id'].item()}")
print(f"时间: {data['t'].item()}")
print(f"温度场形状: {data['Tg'].shape}")
print(f"温度范围: [{data['Tg'].min()}, {data['Tg'].max()}]")
```

## 相关文件

- `read_npz.py` - NPZ读取器主程序
- `io/writers.py` - NPZ文件写入逻辑（了解文件格式）
- `cases/step4_2_evap_withYg.yaml` - 示例配置文件

## 反馈和问题

如有问题或建议，请在项目的issue tracker中提出。
