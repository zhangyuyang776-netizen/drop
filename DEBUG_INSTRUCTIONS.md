# 调试测试失败说明

## 问题现状

测试 `test_run_scipy_const_props.py` 返回错误码 99，表示捕获了未处理的异常。

## 已完成的修复

1. ✅ 在 mechanism/mech.yaml 中添加了 O2 物种
2. ✅ 修复了 run_scipy_case.py 中的 step_id 重复递增 bug

## 需要获取详细错误信息

由于测试使用 `log_level=logging.WARNING`，详细的异常堆栈被隐藏了。

### 方法 1: 使用提供的调试脚本

运行：
```bash
python debug_test_with_logging.py
```

这将使用 DEBUG 日志级别运行测试，输出完整的错误堆栈。

### 方法 2: 直接修改测试文件

临时修改 `tests/test_run_scipy_const_props.py`:

```python
# 将
rc = run_case(str(tmp_yaml), max_steps=25, log_level=logging.WARNING)

# 改为
rc = run_case(str(tmp_yaml), max_steps=25, log_level=logging.DEBUG)
```

然后运行测试：
```bash
pytest tests/test_run_scipy_const_props.py::test_case_A_equilibrium -v -s
```

### 方法 3: 检查临时输出目录

测试失败后，检查临时输出目录中是否有日志文件或输出文件，可能包含错误信息。

## 可能的问题

基于代码分析，可能的问题包括：

1. **机制文件加载问题**
   - Cantera 无法加载修改后的 mech.yaml
   - 物种顺序或数据格式问题

2. **物种数量不匹配**
   - gas_species 配置与机制中的物种数量不匹配
   - state.Yg 形状与预期不符

3. **属性计算失败**
   - CoolProp 调用失败
   - 温度或压力超出有效范围

4. **路径问题**
   - 临时目录创建失败
   - mechanism_dir 路径解析错误

## 下一步

请运行调试脚本并提供完整的输出，包括：
- 所有日志消息
- 异常类型和消息
- 完整的堆栈跟踪

这样我就能准确定位问题并提供针对性的修复。
