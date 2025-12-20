本次改动（build_system_SciPy.py / interface_bc.py / timestepper.py）重点在两块：

1) 让界面平衡 eq_result 在每次残差构造时使用最新状态
2) 将 mpp 方程改成数值雅各比友好版本，避免过度解析线性化与符号抖动

具体修改
---------
- build_system_SciPy.py
  - 引入 compute_interface_equilibrium，并在每次 build_transport_system 调用内重新计算 eq_result（若 include_mpp=True）。优先使用传入的 eq_model，没有则尝试基于当前 Ns_g/Ns_l 和物性推断构造。
  - 保留可选 eq_result 覆盖，避免破坏测试入口。

- interface_bc.py
  - _build_mpp_row 重新实现为“数值雅各比友好”版本：残差 R = j_corr_b - mpp * DeltaY_eff，仅对 mpp 做解析系数，Yg/Ts/Yl 对 j_corr/Yg_eq 的影响交由数值 FD。
  - DeltaY 采用 min_deltaY 软正则，避免硬钳 mpp=0；no_condensation 只记录诊断，不在方程内截断。
  - 诊断保留 j_corr/J_full/mpp_eval 等，方便后续排查。

- timestepper.py
  - 移除外部 eq_result 的预计算传递，改为依赖 build_transport_system 内部重算。
  - 新增 mpp 符号护栏：若求解得到的 mpp 与 interface 诊断 mpp_eval 符号相反且相对差异大于 20%，直接标记 step 失败并记录 diag.extra["mpp_mismatch"]。

实现状态
---------
- 代码已合入上述三个文件；CaseSpecies 及布局也已支持机理主导的物种列表/索引（见 core/types.py、core/layout.py、run_scipy_case.py）。
- 未在本地重跑完整测试，请根据需要运行 pytest 或小算例验证。若 mpp 符号护栏触发，会在 step 结果中返回 success=False 并给出详细信息。
