# Stage 8 最终分析：实施计划

日期：2026-08-06  
依据：`2026-08-06-stage8-final-analysis-design.md`

## 实施原则

- 测试优先：先写手工可核对的小样例，再修改正式实现；
- 单一事实来源：统计定义集中在 `frame/src/stage8_analysis.py`；
- 脚本只负责输入解析、预检、编排、写出和清单；
- 不重新训练、不重新选择模型、不覆盖 Stage 7-R 原始结果；
- 正式目录只在全部验收通过后发布。

## 任务 1：建立 Stage 8 统计核心模块

文件：

- 新增 `frame/src/stage8_analysis.py`；
- 扩展 `frame/tests/test_stage8_analysis.py`。

步骤：

1. 为迁移收益、MAE/RMSE/WAPE/zero-safe MAPE 编写手工算例；
2. 实现统一指标函数并返回有效样本元数据；
3. 实现任务总体、任务—预测步和情境分层的聚合接口；
4. 验证正负号、空分母、零负荷和形状错误行为。

验收：所有纯函数测试通过，MAPE 不再出现由零分母造成的极大伪值。

## 任务 2：实现分层配对 bootstrap 与 FDR

文件：

- `frame/src/stage8_analysis.py`；
- `frame/tests/test_stage8_analysis.py`。

步骤：

1. 编写五种子 × 时间样本的合成配对数据；
2. 实现种子有放回抽样；
3. 在每个抽中种子内实现 24 小时循环移动块抽样；
4. 实现 2,000 次正式 bootstrap、95% CI 和加一修正单侧 p 值；
5. 实现按协议分别划分的任务总体与任务—预测步 BH 检验族；
6. 输出原始和显著负迁移率。

验收：固定随机种子完全可复现，p/q 值位于 `[0,1]`，不同检验族互不污染。

## 任务 3：实现测试情境标签

文件：

- `frame/src/stage8_analysis.py`；
- 必要时复用 `frame/src/kitakyushu_pipeline.py` 的读取接口；
- `frame/tests/test_stage8_analysis.py`。

步骤：

1. 按每个预测目标时刻生成季节标签；
2. 从对应协议训练期温度计算 Q1/Q2/Q3；
3. 将冻结阈值应用于每个测试目标时刻并生成 Q1—Q4；
4. 生成 weekday/weekend 标签；
5. 测试跨月、跨季节、周五至周六和分位点边界。

验收：所有四个预测步使用自己的目标时刻；温度阈值绝不从测试期拟合。

## 任务 4：实现正式输入预检与配对索引

文件：

- 重构 `frame/scripts/run_stage8_analysis.py`；
- `frame/tests/test_stage8_analysis.py`。

步骤：

1. 校验 Stage 7.6 状态、124/124 正式记录和资源测量完整性；
2. 建立 Scheme2R-H4 与 `stl_matched`-H4 的协议—种子配对索引；
3. 校验任务顺序、候选配置、目标、时间戳和 `[N,4,4]` 预测形状；
4. 对缺文件、重复运行、错配置和测试目标不一致立即报错；
5. 输出 `stage8_input_index.csv`。

验收：不允许静默跳过任何协议或种子。

## 任务 5：实现迁移分析与正式表格

文件：

- `frame/src/stage8_analysis.py`；
- `frame/scripts/run_stage8_analysis.py`。

步骤：

1. 聚合五种子配对结果；
2. 生成 `transfer_task_overall.csv`；
3. 生成 `transfer_task_horizon.csv`；
4. 生成季节、温度、星期类型的 `transfer_context.csv`；
5. 生成 `negative_transfer_rates.csv`；
6. 生成 `temperature_bin_thresholds.csv`；
7. 保存 bootstrap 设置和每个分析单元的样本量。

验收：任务级 8 行、任务—预测步级 32 行；情境表行数由非空分层决定并在清单中记录。

## 任务 6：实现门控诊断与误差关联

文件：

- `frame/src/stage8_analysis.py`；
- `frame/scripts/run_stage8_analysis.py`；
- `frame/tests/test_stage8_analysis.py`。

步骤：

1. 导出每个协议—种子的 `rho`、`pi`、`g` 和上下文标签；
2. 校验 `g=rho*pi`、零对角线和 `[N,4,4,4]` 形状；
3. 生成总体及情境分层 `gate_summary.csv`；
4. 生成带符号与绝对方向差异 `gate_asymmetry.csv`；
5. 计算 `delta_AE` 与 `rho/g` 的逐种子 Spearman 相关；
6. 将相关系数按协议、任务、预测步和来源任务汇总到 `gate_error_association.csv`。

验收：关联结果不输出因果措辞，常数向量输出空相关并记录原因。

## 任务 7：生成资源对照和论文级图件

文件：

- 新增 `frame/src/stage8_figures.py`；
- `frame/scripts/run_stage8_analysis.py`；
- 新增或扩展图件测试。

图件契约：

- 核心结论：Scheme2R 的联合共享收益必须按任务和预测步呈现，同时明确显示局部负迁移；
- 证据层级：任务收益及 CI 为主图，预测步/情境热力图与门控行为为解释图；
- 图形类型：定量网格；
- 后端：Python matplotlib；
- 导出：SVG、PDF、300 dpi PNG，白色背景、可编辑文字、色盲友好发散色标。

步骤：

1. 从 Stage 7-R 资源测量生成 `resource_comparison.csv`；
2. 实现任务收益 CI 图；
3. 实现任务—预测步收益热力图；
4. 实现情境收益热力图；
5. 实现门控摘要、非对称性和门控—误差关联图；
6. 导出源数据和图件 QA 说明；
7. 使用 `validate_figure.py` 审计绘图源，再检查实际导出文件。

验收：所有图件均来自正式 CSV/NPZ，任务和协议无缺失，文字在最终尺寸可读。

## 任务 8：实现原子发布、清单和恢复策略

文件：

- `frame/scripts/run_stage8_analysis.py`；
- `frame/tests/test_stage8_analysis.py`。

步骤：

1. 在正式目录旁创建唯一临时目录；
2. 所有表格、NPZ、图件和清单先写入临时目录；
3. 执行行数、有限值、p/q 范围、样本量、图件格式和输入追溯验收；
4. 验收通过后原子重命名为正式目录；
5. 正式目录已存在时默认拒绝覆盖，必须显式 `--overwrite`；
6. 失败时保留独立失败清单，不伪造 passed 状态。

验收：中断或失败不会留下看似完整的正式结果目录。

## 任务 9：回归测试与真实数据冒烟

步骤：

1. 运行 `test_stage8_analysis.py`；
2. 运行全部 `frame/tests`；
3. 使用较少 bootstrap 重复数运行真实 Kitakyushu Stage 8 冒烟；
4. 检查脚本未调用训练接口、未修改冻结配置；
5. 检查全部表格、NPZ、图片和清单；
6. 正式运行前执行 Stage 8 preflight。

验收：定向测试、全量测试、真实数据冒烟和 preflight 全部通过。

## 任务 10：同步计划书和运行说明

文件：

- `plan/Methodology与算法框架-分阶段执行计划.md`；
- `frame/README.md`。

步骤：

1. 将 Stage 8 统计口径、输出表和图件写入计划书；
2. 更新正式命令、冒烟命令、预计耗时和不覆盖规则；
3. 明确 H4 对 H4 的严格负迁移比较；
4. 明确 MAE 为主要推断指标，情境分层为探索性；
5. 明确 Stage 8 不重新训练。

验收：文档、脚本默认值和输出文件名完全一致。
