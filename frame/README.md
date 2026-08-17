# 多能源负荷预测算法框架

本目录的当前实现、方案 2-R 优化及后续阶段安排统一以 `plan/Methodology与算法框架-分阶段执行计划.md` 为准。旧版阶段 7 已完成 104 次运行与汇总，但由于随机种子未覆盖模型初始化，旧结果仅保留为 `legacy_non_strict_seed_control` 工程记录。修复后的 Stage 7-R 已完成当前冻结下的 124 条有效记录；由于主模型是结构匹配 STL，A0—A4 均独立训练，不存在 A4 复用记录。

## 当前数据集：Kitakyushu Energy Station Data

官方数据与说明：

```text
数据论文：A twenty-year dataset of hourly energy generation and consumption from district campus building energy systems
论文 DOI：10.1038/s41597-024-04244-6
Figshare DOI：10.6084/m9.figshare.24978645
官方页面：https://figshare.com/articles/dataset/Energy_Station_Data/24978645
```

原始 ZIP 数据位于项目根目录的 `Kitakyushu dataset/`，审计和清洗后的统一文件为：

```text
frame/reports/data_audit/kitakyushu/cleaned_canonical.csv
```

统一数据包含四个预测任务：

```text
electricity → 园区/能源站电力需求
cooling     → 建筑群供冷需求
heating     → 建筑群供热需求
gas         → 能源站系统侧天然气消耗/购气量
```

`gas` 汇总燃气发动机、燃料电池、吸收式冷热机组和燃气锅炉等能源站设备的天然气使用量，不解释为单栋建筑用户端独立计量的天然气负荷。

正式时间范围为 2015—2021 年小时数据。全年协议为：2015—2019 年训练、2020 年验证、2021 年测试；小样本协议为 2018 年 7 月 1 日至 8 月 31 日，其中 7 月 1 日—8 月 17 日训练、8 月 18 日—8 月 24 日验证、8 月 25 日—8 月 31 日测试。

数据论文说明 2011 年 3 月存在地震造成的数据缺口；本地审计发现 2014 年部分负荷字段异常。两个时期均不进入正式实验范围，目标负荷缺失不得填充为 0。

## 已实现阶段

以下内容已经完成代码实现，并通过单元测试或 CPU 冒烟验证；冒烟结果不等同于正式实验结论。

已完成：

- CPU-only PyTorch 环境和随机种子设置；
- 四任务顺序、24 小时历史输入和 4 小时预测输出；
- Kitakyushu 三个核心 ZIP 数据包读取、年度字段映射和时间对齐；
- Kitakyushu 气象与日历外生变量；
- 数据质量审计、最小清洗、时间切分和连续滑窗；
- Persistence 和 Seasonal Naive 基线；
- MAE、RMSE、WAPE 和 MAPE；
- 深度可分离因果 TCN 残差块；
- 结构匹配的独立单任务 DS-TCN 模型；
- 训练集专属标准化、逆变换和 PyTorch DataLoader；
- Hard-Share MTL 模型；
- 统一 CPU 训练、验证、最佳 checkpoint 保存和测试预测导出接口；
- 静态有向任务门控、跨任务消息投影和残差融合；
- 静态门控矩阵约束测试与导出。
- 独立状态编码器和状态相关动态对称门控；
- 动态对称门控的对称性、零门控退化、梯度测试和 Kitakyushu CPU 冒烟训练。
- 状态相关动态有向门控、目标/来源任务嵌入和有序任务对门控网络；
- 动态有向门控的方向性、零门控退化、梯度测试和 Kitakyushu CPU 冒烟训练。
- 方案 2-R 的全窗口 DS-TCN（卷积核 5、膨胀率 1/2/4，默认感受野 29）；
- 方案 2-R 的增强状态编码、预测步嵌入、独立目标/来源任务角色嵌入；
- 方案 2-R 的共享强度 $\rho$、来源分配 $\pi$ 和最终门控 $g=\rho\pi$ 两级有向路由；
- 方案 2-R 的低秩跨任务消息投影、消息 LayerNorm、稳定残差融合和任务-预测步专属标量头；
- 六种内部模型的统一构造、输入输出接口、训练配置和结果导出；
- 参数量、训练耗时、测试评估耗时和门控导出元数据记录。

已完成：

- Stage 6-R.2—6-R.6 的正式选择、鲁棒性、门控诊断、迁移分析和冻结；
- Stage 7-R.3、7-R.4、7-R.5、结构匹配 STL 参考实验和 7-R.6 汇总，共 124 条有效记录；
- Stage 7.7 的 PLE-lite、Hard-Share-H2 和 Dynamic-Symmetric-H1 补充契约、30 次 dry-run 与 6 次 CPU smoke。

当前待办是运行 Stage 7.7 的 30 次正式实验，再运行修订后的 Stage 8。Stage 8 已改为五个联合多任务模型的主比较；STL-H3 只作选择审计，结构匹配 STL-H4 只作负迁移参照。

## 模型接口

```python
prediction = model(loads, exog)
```

```text
loads      [batch, 24, 4]
exog       [batch, 24, F]
prediction [batch, 4, 4]
```

当前 Kitakyushu 默认外生变量维度为 12：温度、湿度、太阳辐照度、风速、风向 5 个气象变量，以及小时、星期、月份的正余弦和周末标记 7 个日历特征。模型代码不硬编码该维度，实例化时传入实际 `exog_dim`。

## 运行命令

### 质量审计

```powershell
D:\anaconda\envs\pytorch\python.exe frame\scripts\audit_kitakyushu.py `
  --data-dir "Kitakyushu dataset" `
  --output-dir frame\reports\data_audit\kitakyushu
```

### 基线

```powershell
D:\anaconda\envs\pytorch\python.exe frame\scripts\run_baselines.py `
  --dataset kitakyushu_energy_station `
  --kitakyushu-data-dir "Kitakyushu dataset" `
  --protocol both `
  --output-dir frame\reports\phase2\kitakyushu
```

两个脚本不要求把数据复制到 `frame/data/raw`。Kitakyushu 协议默认读取项目根目录的 `Kitakyushu dataset/`；如数据位于其他目录，使用 `--data-dir` 或 `--kitakyushu-data-dir` 显式指定。

### 阶段3训练、验证和保存

先运行 CPU 冒烟训练，确认完整链路：

```powershell
D:\anaconda\envs\pytorch\python.exe frame\scripts\train_models.py `
  --dataset kitakyushu_energy_station `
  --kitakyushu-data-dir "Kitakyushu dataset" `
  --model hard_share --protocol full `
  --output-dir frame\reports\phase3\smoke `
  --max-epochs 2 --patience 1 `
  --max-train-samples 512 --max-validation-samples 128 `
  --max-test-samples 256 --batch-size 128 --threads 2
```

`--model stl` 用于结构匹配的独立单任务参照；`--model hard_share` 用于硬共享多任务参照；`--model static_gate` 用于静态有向门控模型；`--model dynamic_symmetric` 用于样本级动态对称门控模型；`--model dynamic_directed` 用于原始动态有向门控模型；`--model scheme2r` 用于方案 2-R。上面的命令是 CPU 冒烟配置（batch size=128、threads=2）；正式实验使用阶段 6 契约规定的资源配置，并去掉 `--max-*-samples` 限制。每次运行会生成：

- `normalization_stats.npz`：仅由训练集拟合的均值和尺度；
- `best_model.pt`：按验证集 Smooth L1 损失保存的最佳模型；
- `history.json`：训练/验证损失；
- `metrics_test.json`：原始量纲下逐任务、逐预测步和等权总体指标；
- `predictions_test.npz`：测试目标、预测值及目标时间。

静态门控模型还会额外生成 `gate_matrix.json`，其中行是目标任务、列是来源任务，对角线固定为0。
动态对称和原始动态有向门控模型额外生成 `gate_matrix_test.npz`，保存测试样本的 `[样本, 4, 4]` 门控矩阵。方案 2-R 额外保存 `gates`、`rho` 和 `pi`：最终门控形状为 `[样本, 4, 4, 4]`，分别对应样本、预测步、目标任务和来源任务；`rho` 形状为 `[样本, 4, 4]`，`pi` 形状为 `[样本, 4, 4, 4]`。正式阶段 6 的门控诊断读取验证集文件，不读取测试集。
`metrics_test.json` 还会记录统一模型接口、模型参数量、训练/测试评估耗时和门控文件元数据。

## 环境和测试

必须使用：

```text
D:\anaconda\envs\pytorch\python.exe
Python 3.9.25
PyTorch 2.8.0+cpu
```

运行测试：

```powershell
& D:\anaconda\envs\pytorch\python.exe -m unittest discover -s frame/tests -v
```

当前代码已覆盖阶段 0—5.5 的主要实现与统一接口，方案 2-R 的阶段 4.6—4.10 也已完成，并完成阶段 6.1 的六模型验证协议冻结。阶段 5.1—5.5 的外部基线和阶段 6.2—6.5 的编排脚本均已具备；旧阶段 6/7 结果因协议问题封存，当前应先按正式实验前审计计划完成 smoke、准入检查和 Stage 6-R 重选，再进入 Stage 7-R。

### 阶段5.1：外部基线公平性契约

外部基线比较契约位于：

```text
frame/configs/external_baseline_fairness.json
```

它固定了任务顺序、24→4预测协议、全年/小样本时间切分、训练集专属标准化、
Smooth L1损失、评价指标、CPU资源记录和三个外部基线的输入声明。校验命令：

```powershell
D:\anaconda\envs\pytorch\python.exe frame\scripts\validate_fairness_contract.py
```

当前契约明确：`MMoE-lite`只保留 MMoE 专家路由思想，不包含 Shao 等人完整模型中的 Frequency 和 STIM 模块，不标注为 “Shao model” 或完整复现；DLinear使用
负荷历史；SOFTS当前以负荷历史最小兼容版本适配，不接入外生变量。

### 阶段5.2：DLinear

已实现 `DLinearBaseline`，采用5点滑动平均分解趋势/季节项，再分别使用线性层
预测未来4步。该基线严格遵守契约中的 `loads_only` 输入模式，不接收天气和日历变量。

CPU 冒烟训练命令：

```powershell
D:\anaconda\envs\pytorch\python.exe frame\scripts\train_external_baseline.py `
  --model dlinear --protocol full `
  --output-dir frame\reports\phase5\dlinear_full
```

当前冒烟结果位于 [dlinear_smoke](/D:/Paper/frame/reports/phase5/dlinear_smoke)，
预测和目标形状均为 `[256, 4, 4]`；参数量和耗时以对应 Kitakyushu 结果目录中的配置与指标文件为准。该结果仅用于验证基线链路，
不代表正式实验结论。训练统计量会严格限制在当前协议的训练起止边界内，小样本协议
不会把训练开始日期以前的数据混入标准化参数。

### 阶段5.3：MMoE-lite

已实现 `MMoELiteBaseline`，使用4个共享小型 MLP 专家、4个任务专属门控和4个任务
专属预测头。每个任务门控对4个专家输出 softmax 权重，再对专家表示加权组合并预测
未来4步。该模型使用24小时历史负荷和历史气象/日历变量，输出固定为 `[batch, 4, 4]`，
不使用未来外生变量；它只保留 MMoE 专家路由思想，不包含 Shao 等人完整 Frequency-STIM-MMoE 模型中的 Frequency 和 STIM 模块，也不声称复现该完整模型。

CPU 冒烟训练命令：

```powershell
D:\anaconda\envs\pytorch\python.exe frame\scripts\train_external_baseline.py `
  --model mmoe-lite --protocol small_sample `
  --output-dir frame\reports\phase5\mmoe_lite_small_smoke
```

冒烟结果会额外导出 `gate_weights_test.npz`，其形状为
`[测试样本数, 4, 4]`，分别对应样本、任务和专家。参数量以对应运行目录的结果配置为准；该结果只用于验证实现和输出链路，不代表正式论文结论。

### 阶段5.4：SOFTS最小适配

已核验 SOFTS 官方 PyTorch 代码，并在 `external_models.py` 中实现本地最小适配
`SOFTSBaseline`。该版本保留反转时间嵌入、STAR 全局核心聚合—通道分发、残差 MLP
和多步线性投影，仅使用历史电/冷/热/气负荷，显式拒绝外生变量；不引入官方仓库的
旧版依赖、独立数据加载器或 GPU 配置，因此不称为 SOFTS 原论文完整复现。

运行命令：

```powershell
D:\anaconda\envs\pytorch\python.exe frame\scripts\train_external_baseline.py `
  --model softs --protocol small_sample `
  --output-dir frame\reports\phase5\softs_small_smoke
```

默认配置为 `d_model=32`、`d_core=16`、`d_ff=64`、1个 STAR 残差块；结果配置会记录
实例归一化和随机核心池化开关。参数量以对应运行目录的结果配置为准，预测输出遵循 `[batch, 4, 4]`。

### 阶段5.5：统一外部基线报告

使用统一编排脚本可在同一协议、样本上限、训练轮数和 CPU 线程下运行三个外部基线：

```powershell
D:\anaconda\envs\pytorch\python.exe frame\scripts\run_external_baselines.py `
  --protocol small_sample `
  --output-dir frame\reports\phase5\unified_small_smoke `
  --max-epochs 2 --patience 1 --threads 2
```

脚本会校验三个模型的协议、任务顺序、测试样本数、输出形状和未来外生变量使用情况，
并生成总体、逐任务、逐预测步 CSV 以及 Markdown 公平性表。当前统一冒烟结果位于
[unified_small_smoke](/D:/Paper/frame/reports/phase5/unified_small_smoke)。

### 阶段6.1：验证协议和选择规则冻结

阶段6.1只冻结预实验规则，不训练模型、不读取测试集指标。校验命令：

```powershell
D:\anaconda\envs\pytorch\python.exe frame\scripts\validate_stage6_selection.py
```

契约文件为 `frame/configs/stage6_selection_contract.json`，规定全年2020验证集为
主要选择依据，小样本验证集为鲁棒性检查，并固定六个候选核心模型和四组有限超参数。

### 旧版阶段6.2—6.5：通用预实验接口（legacy）

以下脚本保留用于工程追溯和小规模接口测试；其旧结果不纳入论文结论。当前正式结论链路使用
`stage6r_2_kitakyushu_full` 至 `stage6r_6_kitakyushu` 目录，已经完成验证集选择、鲁棒性、门控诊断、迁移分析和冻结：

```powershell
D:\anaconda\envs\pytorch\python.exe frame\scripts\run_stage6_selection.py
D:\anaconda\envs\pytorch\python.exe frame\scripts\run_stage6_robustness.py
D:\anaconda\envs\pytorch\python.exe frame\scripts\run_stage6_gate_diagnostics.py
D:\anaconda\envs\pytorch\python.exe frame\scripts\run_stage6_transfer_analysis.py
```

阶段 6-R 的 Kitakyushu 脚本只加载 2015—2020 年原始数据，并只构造训练/验证窗口；2021 年原始文件在 Stage 6-R 中不读取，测试集在阶段 7 前封存，不用于选模型、调超参数或诊断门控行为。正式运行目录非空时必须显式使用 `--resume`，脚本只跳过通过完整性检查的运行。

阶段 6.4 以阶段 6.3 入选候选为诊断对象，因此需要允许契约外候选缺失：

```powershell
D:\anaconda\envs\pytorch\python.exe frame\scripts\run_stage6_gate_diagnostics.py `
  --input-dir "frame\reports\stage6r_3_kitakyushu_small" `
  --output-dir "frame\reports\stage6r_4_kitakyushu" `
  --dataset kitakyushu_energy_station `
  --allow-missing
```

门控诊断数量由阶段 6.3 的入选模型动态决定：Scheme2R 贡献 4 个预测步诊断，其他入选门控模型各贡献 1 个诊断，通常为 5 或 6 个，而不是固定 28 个。

### 阶段 7.0—7.2：冻结交接、消融接口与 CPU 冒烟（历史工程记录）

阶段 7.0 已实现根据 Stage 6.6 冻结文件生成 `frame/configs/stage7r_contract.json` 的接口；阶段 7.1 已实现 A0—A4 递进消融接口，配置记录位于 `frame/configs/stage7_ablation_specs.json`。在新的 Stage 6-R 冻结后，必须重新运行阶段 7.0 生成新的契约，不能继续使用旧的 H3 冻结文件。

阶段 7.2 使用真实 Kitakyushu 数据进行小规模工程验收，默认运行 128 个训练窗口、64 个验证窗口和 64 个测试窗口，最多 2 个 epoch、patience=1。运行命令：

```powershell
D:\anaconda\envs\pytorch\python.exe frame\scripts\run_stage7_2.py `
  --kitakyushu-data-dir "D:\Paper\Kitakyushu dataset" `
  --force
```

结果位于 `frame/reports/stage7_2_kitakyushu_smoke/`，包含 A0—A4 和 Dynamic Symmetric 的最佳检查点、标准化参数、训练历史、验证/测试指标和预测文件。该阶段的测试集读取已得到阶段 6.6 冻结授权，冒烟结果只用于工程验收，不作为论文性能结论。

### 历史阶段 7.3：主模型与主要对照（legacy，不得作为论文结果）

旧脚本曾固定运行 Scheme2R-H3 与 Dynamic Symmetric-H3，共 20 个运行。由于旧版本随机性、STL 参照和冻结协议存在问题，这些结果只能作为 `legacy_non_strict_seed_control` 工程记录，不得进入论文，也不要覆盖或追加运行。

```powershell
D:\anaconda\envs\pytorch\python.exe frame\scripts\run_stage7_3.py --dry-run
```

不要再执行旧版正式训练命令；最终运行统一由阶段 7-R 编排器负责。

```powershell
D:\anaconda\envs\pytorch\python.exe frame\scripts\run_stage7_3.py `
  --kitakyushu-data-dir "D:\Paper\Kitakyushu dataset" `
  --output-dir "frame\reports\stage7r_3_kitakyushu_formal"
```

旧目录中虽然存在 20 个运行文件，但它们属于 legacy 结果，不得用于论文性能结论；正式结果必须由 Stage 7-R 在新冻结文件下重新生成。

### 历史阶段 7.4：A0—A4 消融（legacy，不得作为论文结果）

旧脚本曾把 A0—A4 全部重复训练为 50 个运行。当前冻结主模型不是 Scheme2R，因此阶段 7-R 按动态规则保留A0—A4的50个独立训练记录；只有冻结主模型确实为 Scheme2R 且配置、协议和种子完全匹配时，才允许复用A4。

```powershell
& $py frame\scripts\run_stage7_4.py --dry-run
```

不要再执行旧版 50 次运行命令；最终矩阵由 `generate_formal_run_matrix.py` 自动生成。

```powershell
& $py frame\scripts\run_stage7_4.py `
  --kitakyushu-data-dir "D:\Paper\Kitakyushu dataset" `
  --output-dir "frame\reports\stage7r_4_kitakyushu_formal"
```

旧目录中虽然存在 50 个运行文件，但它们属于 legacy 结果，不得用于论文性能结论；当前修复后的矩阵已对A0—A4全部独立训练并通过验收。

### 历史阶段 7.5：外部基线（legacy，不得作为论文结果）

旧阶段 7.5 的 34 次运行同样只保留为工程追溯。修复后的阶段 7-R 会重新按严格随机种子和新契约运行 44 条记录：4 条确定性基线、30 条外部学习基线，以及 10 条 Scheme2R-loads-only 公平输入控制。

```powershell
& $py frame\scripts\run_stage7_5.py --dry-run
```

不要再执行旧版正式训练命令。

```powershell
& $py frame\scripts\run_stage7_5.py `
  --kitakyushu-data-dir "D:\Paper\Kitakyushu dataset" `
  --output-dir "frame\reports\stage7r_5_kitakyushu_formal"
```

脚本不接受样本上限或冒烟 epoch 参数，逐运行保存指标、预测和训练模型文件，并在根目录生成 `external_runs.csv`、`external_summary_mean_std.csv` 和 `stage7_5_manifest.json`。修复后的正式计划为 44 次运行，结果位于 `frame/reports/stage7r_5_kitakyushu_formal`。下一步是阶段 7.6 统一汇总与阶段 7 验收。

### 历史阶段 7.6：统一汇总（legacy，不得作为论文结果）

旧阶段 7.6 只汇总旧版 104 次运行，不能与修复后结果混合。当前汇总读取四个来源：7.3（20）、7.4训练（50）、7.5（44）和结构匹配STL（10），共124条有效记录。A4是否复用由冻结文件动态决定；本次冻结下复用数为0。

```powershell
& $py frame\scripts\run_stage7_6.py --dry-run
```

确认四个源阶段均为 `passed` 后运行：

```powershell
& $py frame\scripts\run_stage7_6.py `
  --output-dir "frame\reports\stage7r_6_kitakyushu_acceptance"
```

输出目录将包含 `raw_result_index.csv`、`metrics_by_run.csv`、`detailed_metrics_mean_std.csv`、`overall_comparison_mean_std.csv`、`resource_summary_mean_std.csv`、`diagnostic_linkage.csv` 和 `stage7_6_manifest.json`。脚本不会改变任何源结果，也不会重新训练模型；若任一正式运行缺失或失败，会在汇总前直接报错。

在阶段 8 前补充后验资源测量（只加载现有检查点，不调用训练接口）：

```powershell
& $py frame\scripts\run_resource_benchmarks.py `
  --kitakyushu-data-dir "D:\Paper\Kitakyushu dataset" `
  --freeze-config "frame\reports\stage6r_6_kitakyushu\stage6_selected_config.json" `
  --output-csv "frame\reports\stage7r_resource_benchmarks.csv"
```

资源 CSV 完整后，用 `--force` 重新运行上面的 Stage 7.6 汇总命令，使
`resource_summary_mean_std.csv` 和清单中的资源完整性字段同步更新。

阶段 8 最终分析同样不训练模型，只读取冻结后的测试预测和检查点：

```powershell
& $py frame\scripts\run_stage8_analysis.py `
  --kitakyushu-data-dir "D:\Paper\Kitakyushu dataset" `
  --freeze-config "frame\reports\stage6r_6_kitakyushu\stage6_selected_config.json" `
  --stage7-7-dir "frame\reports\stage7r_7_joint_baselines_formal" `
  --bootstrap-replicates 2000 `
  --output-dir "frame\reports\stage8r_joint_revised"
```

修订后的 Stage 8 将 Hard-Share-H2、Dynamic-Symmetric-H1、MMoE-lite、
PLE-lite 和 Scheme2R-H4 作为同赛道联合模型主比较；DLinear、SOFTS adapter、
Persistence 和 Seasonal Naive 进入独立补充表。`stl_matched`-H3 只作为验证集
选择审计记录，`stl_matched`-H4 只作为 Scheme2R-H4 的结构匹配负迁移参照，
二者均不进入联合模型主排行榜。正式运行前可先使用 `--dry-run`；调试可增加
`--bootstrap-replicates 20 --gate-sample-limit 256`，正式运行不要设置
`--gate-sample-limit`。脚本会输出任务总体、任务—预测步、季节/温度分位档/
weekday-weekend 分层迁移表、负迁移率、门控摘要、方向非对称性、门控—误差
关联、资源对照、NPZ 门控数组以及 SVG/PDF/PNG/TIFF 图件。输出目录必须是
新的可写目录；脚本采用临时目录和验收通过后的原子发布，失败时保留失败清单。

### 阶段 7.7：联合模型补充实验（当前待正式运行）

Stage 7.7 新增 `PLELiteBaseline`，并把已固定的 Hard-Share-H2 与
Dynamic-Symmetric-H1 补回同赛道正式比较。PLE-lite 使用两层 CGC：每层包含
2 个共享专家和每任务 1 个私有专家；任务门控只能组合自身私有专家与共享专家，
输出仍为 `[batch,4,4]`，不使用未来外生变量。该实现是轻量 PLE 基线，不声称
复现任何特定论文的完整工程代码。

正式矩阵固定为 3 个模型 × 2 套协议 × 5 个随机种子，共 30 次。全年协议使用
batch size 256、最多 100 epoch、patience 12；小样本协议使用 batch size 32、
最多 200 epoch、patience 20。契约文件为
`frame/configs/stage7_7_joint_baselines_contract.json`。

```powershell
& $py frame\scripts\run_stage7_7.py `
  --kitakyushu-data-dir "D:\Paper\Kitakyushu dataset" `
  --contract "frame\configs\stage7_7_joint_baselines_contract.json" `
  --freeze-config "frame\reports\stage6r_6_kitakyushu\stage6_selected_config.json" `
  --output-dir "frame\reports\stage7r_7_joint_baselines_formal"
```

如被中断，在相同命令末尾增加 `--resume`。脚本只复用通过完整产物校验的运行；
它不会覆盖 Stage 6-R、原 Stage 7-R 或旧 Stage 8 结果。Stage 7.7 的 dry-run
已验证 30 条唯一运行，6 次真实 Kitakyushu CPU smoke 已全部通过；smoke 结果
只证明链路可运行，不进入论文结论。

### 阶段 7-R：严格可复现实验（当前唯一正式入口）

当前状态：实现与审计修复已完成，结构匹配STL接口、Scheme2R-loads-only控制、回归测试和统一 `dry-run` 均已通过；Stage 6-R 和 Stage 7-R 正式结果均已生成。阶段 8 前还需完成资源基准、清单修复和最终分析接口。

阶段 7-R 保留所有旧结果，不覆盖旧目录。当前最终有效矩阵为124条记录：120条学习型训练和4条确定性基线；实际物化运行目录也是124条。新结果分别写入：

```text
frame/reports/stage7r_3_kitakyushu_formal/
frame/reports/stage7r_4_kitakyushu_formal/
frame/reports/stage7r_5_kitakyushu_formal/
frame/reports/stage7r_stl_reference_kitakyushu_formal/
frame/reports/stage7r_6_kitakyushu_acceptance/
```

随机种子现在在模型构造之前设置；训练 DataLoader 同时使用显式种子生成器。结构匹配 STL 将四个任务分别训练，保留全窗口 DS-TCN 和任务—预测步专属头，移除全部跨任务共享。

兼容编排脚本 `run_stage7r_reproducible.py` 的默认输出根目录仍是旧的
`stage7r_kitakyushu_reproducible/`，不要把它与当前已经完成的四个正式结果目录混用。
若要从头重建，必须显式指定独立输出根目录；当前正式结果不需要重新训练。

正式运行前先完成 Stage 6-R 并重新生成 Stage 7.0 契约（旧的 `frame/configs/stage7_contract.json` 已标记为 legacy）：

```powershell
& $py frame\scripts\run_stage7_0.py `
  --freeze-config "frame\reports\stage6r_6_kitakyushu\stage6_selected_config.json" `
  --contract-output "frame\configs\stage7r_contract.json" `
  --force
```

然后检查完整的默认 124 条有效计划：

```powershell
& $py frame\scripts\run_stage7r_reproducible.py --dry-run
```

通过六道准入门后，正式运行支持断点续跑：

```powershell
& $py frame\scripts\run_stage7r_reproducible.py `
  --kitakyushu-data-dir "D:\Paper\Kitakyushu dataset" `
  --resume
```

如果是第一次运行，也可以省略 `--resume`。全部运行完成后检查当前124条汇总计划：

```powershell
& $py frame\scripts\run_stage7r_acceptance.py --dry-run
```

最后执行只读汇总：

```powershell
& $py frame\scripts\run_stage7r_acceptance.py
```

兼容性验收脚本和规范的 Stage 7.6 都应按冻结文件动态检查运行数；当前冻结下应显示124条有效记录、严格种子控制为真，并明确A4复用数为0。

### 阶段 10.0—10.1：预测—调度双轨契约与参数审计（已实现）

调度模块先冻结两条互不混用的证据轨道：`real_replay` 只回放 Kitakyushu
2021 年真实运行序列，`simulated_dispatch` 在明确标注为仿真的标准 IES 中进行
滚动调度。契约文件为
`frame/configs/scheduling_dual_track_contract_v2.yaml`，固定 2015—2019 训练、
2020 验证、2021 测试、24→4 窗口、四任务顺序以及 gas 的能源站设备侧语义；契约
会拒绝共享输出目录、把 gas 写成刚性需求、读取真实未来值或使用测试年缩放参数。

参数证据台账为 `frame/configs/scheduling_parameter_ledger_v2.csv`，由
`frame/src/scheduling/parameter_audit.py` 读取并检查单位、来源、链接、数值、
敏感性范围和 `data_origin` 标签。阶段 10.1 审计命令：

```powershell
& $py -m pytest frame\tests\test_scheduling_contracts.py frame\tests\test_scheduling_parameter_audit.py -q
& $py frame\scripts\audit_scheduling_identifiability.py `
  --contract frame\configs\scheduling_dual_track_contract_v2.yaml `
  --ledger frame\configs\scheduling_parameter_ledger_v2.csv `
  --data-dir "D:\Paper\Kitakyushu dataset" `
  --output-dir frame\reports\scheduling_v2\identifiability_audit
```

只有 `audit_manifest.json` 同时报告 `parameter_evidence=pass`、
`decision_space=pass` 且 `test_year_used_for_scaling=false`，才进入阶段 10.2 的
SciPy/HiGHS 最小求解器验证；这两阶段不运行正式调度，也不改变已冻结的预测模型。

### 阶段 10.2—10.12：调度数据、LP、滚动结算与正式冻结（已实现）

阶段 10.2 固定 SciPy 1.13.x/HiGHS；10.3 将负荷、设备侧 gas、气象、购电和 PV
按时间戳严格合并，并把未来真实字段隔离到结算阶段；10.4—10.5 生成透明 PV/WT
预测并适配五个已冻结的联合预测模型。10.6 的 `real_replay` 只评价真实站点的
购电/购气申报偏差，10.7—10.9 的 `simulated_dispatch` 使用训练期统计冻结的
标准 IES 参数、四小时滚动 LP 和首小时再平衡。10.10—10.11 提供指标、按日
bootstrap、BH 校正、smoke 和正式预检。

正式调度协议由 `freeze_scheduling_contract_v2.py` 生成，不能直接把仓库中的模板
当作冻结证据。示例：

```powershell
& $py frame\scripts\freeze_scheduling_contract_v2.py `
  --repo-root (Get-Location) `
  --benchmark "D:\Paper\standard_ies_benchmark_v1.yaml" `
  --ledger frame\configs\scheduling_parameter_ledger_v2.csv `
  --preflight "D:\Paper\scheduling_preflight_current3\scheduling_preflight_manifest.json" `
  --data-dir "D:\Paper\Kitakyushu dataset" `
  --output "D:\Paper\scheduling_formal_contract_v2.json"
```

### 阶段 10.13：2021 双轨正式执行器（已实现，尚未启动正式运行）

`frame/scripts/run_scheduling_formal_v2.py` 的 `--dry-run` 已固定 25 个 R 轨运行和
150 个 S 轨情景运行，共 175 个。每个运行独立保存 `rows.csv`、`summary.json` 和
带 SHA-256 的 `run_manifest.json`；`--resume` 只跳过哈希完整的运行。R 轨先运行，
随后才运行 S 轨；任何失败都会写入根目录清单并停止，不覆盖另一轨结果。正式执行
命令如下，运行前应确认磁盘空间和 CPU 占用：

```powershell
& $py frame\scripts\run_scheduling_formal_v2.py --dry-run `
  --contract "D:\Paper\scheduling_formal_contract_v2.json"

& $py frame\scripts\run_scheduling_formal_v2.py `
  --contract "D:\Paper\scheduling_formal_contract_v2.json" `
  --data-dir "D:\Paper\Kitakyushu dataset" `
  --benchmark "D:\Paper\standard_ies_benchmark_v1.yaml" `
  --ledger frame\configs\scheduling_parameter_ledger_v2.csv `
  --renewable-file "D:\Paper\renewable_forecasts_test\renewable_predictions_test.npz" `
  --output-dir frame\reports\scheduling_v2\formal
```

该命令才会读取 2021 测试窗口；它不重训预测模型，也不根据 2021 结果调参。
阶段 10.14 的汇总、统计检验、图表和论文同步必须等待 10.13 正式结果完成后再做。

## Pure-simulation scheduling proxy

The scheduling proxy is a separate CPU-trained model.  Scheme2R remains the
frozen forecasting model; the existing SciPy/HiGHS LP remains the exact teacher
and oracle.  Proxy training uses only deterministic synthetic standard-IES
scenarios, never Kitakyushu windows.  Raw proxy dispatch and its feasibility
metrics are reported separately from the optional exact-LP fallback output.
The smoke command validates the pipeline and artifacts only; it is not a formal
paper result or evidence that the raw neural dispatch is feasible.
Balance, conversion, and SOC penalty terms are dimensionless during training
using demand/capacity scales; raw physical residual tensors remain available
for evaluation.
