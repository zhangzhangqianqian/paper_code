# 多能源负荷预测算法框架

本目录的当前实现、方案 2-R 优化及后续阶段安排统一以 `plan/Methodology与算法框架-分阶段执行计划.md` 为准。方案 2-R 尚未进入代码，README 中的“已实现”只指当前模型。

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
- 五种内部模型的统一构造、输入输出接口、训练配置和结果导出；
- 参数量、训练耗时、测试评估耗时和门控导出元数据记录。

暂未完成：

- 阶段 6.2—6.5 的正式预实验、候选模型选择和门控诊断；
- 阶段 6.6 的冻结决策、阶段 7 的正式实验、消融实验和负迁移分析。

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

`--model stl` 用于结构匹配的独立单任务参照；`--model hard_share` 用于硬共享多任务参照；`--model static_gate` 用于静态有向门控模型；`--model dynamic_symmetric` 用于样本级动态对称门控模型；`--model dynamic_directed` 用于样本级动态有向门控模型。上面的命令是 CPU 冒烟配置（batch size=128、threads=2）；正式实验使用阶段 6 契约规定的资源配置，并去掉 `--max-*-samples` 限制。每次运行会生成：

- `normalization_stats.npz`：仅由训练集拟合的均值和尺度；
- `best_model.pt`：按验证集 Smooth L1 损失保存的最佳模型；
- `history.json`：训练/验证损失；
- `metrics_test.json`：原始量纲下逐任务、逐预测步和等权总体指标；
- `predictions_test.npz`：测试目标、预测值及目标时间。

静态门控模型还会额外生成 `gate_matrix.json`，其中行是目标任务、列是来源任务，对角线固定为0。
动态对称和动态有向门控模型额外生成 `gate_matrix_test.npz`，保存测试样本的 `[样本, 4, 4]` 门控矩阵。方案 2-R 尚未实现；其未来门控将增加预测步维度，形状为 `[样本, 4, 4, 4]`。
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

当前代码已覆盖阶段 0—5.5 的主要实现与统一接口，并完成阶段 6.1 的验证协议冻结。阶段 5.1—5.5 的外部基线和阶段 6.2—6.5 的编排脚本均已具备，但目前仍需运行正式实验；下一步是执行阶段 6.2—6.5，随后完成阶段 6.6 冻结决策和阶段 7 正式实验。

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
主要选择依据，小样本验证集为鲁棒性检查，并固定五个候选核心模型和四组有限超参数。

### 阶段6.2—6.5：正式预实验编排

以下脚本用于正式预实验，当前仅完成统一接口和冒烟链路，尚未形成正式论文结论：

```powershell
D:\anaconda\envs\pytorch\python.exe frame\scripts\run_stage6_selection.py
D:\anaconda\envs\pytorch\python.exe frame\scripts\run_stage6_robustness.py
D:\anaconda\envs\pytorch\python.exe frame\scripts\run_stage6_gate_diagnostics.py
D:\anaconda\envs\pytorch\python.exe frame\scripts\run_stage6_transfer_analysis.py
```

阶段 6 的模型选择只读取训练集和验证集；测试集在阶段 7 前封存，不用于选模型、调超参数或诊断门控行为。
