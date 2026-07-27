# 多能源负荷预测算法框架

本目录按照 `plan/Methodology与算法框架_分阶段执行计划.md` 实现论文算法框架。

## 当前数据集：HEEW 区域级汇总

正式数据位于项目根目录：

```text
dataset/HEEW/cleaned_data/Total_energy.csv
dataset/HEEW/cleaned_data/Total_weather.csv
```

两个文件合并后得到三个预测任务：

```text
Electricity → electricity
Cooling     → cooling
Heat        → heating
```

时间范围为 2014—2022 年小时数据。全年协议为：2014—2019 年训练、2020 年验证、2021—2022 年测试；小样本协议为 2018 年 7 月 1 日至 8 月 31 日的夏季切分。

## 已实现阶段

已完成：

- CPU-only PyTorch 环境和随机种子设置；
- 三任务顺序、24 小时历史输入和 4 小时预测输出；
- HEEW 双文件时间构造、字段映射和时间对齐；
- HEEW 气象与日历外生变量；
- 数据质量审计、最小清洗、时间切分和连续滑窗；
- Persistence 和 Seasonal Naive 基线；
- MAE、RMSE、WAPE 和 MAPE；
- 深度可分离因果 TCN 残差块；
- 结构匹配的独立单任务 DS-TCN 模型；
- 训练集专属标准化、逆变换和 PyTorch DataLoader；
- Hard-Share MTL 模型；
- 统一 CPU 训练、验证、最佳 checkpoint 保存和测试预测导出接口。
- 静态有向任务门控、跨任务消息投影和残差融合；
- 静态门控矩阵约束测试与导出。

暂未完成：

- 状态相关动态对称/动态有向门控；
- 正式全量训练、消融实验和负迁移分析（阶段6—7）。

## 模型接口

```python
prediction = model(loads, exog)
```

```text
loads      [batch, 24, 3]
exog       [batch, 24, F]
prediction [batch, 4, 3]
```

当前 HEEW 默认外生变量维度为 14：7 个气象变量和 7 个日历特征。模型代码不硬编码该维度，实例化时传入实际 `exog_dim`。

## 运行命令

### 质量审计

```powershell
D:\anaconda\envs\pytorch\python.exe frame\scripts\audit_data.py `
  --energy-file dataset\HEEW\cleaned_data\Total_energy.csv `
  --weather-file dataset\HEEW\cleaned_data\Total_weather.csv `
  --output-dir frame\reports\phase1
```

### 基线

```powershell
D:\anaconda\envs\pytorch\python.exe frame\scripts\run_baselines.py `
  --energy-file dataset\HEEW\cleaned_data\Total_energy.csv `
  --weather-file dataset\HEEW\cleaned_data\Total_weather.csv `
  --output-dir frame\reports\phase2
```

两个脚本不要求把数据复制到 `frame/data/raw`。不带文件参数时，它们会尝试读取上述 HEEW 默认路径。

### 阶段3训练、验证和保存

先运行 CPU 冒烟训练，确认完整链路：

```powershell
D:\anaconda\envs\pytorch\python.exe frame\scripts\train_models.py `
  --model hard_share --protocol full `
  --output-dir frame\reports\phase3\smoke `
  --max-epochs 2 --patience 1 `
  --max-train-samples 512 --max-validation-samples 128 `
  --max-test-samples 256 --batch-size 128 --threads 2
```

`--model stl` 用于结构匹配的独立单任务参照；`--model hard_share` 用于硬共享多任务参照；`--model static_gate` 用于静态有向门控模型。正式运行时去掉三个 `--max-*-samples` 限制，并固定输出目录。每次运行会生成：

- `normalization_stats.npz`：仅由训练集拟合的均值和尺度；
- `best_model.pt`：按验证集 Smooth L1 损失保存的最佳模型；
- `history.json`：训练/验证损失；
- `metrics_test.json`：原始量纲下逐任务、逐预测步和等权总体指标；
- `predictions_test.npz`：测试目标、预测值及目标时间。

静态门控模型还会额外生成 `gate_matrix.json`，其中行是目标任务、列是来源任务，对角线固定为0。

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

后续实现顺序：动态状态编码器 → 动态对称门控 → 动态有向门控 → 预实验、消融和正式实验。
