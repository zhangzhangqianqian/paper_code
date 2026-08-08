# Stage 7.7 PLE-lite 与 Stage 8 多任务联合比较设计

日期：2026-08-08

## 1. 设计目标

本次修订不改变 Scheme2R-v1 的网络结构，也不覆盖已经完成的 Stage 6-R、Stage 7-R 正式结果。修订目标是：

1. 将多任务联合模型作为论文的主要性能比较对象；
2. 补充 Hard-Share MTL、Dynamic Symmetric 和 PLE-lite 的正式多随机种子结果；
3. 将结构匹配 STL-H4 限定为负迁移诊断参照；
4. 将验证集最优 STL-H3 限定为选模过程审计记录，不进入正文主排行榜；
5. 重构 Stage 8，使联合模型比较、负迁移分析、消融分析和门控分析相互分离；
6. 修复 Stage 6 排序实现与 `0.1` 个百分点近似并列契约不一致的问题，但不追溯修改原冻结结果。

## 2. 不变研究协议

- 数据集：Kitakyushu Energy Station Data；
- 任务顺序：`electricity, cooling, heating, gas`；
- 输入窗口：过去 24 小时；
- 预测范围：未来 4 小时；
- 输出形状：`[batch, 4, 4]`；
- 全年协议：2015—2019 训练、2020 验证、2021 测试；
- 小样本协议：沿用冻结的 Kitakyushu 小样本时间切分；
- 标准化：仅使用对应训练切分拟合 z-score 参数；
- 未来外生变量：禁止使用；
- 正式随机种子：2026、2027、2028、2029、2030；
- 测试集不得用于模型选择、超参数选择或早停。

## 3. 两层 PLE-lite 结构

### 3.1 输入

历史负荷与历史外生变量分别为：

\[
X\in\mathbb{R}^{B\times24\times4},\qquad
R\in\mathbb{R}^{B\times24\times12}.
\]

PLE-lite 复用 MMoE-lite 的输入边界，将二者拼接后展平：

\[
z=\operatorname{Flatten}([X,R]).
\]

PLE-lite 不读取未来外生变量。

### 3.2 第一层 CGC

第一层包含：

- 2 个共享专家；
- 每个任务 1 个任务专属专家，共 4 个；
- 4 个任务门控；
- 1 个共享门控。

每个任务门控只在本任务专属专家和 2 个共享专家之间分配权重。共享门控可在全部共享专家和全部任务专属专家之间分配权重。第一层输出 4 个任务表示和 1 个共享表示。

### 3.3 第二层 CGC

第二层继续使用 2 个共享专家和每任务 1 个专属专家。第二层接收第一层的任务流和共享流，进一步分离任务专属信息与共享信息。最终层只输出 4 个任务表示，不继续输出共享流。

### 3.4 预测头

每个任务使用独立预测头，将 32 维任务表示映射为未来 4 个预测步。四任务结果按固定顺序组合为 `[batch, 4, 4]`。

### 3.5 固定结构与训练参数

| 参数 | 数值 |
|---|---:|
| CGC 层数 | 2 |
| 每层共享专家数 | 2 |
| 每任务每层专属专家数 | 1 |
| 专家隐藏维度 | 32 |
| 表示维度 | 32 |
| 预测头隐藏维度 | 16 |
| 激活函数 | GELU |
| Dropout | 0.1 |
| 学习率 | 0.001 |
| Weight decay | 0.0001 |
| 全年最大 epoch | 100 |
| 全年早停 patience | 12 |
| 小样本最大 epoch | 200 |
| 小样本早停 patience | 20 |
| 小样本 batch size | 32 |

该实现命名为 `PLE-lite`，定义为面向多能源负荷预测的轻量两层 PLE 适配，不声称复现推荐系统场景下的完整原始 PLE。

## 4. Stage 7.7 多任务基线追加实验

### 4.1 模型与配置来源

| 模型 | 正式配置来源 |
|---|---|
| Hard-Share MTL | Stage 6 全年验证集最优 H2 |
| Dynamic Symmetric | Stage 6 全年验证集最优 H1 |
| PLE-lite | 本设计预先固定配置 |
| MMoE-lite | 复用已验收 Stage 7.5 正式结果 |
| Scheme2R | 复用已验收 Scheme2R-H4 正式结果 |

不依据任何测试集结果修改上述配置。

### 4.2 新增正式运行矩阵

Hard-Share-H2、Dynamic-Symmetric-H1 和 PLE-lite 分别运行：

\[
3\text{ models}\times2\text{ protocols}\times5\text{ seeds}=30\text{ runs}.
\]

追加实验使用独立输出目录和独立机器可读契约，不修改 Stage 6.6 原冻结文件，不覆盖 Stage 7.3—7.6 结果。

### 4.3 运行前验收

正式运行前必须通过：

- PLE-lite 输出形状测试；
- 专家与门控数量测试；
- 任务门控权重和为 1 的约束测试；
- 任务门控不可访问其他任务专属专家的隔离测试；
- 有限梯度测试；
- 无未来外生变量测试；
- Hard-Share、Dynamic Symmetric 和 PLE-lite 两套协议 CPU 冒烟训练；
- 30 条正式运行矩阵 dry-run；
- 测试集未参与选模的契约检查。

## 5. Stage 8 分析结构

### 5.1 多任务联合模型主比较

正文主表只比较：

- Hard-Share-H2；
- Dynamic-Symmetric-H1；
- MMoE-lite；
- PLE-lite；
- Scheme2R-H4。

每个协议报告 MAE、RMSE、WAPE、参数量、训练时间和 CPU 推理时间。学习模型报告 5 个随机种子的均值与标准差。以 Scheme2R 为目标模型，对其与各联合基线的配对差异执行预先规定的统计分析。

主要输出：

- `joint_model_comparison.csv`；
- `joint_model_significance.csv`。

SOFTS、DLinear、Persistence 和 Seasonal Naive 作为通用或朴素预测基线保留，但与多任务联合模型分组展示，不将其错误标记为多任务学习模型。

### 5.2 单任务模型的限定角色

`STL-H3` 只作为 Stage 6 验证集最优单任务配置的审计记录，写入输入索引或实验协议附录，不生成独立正文性能比较表，也不进入多任务主排行榜。

`STL-H4` 是与 Scheme2R-H4 结构配置匹配的单任务参照，仅用于任务级迁移收益和负迁移分析。

### 5.3 负迁移分析

对误差越小越好的指标，定义：

\[
G_i=
\frac{E_i^{\mathrm{STL-H4}}-E_i^{\mathrm{Scheme2R-H4}}}
{E_i^{\mathrm{STL-H4}}}\times100\%.
\]

其中 `G_i < 0` 表示任务 `i` 相对结构匹配 STL-H4 出现负迁移。分析同时覆盖任务整体和任务—预测步粒度。

主要输出：

- `transfer_task_overall.csv`；
- `transfer_task_horizon.csv`；
- `negative_transfer_rates.csv`。

### 5.4 消融、门控和资源分析

继续分析：

- A0—A4 消融结果；
- Scheme2R 门控强度与非对称性；
- 门控与误差改善的相关性；
- `scheme2r_loads_only` 输入消融；
- 参数量、训练时间和 CPU 推理开销。

主要输出：

- `ablation_comparison.csv`；
- `gate_summary.csv`；
- `gate_asymmetry.csv`；
- `gate_error_association.csv`；
- `resource_comparison.csv`。

门控分析只应用于实际具有相应门控张量的模型，不对 STL、DLinear 等模型伪造门控结果。

## 6. 指标与统计策略

### 6.1 主要指标

论文核心结论使用 MAE、RMSE 和 WAPE。MAPE 仅在真实值绝对值超过预设阈值的样本上计算，并同时输出 `MAPE_valid_count` 和 `MAPE_valid_fraction`。接近零的热负荷样本不得用于产生异常巨大的 MAPE 结论。

### 6.2 配对与 Bootstrap

所有模型比较必须先验证目标时间戳和真实值完全一致。置信区间采用 24 小时块 Bootstrap；正式分析默认 2000 次重复。多重比较校正在预先定义的协议和分析粒度族内分别执行。

测试集结果只用于冻结后的最终评价，不得用于重新排序超参数或更换模型配置。

## 7. Stage 6 排序一致性修复

Stage 6 契约规定：全年验证集四任务等权平均 WAPE 为主指标；差异不超过 `0.1` 个百分点时视为近似并列，依次使用最差任务 WAPE、负迁移率、参数量和训练时间决胜。

当前排序实现直接按原始 WAPE 排序，未应用容差。本次修复应：

1. 实现 `0.1` 个百分点近似并列逻辑；
2. 增加边界与决胜顺序测试；
3. 使用既有验证结果重新汇总，不重新训练；
4. 生成一致性审计，确认 STL-H3 仍为第一、Scheme2R-H4 仍为第二；
5. 不覆盖原 Stage 6.6 冻结文件。

## 8. 错误处理与最终验收

Stage 7.7 和 Stage 8 必须拒绝以下情况：

- 缺少协议、模型或随机种子；
- 运行重复；
- 候选配置与追加契约不一致；
- 目标时间戳或真实值不一致；
- 预测形状不是 `[N, 4, 4]`；
- 测试集被标记为参与选模；
- 非有限预测或非有限主要指标；
- PLE-lite 门控权重不满足概率约束；
- 正式输出缺少 manifest 或预期文件。

最终 Stage 8 manifest 必须明确记录：

- 多任务主比较模型；
- STL-H3 的审计角色；
- STL-H4 的结构匹配迁移参照角色；
- 输入目录和运行数量；
- 指标、Bootstrap、随机种子和多重比较策略；
- `training_started=false`；
- `test_used_for_selection=false`；
- 完整输出清单和完成状态。

## 9. 成功标准

本次修订完成的判据是：

1. 两层 PLE-lite 通过结构、梯度和 CPU 冒烟测试；
2. Stage 7.7 追加契约与 30 条运行矩阵通过 dry-run；
3. Hard-Share-H2、Dynamic-Symmetric-H1、PLE-lite 的正式结果能够按协议生成和续跑；
4. Stage 8 能同时读取既有正式结果与 Stage 7.7 新结果；
5. 正文多任务主表不再由 STL 主导；
6. Scheme2R-H4 与 STL-H4 的负迁移分析保持独立；
7. Stage 6 排序代码与契约一致，且原冻结选择不变；
8. Stage 8 正式输出通过 manifest 和文件级验收。
