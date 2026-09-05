# RSC-PF 外部基线 v4.6 协议适配设计

## 目标

在不修改 RSC-PF 主模型的前提下，将 iTransformer-PTO、DecisionFocused-Online 和 DigitalTwins-Policy 统一到当前 formal v4.6 的因果数据协议，并先完成短校准，再决定是否启动五种子正式验证。

## 现状与边界

- 当前 formal v4.6 可用材料是 `train`、`early_stop` 和 2019 `selection_full` 窗口；旧 `reports/joint_forecast_dispatch_v1/data` 不存在。
- 当前设备历史是 17 维连续设备出力，另有 6 维活动/启停状态；不能把它误写成 21 维历史设备输出。
- 21 维只用于未来调度输出和教师标签，17 维用于历史输入。
- 预测任务保持电、冷、热、气四项；气作为站侧燃气先验，不作为刚性终端平衡负荷。
- 不读取封存测试集，不把 2019 选择窗口用于训练或参数拟合。

## 方案

### 数据适配

新增外部专用窗口容器，不改动核心 `JointWindowSplit`。适配器从 formal v4.6 NPZ 构造：

- `load_history`: `[N,24,4]`
- `exog_history`: `[N,24,12]`
- `device_history`: `[N,24,17]`
- `device_status`: `[N,24,6]`
- `scheduler_context`: `[N,4,6]`，由 PV/WT 预测、价格/碳权重和重复到四个 horizon 的初始 SOC 组成
- `forecast_target`: `[N,4,4]`
- `teacher_dispatch`: `[N,4,21]`，仅训练标签；缺失时由同一 LP 协议离线生成
- `oracle_first_step_objective`: 仅用于评估标签，不进入模型输入

训练归一化只从 `train` 拟合。`early_stop` 只用于短校准和验证选择；`selection_full` 只在验证通过后作为 2019 Pilot 评价窗口。

### 三个基线

1. **iTransformer-PTO**：官方 iTransformer 代码的明确适配版本，保留倒置 token 预测结构；预测后调用统一滚动 LP。
2. **DecisionFocused-Online**：预测器按决策导向损失训练，评价时调用 LP；不把它描述成无优化器端到端策略。
3. **DigitalTwins-Policy**：历史状态直接产生连续控制量，经统一物理解码器得到 21 维调度；评价时 exact LP 调用数必须为零。

### PTO 接口

补齐 `src/joint_dispatch/pto.py`，提供统一的预测结果容器、滚动 LP 求解、缓存读写和季节朴素预测接口。PTO 求解必须记录成功率、窗口数和 exact LP 调用数。

## 训练与评价顺序

1. 先通过协议测试和每个方法的极小窗口短校准。
2. 若任一方法出现形状、因果性、非有限损失、物理不可行或优化器角色错误，停止，不启动正式五种子训练。
3. 三个方法均通过短校准后，再进行 5 个种子、20–30 epoch 的验证训练。
4. 验证完成后，在 2019 `selection_full` 上统一评价，并与 RSC-PF/Fair Decoupled 汇总；不提前打开封存测试集。

## 成功标准

- 所有外部输入输出形状与 v4.6 协议一致。
- 未来标签改变不会改变模型输入。
- iTransformer-PTO 和 DecisionFocused-Online 的 LP 调用角色可审计；DigitalTwins-Policy 为零 exact LP 调用。
- 所有短校准损失和输出有限，物理解码的结构性残差有限；短校准不把“零负荷缺口”误当作物理解码器必须保证的条件，缺口由正式调度指标评价。
- 生成的 receipt 明确记录数据来源、适配级别、分割、种子、LP 调用和 `test_set_accessed=false`。
