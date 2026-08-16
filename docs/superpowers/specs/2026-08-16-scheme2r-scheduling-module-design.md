# Scheme2R 预测—调度双轨评估模块设计规格

**日期：** 2026-08-16  
**状态：** 双轨方案已获用户确认；本文件取代此前“仅按 2021 实测拓扑调度”的方案 A 规格  
**研究定位：** 保留真实多能源预测证据，同时在参数透明的标准综合能源系统中评价预测对滚动调度的下游价值

## 1. 设计结论

本研究不修改已经冻结并完成正式实验的 Scheme2R-H4 预测模型，而是在其后增加一个独立、可审计的预测—优化模块。完整研究由两条互补证据轨道组成：

1. **真实运行回放轨（Real-station replay，R 轨）**：使用 Kitakyushu Energy Station 的 2021 年实测负荷、气象、购电、光伏和设备燃气数据，评价预测误差对购电/购气申报及偏差结算的影响。该轨道用于证明真实数据相关性，不宣称 2021 实测系统具有充分的设备调度自由度。
2. **标准 IES 仿真调度轨（Standard-IES dispatch benchmark，S 轨）**：沿用 2021 年的真实电、冷、热负荷和气象时间轴，在明确标注为仿真的能源枢纽中配置电网、天然气、PV、WT、CHP、燃气锅炉、电制冷机、吸收式制冷机和电池储能，执行四小时滚动线性调度。该轨道用于检验不同预测模型能否降低运行成本、碳排放、弃能和供能不足。

双轨设计解决两个不同问题：R 轨回答“预测在真实站点购能回放中有什么价值”，S 轨回答“预测在具有可替代供能路径和跨时段状态的 IES 中能否改善调度”。两类结果必须分表、分图、分结论报告。

## 2. 研究边界与术语

### 2.1 本研究做什么

- 预测对象仍为 electricity、cooling、heating 和 station-side gas 四个小时级序列；
- 使用过去 24 h 预测未来 4 h；
- 采用顺序式 predict-then-optimize（PTO）框架；
- 对不同联合预测模型使用完全相同的调度器、参数和结算规则；
- 评价计划成本、实现成本、决策后悔、碳排放、可再生能源消纳、供能不足和求解可靠性。

### 2.2 本研究不做什么

- 不把调度损失反向传播到 Scheme2R，不声称端到端 decision-focused learning；
- 不因调度结果重新选择 Scheme2R 超参数或随机种子；
- 不把仿真 CHP、WT、BESS 等设备写成 Kitakyushu 2021 年真实运行设备；
- 不把 Scheme2R 的 gas 输出当作用户侧刚性天然气需求；
- 不使用 2021 年真实未来负荷、真实未来气象或真实未来可再生出力生成普通预测驱动决策；
- 不在缺少启停参数时加入最小开停机时间、启停成本等整数约束；
- 不把 R 轨的直接购能回放夸大为多设备协同优化。

## 3. 数据、年份与无泄漏协议

### 3.1 年份分工

| 年份 | 作用 | 是否允许用于模型/参数选择 |
|---|---|---|
| 2015—2019 | 预测器训练、归一化统计、负荷尺度和仿真设备容量的训练期统计 | 允许 |
| 2020 | 预测器验证、辅助模型选择、调度超参数与敏感性区间冻结 | 允许 |
| 2021 | 正式预测测试、R 轨回放、S 轨正式调度评价 | 禁止据此调参 |

数据集虽然覆盖多年，但 2021 年仍然重要，因为它是当前冻结协议中的独立测试年。2015—2020 用于训练、验证和参数冻结，不能与 2021 共同用于正式结论的模型选择。

### 3.2 数据角色

- **真实需求序列：** electricity、cooling、heating；
- **真实设备侧燃气序列：** gas，用于 Scheme2R 第四任务预测、R 轨购气申报和 S 轨外部一致性诊断；
- **真实气象序列：** 温度、湿度、太阳辐射、风速及日历变量；
- **真实运行序列：** 购电、PV、各燃气设备用气等，仅用于 R 轨回放和事后评价；
- **仿真参数：** S 轨设备容量、效率、价格和排放因子，必须来自公开来源或训练/验证期的预先规定缩放规则。

### 3.3 未来信息权限

普通预测器只能使用预测起点可获得的信息。若未来电价和日历在实际调度时可提前公布，则允许作为已知调度参数；未来天气不得使用真实观测值，必须使用历史可得特征、公开天气预报接口的冻结结果，或不依赖未来天气的透明基线预测。perfect-information oracle 使用真实未来值，但必须置于独立结果目录且只作为不可部署上界。

## 4. 预测器—调度器接口

Scheme2R 输出

\[
\widehat{\mathbf Y}_t\in\mathbb R^{4\times4},
\]

第一维为未来第 1—4 h，第二维顺序固定为 electricity、cooling、heating 和 gas。

### 4.1 S 轨的核心输入

S 轨直接使用

\[
\widehat{\mathbf d}_\tau=
\left[
\widehat P^{\mathrm e}_\tau,
\widehat Q^{\mathrm c}_\tau,
\widehat Q^{\mathrm h}_\tau
\right],
\qquad \tau=1,\ldots,4.
\]

gas 预测不构成第四个刚性需求平衡。它作为设备侧燃气先验，仅进入两项辅助分析：

1. 与优化得到的燃气采购量比较，计算 gas forecast–dispatch deviation；
2. 在“gas-prior”消融中，以有界软惩罚约束优化购气量偏离预测值，检验该先验是否有益。

gas-prior 不进入主结果，除非在 2020 验证期证明其方向稳定且不会掩盖设备物理平衡。

### 4.2 可再生能源可用出力

S 轨的 PV 和 WT 可用出力由 Kitakyushu 气象变量通过冻结的透明模型生成：

- PV 使用太阳辐射、额定容量和固定转换效率/温度修正；
- WT 使用风速、切入/额定/切出风速和分段功率曲线；
- 额定容量为仿真基准参数，不声称属于 2021 实测站点；
- 2021 普通运行使用预测气象或仅基于历史的可再生出力预测，真实未来气象只用于结算和 oracle。

### 4.3 R 轨的购能申报输入

R 轨将 electricity 与 gas 预测分别转化为未来 4 h 的购电和购气申报参考。PV 预测用于修正净购电申报：

\[
\widehat P^{\mathrm{grid}}_\tau
=\max\left(0,\widehat P^{\mathrm e}_\tau-widehat P^{\mathrm{PV}}_\tau\right).
\]

R 轨使用实际购电和实际设备燃气总量进行事后结算。cooling 与 heating 预测用于按季节和工况解释 gas 申报误差，但不在 R 轨中伪造不存在的设备选择。

## 5. S 轨标准 IES 拓扑

### 5.1 能源输入

- 外部电网；
- 城市天然气；
- 光伏；
- 风力发电。

### 5.2 转换和储能设备

- CHP：天然气同时转化为电和可用热；
- gas boiler：天然气转化为热；
- electric chiller：电转化为冷；
- absorption chiller：热转化为冷；
- battery energy storage system（BESS）：电能跨时段转移。

上述拓扑在综合能源系统/energy hub 调度研究中具有明确的电—气—热—冷耦合路径。引入 BESS 和 CHP 爬坡约束使四小时预测能够影响第一小时决策，避免优化退化为四个互不相关的单小时能量平衡。

### 5.3 为什么暂不加入更多设备

核心基准不加入 P2G、燃料电池、热储能、冷储能、需求响应和启停二进制变量。其原因不是这些技术不重要，而是当前论文的首要任务是建立可复现、可解释且 CPU 可求解的线性基准。额外设备只可作为后续扩展，不得与主实验同时无边界增长。

## 6. S 轨连续线性调度模型

时间步长为 \(\Delta t=1\ \mathrm h\)，滚动窗口为 \(\tau=1,\ldots,4\)。

### 6.1 主要决策变量

- \(P^{\mathrm{grid}}_\tau\)：购电功率；
- \(P^{\mathrm{PV,use}}_\tau,P^{\mathrm{PV,curt}}_\tau\)：PV 消纳和弃光；
- \(P^{\mathrm{WT,use}}_\tau,P^{\mathrm{WT,curt}}_\tau\)：WT 消纳和弃风；
- \(G^{\mathrm{CHP}}_\tau,G^{\mathrm{GB}}_\tau\)：CHP 和锅炉用气；
- \(P^{\mathrm{CHP}}_\tau,Q^{\mathrm{CHP}}_\tau\)：CHP 电、热出力；
- \(Q^{\mathrm{GB}}_\tau\)：锅炉热出力；
- \(P^{\mathrm{EC}}_\tau,Q^{\mathrm{EC}}_\tau\)：电制冷机耗电和制冷量；
- \(Q^{\mathrm{AC,in}}_\tau,Q^{\mathrm{AC}}_\tau\)：吸收式制冷机耗热和制冷量；
- \(P^{\mathrm{ch}}_\tau,P^{\mathrm{dis}}_\tau,E^{\mathrm B}_\tau\)：电池充电、放电和能量状态；
- \(Q^{\mathrm{dump}}_\tau\)：不可利用余热；
- 电、冷、热供能不足松弛变量。

### 6.2 设备转换关系

设 \(\kappa_g\) 为天然气低位热值换算系数：

\[
P^{\mathrm{CHP}}_\tau=\eta^{\mathrm{CHP}}_{\mathrm e}\kappa_gG^{\mathrm{CHP}}_\tau,
\qquad
Q^{\mathrm{CHP}}_\tau=\eta^{\mathrm{CHP}}_{\mathrm h}\kappa_gG^{\mathrm{CHP}}_\tau,
\]

\[
Q^{\mathrm{GB}}_\tau=\eta^{\mathrm{GB}}\kappa_gG^{\mathrm{GB}}_\tau,
\]

\[
Q^{\mathrm{EC}}_\tau=\mathrm{COP}_{\mathrm{EC}}P^{\mathrm{EC}}_\tau,
\qquad
Q^{\mathrm{AC}}_\tau=\mathrm{COP}_{\mathrm{AC}}Q^{\mathrm{AC,in}}_\tau.
\]

### 6.3 能量平衡

\[
P^{\mathrm{grid}}_\tau+P^{\mathrm{PV,use}}_\tau+P^{\mathrm{WT,use}}_\tau
+P^{\mathrm{CHP}}_\tau+P^{\mathrm{dis}}_\tau+s^{\mathrm e}_\tau
=\widehat P^{\mathrm e}_\tau+P^{\mathrm{EC}}_\tau+P^{\mathrm{ch}}_\tau,
\]

\[
Q^{\mathrm{CHP}}_\tau+Q^{\mathrm{GB}}_\tau+s^{\mathrm h}_\tau
=\widehat Q^{\mathrm h}_\tau+Q^{\mathrm{AC,in}}_\tau+Q^{\mathrm{dump}}_\tau,
\]

\[
Q^{\mathrm{EC}}_\tau+Q^{\mathrm{AC}}_\tau+s^{\mathrm c}_\tau
=\widehat Q^{\mathrm c}_\tau,
\]

\[
G^{\mathrm{buy}}_\tau=G^{\mathrm{CHP}}_\tau+G^{\mathrm{GB}}_\tau.
\]

### 6.4 储能与跨时段约束

\[
E^{\mathrm B}_{\tau+1}
=E^{\mathrm B}_\tau
+\eta_{\mathrm{ch}}P^{\mathrm{ch}}_\tau\Delta t
-\frac{P^{\mathrm{dis}}_\tau\Delta t}{\eta_{\mathrm{dis}}}.
\]

电池能量、充放电功率和初末状态受限；不采用同时充放电的二进制约束，而通过正电价、往返损耗、正退化成本和核心情景零弃能罚金抑制无意义循环，并在结果审计中检查同时充放电量。若仍出现超过数值容差的同时充放电，核心 LP 不得进入正式实验。CHP 采用连续爬坡约束，不加入启停状态。

### 6.5 目标函数

主目标为可审计的经济运行成本：

\[
\min J^{\mathrm{plan}}=
\sum_{\tau=1}^{4}\Delta t\left(
c^{\mathrm e}_\tau P^{\mathrm{grid}}_\tau
+c^{\mathrm g}_\tau G^{\mathrm{buy}}_\tau
+c^{\mathrm{deg}}(P^{\mathrm{ch}}_\tau+P^{\mathrm{dis}}_\tau)
+\sum_{k\in\{\mathrm e,\mathrm c,\mathrm h\}}c_k^{\mathrm{unserved}}s^k_\tau
\right).
\]

其中总弃能量另行定义为

\[
P^{\mathrm{curt}}_\tau
=P^{\mathrm{PV,curt}}_\tau+P^{\mathrm{WT,curt}}_\tau.
\]

核心经济目标不对弃能额外收费，以避免连续电池模型通过同时充放电虚假消纳可再生能源；弃能量仍作为主指标报告。碳排放作为独立主指标报告；带碳价目标作为预先冻结的敏感性情景，不将无单位统一的成本和排放直接相加。

## 7. 参数来源与缩放规则

### 7.1 证据优先级

1. 官方数据论文、Figshare 元数据和公开设备说明；
2. 同行评议的 IES/energy hub 调度论文；
3. 训练期统计形成的可复现尺度参数；
4. 若仍无可靠实价，则使用明确标注的 normalized benchmark cost，不报告为真实日元成本。

### 7.2 容量冻结规则

仿真容量不得根据 2021 测试结果调整。若参数证据台账没有给出更高优先级的公开基准，主配置使用以下确定性回退规则：

- 记训练期电、冷、热负荷 95% 分位数为 \(P_{95}^{\mathrm e},Q_{95}^{\mathrm c},Q_{95}^{\mathrm h}\)；
- CHP 额定电功率取 \(0.35P_{95}^{\mathrm e}\)；燃气锅炉额定热功率取 \(1.20Q_{95}^{\mathrm h}\)；
- 电制冷机和吸收式制冷机的额定制冷量各取 \(0.60Q_{95}^{\mathrm c}\)，二者合计提供 20% 备用；
- BESS 额定功率取 \(0.20P_{95}^{\mathrm e}\)，额定能量取额定功率的 4 h；
- PV 与 WT 额定容量分别缩放至训练期年用电量的 15% 和 10% 理论年发电渗透率；
- 上述容量比例统一做 \(-25\%/0/+25\%\) 敏感性分析，但主配置不按 2021 表现重选。

具体比例、效率、价格、排放因子和允许区间必须在参数证据台账中给出唯一基准值、来源、单位和敏感性范围。任何没有来源或规则的数字均不得进入正式配置。

## 8. 四小时滚动计划与实现结算

### 8.1 计划阶段

每个小时使用当时可获得的信息生成未来 4 h 的需求和可再生出力预测，求解 S 轨 LP，只执行第一小时的 CHP、电池和设备计划，然后向前滚动一小时。

### 8.2 实现阶段

真实第一小时负荷和可再生出力只在计划生成后进入结算。实现模型固定第一小时的慢速/状态相关决策，并允许购电、购气及可快速调节设备在冻结的再平衡范围内调整。上调和下调分别计费：

\[
c^{\uparrow}=\gamma^{\uparrow}c,\qquad
c^{\downarrow}=\gamma^{\downarrow}c,
\]

其中基准 \(\gamma^{\uparrow}>1\)、\(0\le\gamma^{\downarrow}<1\)，具体数值在 2020 验证期前冻结，并进行敏感性分析。这样可以避免“同价无限再平衡”使预测误差没有经济后果。

### 8.3 决策后悔

\[
\mathrm{Regret}
=J^{\mathrm{realized}}_{\mathrm{forecast}}
-J^{\mathrm{realized}}_{\mathrm{oracle}}.
\]

oracle 使用同一调度器和同一参数，只将未来负荷和可再生出力替换为真实值。它不参与普通模型排名，只提供下界。

## 9. 正式比较与消融

### 9.1 预测模型比较

主表比较同赛道联合模型：Scheme2R-H4、Dynamic Symmetric-H1、Hard-Share-H2、MMoE-lite 和 PLE-lite。Persistence、Seasonal Naive、matched STL-H4 和 oracle 放入辅助参照表。所有学习模型按五个正式随机种子逐种子进入调度，不挑选测试表现最好的种子。

### 9.2 调度消融

- D0：无 BESS，检验跨时段自由度；
- D1：无 CHP，仅网电+燃气锅炉+制冷机；
- D2：无可再生能源；
- D3：无碳价情景；
- D4：无 gas-prior 的核心模型与可选 gas-prior 扩展比较；
- D5：单小时调度与四小时滚动调度比较；
- D6：同价再平衡与非对称再平衡价格比较，仅用于说明结算机制的重要性。

### 9.3 指标

- planned cost、realized cost、regret；
- grid energy、gas purchase、CHP/boiler/chiller outputs；
- total carbon emissions；
- renewable utilization、curtailment；
- unmet electricity/cooling/heating；
- BESS throughput 和终端 SOC；
- gas forecast–dispatch deviation；
- solver success rate、最大约束残差、平均/95% 求解时间。

统计分析以日为区块进行 bootstrap，跨模型比较采用相同日期和相同随机种子配对；多重比较需校正。

## 10. R 轨真实运行回放

R 轨只使用真实存在的 2021 运行字段。购电和购气预测按照冻结的申报/偏差价格规则结算，并与 Persistence、Seasonal Naive、联合模型及 oracle 比较。其核心指标为：

- 购电和购气申报 MAE；
- 上调/下调偏差能量；
- 归一化偏差结算成本；
- 供冷季、供热季和全年分层结果；
- gas 预测误差与 cooling/heating 工况的关系。

R 轨不得输出“优化设备出力降低了成本”的表述；其结论限定为预测对真实站点购能计划和偏差暴露的影响。

## 11. 技术实现

- Python 3.9.25；
- NumPy、pandas、PyYAML、PyTorch（复用预测结果）；
- SciPy 1.13.x 的 `scipy.optimize.linprog(method="highs")`；
- pytest；
- CSV/JSON/NPZ 作为可追溯结果；
- 先实现连续 LP，不引入 CVXPY、Pyomo 或商业求解器。

在参数证据台账和最小求解器测试通过前，不安装依赖、不开始正式编码。

参数台账的每一行同时记录 `data_origin` 标签（`real`、`simulated` 或
`derived`）。R 轨实测序列和现场信息只能标记为 `real`；S 轨设备、价格、
排放与缩放规则标记为 `simulated` 或 `derived`，审计脚本据此阻止现场参数与仿真
参数静默混用。

## 12. 验收标准

- R 轨和 S 轨的目录、配置、表格及论文表述完全分离；
- 2021 普通调度不读取真实未来输入；
- S 轨每个设备、参数和单位都有来源或确定的训练期缩放规则；
- 所有能量平衡最大绝对残差小于 \(10^{-7}\)；
- 人工可行与不可行样例的求解状态符合预期；
- 扰动未来第 2—4 h 预测时，含 BESS/爬坡约束的部分窗口会改变第一小时决策；
- oracle 的实现成本不劣于同参数下的普通预测模型；
- 同时充放电、能量凭空产生、负购能量和容量越界均为零或在数值容差内；
- 论文明确区分真实数据、仿真设备、预测结果和调度结果；
- 不在正式调度完成前声称降低了成本或碳排放。

## 13. 设计依据

- Kitakyushu 数据、历史设备拓扑及其适用边界以 [Scientific Data 数据论文](https://www.nature.com/articles/s41597-024-04244-6) 与 [Figshare 数据记录](https://doi.org/10.6084/m9.figshare.24978645) 为首要依据。
- CHP、锅炉、电制冷机、吸收式制冷机、可再生能源和储能形成的 energy hub 拓扑与综合能源调度研究的常见结构一致，可参见 [Integrated Energy Micro-Grid Planning Using Electricity, Heating and Cooling Demands](https://doi.org/10.3390/en11102810) 和 [IES 协同调度模型综述](https://doi.org/10.3390/en17184718)。这些设备用于提供电—气—热—冷之间的替代路径和跨时段自由度，而不是重构 Kitakyushu 2021 的真实现场。
- 多步预测进入实时调度并通过滚动更新处理未来不确定性的研究依据包括 [Energy 2024, 300:131639](https://doi.org/10.1016/j.energy.2024.131639)。本研究首版仍采用确定性 PTO+LP，概率预测和鲁棒优化保留为后续扩展。
