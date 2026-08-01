# 面向负迁移缓解的轻量化状态相关动态任务共享多能源负荷预测

## 1. Introduction

随着电、冷、热、气等能源载体在综合能源系统（integrated energy system, IES）中的协同运行，准确的多能源负荷预测已成为供能计划、优化调度和需求响应的重要基础。相关综述系统总结了人工智能方法在综合能源系统多能源负荷预测中的应用与发展趋势[1]。针对数据量有限的综合能源系统，已有研究将神经网络、高斯过程与多任务学习结合，以同时利用多类负荷之间的互补信息[2]。面向用户级综合能源系统的研究则采用全局—局部注意力和多解码器结构，分别建模共享规律与任务差异[3]。与单一电力负荷预测不同，多能源负荷预测既要刻画每类负荷自身的时间规律，又要利用能源转换设备、气象条件、日历属性和用户行为形成的跨能源联系。分别训练多个独立模型可能遗漏任务之间可共享的信息，而将所有变量无差别地输入同一模型又可能掩盖不同负荷的特有动态。因此，多能源联合预测需要在利用共同规律与保留任务差异之间取得平衡。

这种平衡之所以困难，是因为多能源任务之间的关系具有状态依赖性和潜在的方向差异。已有研究通过动态耦合关系建模，刻画不同能源负荷之间随时间变化的联系[4]。考虑气象变化的联合预测研究进一步表明，天气条件会改变不同负荷之间的关联模式[5]。例如，冷负荷与电负荷可能在高温季节表现出更强的同步变化，热负荷的主导因素则可能在低温季节改变；工作日/周末等日历属性和气象突变也会重新塑造跨负荷关系。此外，某一类能源负荷向另一类能源负荷提供的信息价值未必具有对称性。例如，电负荷的历史状态可能有助于预测冷负荷，但冷负荷的历史状态对电负荷预测的帮助程度不一定相同。这意味着，联合预测的关键不只是增加输入变量，而是根据当前状态判断每个目标任务是否需要其他任务的信息、需要来自哪个任务的信息，以及应当接收多少信息。

近年来，多任务学习与时空注意力被用于同时提取多能源负荷的任务内时间依赖和任务间耦合，分层多任务学习就是其中的代表性方案[6]。图注意力多任务模型则进一步利用任务或能源节点之间的结构关系进行信息聚合[7]。与此同时，多变量时间序列研究开始关注如何在保留各变量自身时间规律的同时，有选择地利用变量之间的联系。iTransformer将每个变量的完整历史序列作为独立表示，并在此基础上学习不同变量之间的关系[8]。LIFT识别能够提前反映目标变量未来变化的领先变量，只引入具有预测价值的跨变量信息[9]。SOFTS则先将多个变量的信息汇总为全局核心表示，再将其传递给各个变量，从而以较低的计算成本实现变量之间的信息交互[10]。这些研究表明，跨任务信息具有预测价值，但完全独立与无条件混合都不是对所有数据和运行状态普遍最优的选择。

近期能源预测研究已经开始处理上述问题。动态耦合建模研究从长期趋势、周期变化和短期波动等时间尺度分析跨负荷关系，为状态相关的信息传递提供了启发[4]。多任务平衡研究则通过共享矩阵和梯度协调方法处理不同能源任务之间的优化冲突[11]。面向小样本场景的频率增强多任务混合专家网络进一步将缓解负迁移明确作为模型设计目标[12]。与依赖频率分解和多专家门控的现有方法不同，本文不引入频率分解或多专家网络，而是采用参数量受控的时间编码器和状态相关的定向任务门控实现选择性信息传递，并以结构匹配的单任务模型为参照直接评价各能源任务的迁移收益。因此，本文的研究重点不是重新提出动态关系建模或负迁移缓解，而是在同一框架内进一步考察三个相互关联的问题。第一，跨任务共享强度和来源选择能否根据当前样本的负荷、天气、日历状态及预测步长实时变化，并允许不同传递方向具有不同权重；第二，联合学习的平均误差改善是否掩盖了个别任务、特定季节或部分预测步长上的性能下降；第三，动态关系建模所带来的收益能否在受控的参数量和计算开销下实现。关于多任务资源分配的研究表明，不同任务从联合优化中获得的资源与收益可能并不均衡[13]。多任务线性预测研究也从变量协同建模角度说明了任务间收益差异的重要性[14]。如果联合模型在某一任务上的误差高于结构匹配的单任务模型，则该任务发生了负迁移。仅报告多任务平均误差，无法判断每个能源任务是否真正从信息共享中受益。

针对上述问题，本文拟构建一种轻量化状态相关动态任务共享框架。该框架首先使用感受野完整覆盖24小时历史窗口的深度可分离时间卷积编码器，分别提取电、冷、热、气任务的时间表示；随后将天气、日历和负荷状态与预测步嵌入结合，通过两级有向任务路由分别计算目标任务的总体共享强度和不同来源任务的相对分配，从而把“是否共享、从谁共享以及共享多少”分解为可学习决策。来源任务表示经过低秩有向投影后，以消息归一化和可控残差方式进入目标任务；当共享强度为零时，目标任务严格保留其独立预测路径。最后，任务—预测步专属预测头输出四类负荷未来1—4小时的预测结果。实验将设置结构匹配的单任务模型、固定共享多任务模型、原始动态门控模型和所提模型，并逐任务、逐季节和逐预测步长比较预测误差与相对迁移收益；同时报告参数量、理论计算量以及统一硬件条件下的训练和推理时间，以检验动态共享带来的收益是否具有合理的计算代价。

实验采用公开的 Kitakyushu Energy Station Data。该数据集记录了日本北九州学术研究园区能源站及建筑群的小时级供能、用能和气象信息，并公开于 Figshare[15]。本文固定使用 2015—2021 年数据，将电力需求、供冷需求、供热需求和能源站系统侧天然气消耗量定义为四个预测任务。其中，气任务由能源站燃气发动机、燃料电池、吸收式冷热机组和燃气锅炉等设备的天然气使用量构成，不解释为单栋建筑用户端燃气表负荷。数据论文说明 2011 年 3 月存在地震造成的缺口[15]；本地数据审计还发现 2014 年部分负荷字段异常，因此两段数据均不进入本文的正式实验范围。本文的 MMoE-lite 仅作为保留专家路由思想的轻量外部对照，不包含 Shao 等人模型中的 Frequency 和 STIM 模块，也不声称复现其完整 Frequency-STIM-MMoE 模型。

本文的主要贡献如下：

1. 将多能源联合预测表述为状态相关的定向任务共享问题。通过将联合模型与结构匹配的单任务模型进行逐任务比较并计算相对迁移收益，直接判断联合训练是否降低某类负荷的预测性能，以及动态门控能否缓解这种性能下降。
2. 构建由全窗口轻量时间编码器、状态与预测步表示、共享强度—来源分配两级有向路由、低秩消息投影和任务—预测步专属预测头组成的共享—专属框架，使跨能源信息传递能够随当前状态和预测距离变化，并在共享关闭时严格保留任务独立路径。
3. 建立同时覆盖平均精度、任务级负迁移、季节与预测步长稳定性以及计算开销的评价方案，用于区分联合预测的总体收益、个别任务受益情况和模型复杂度代价。

## 参考文献

[1] DUAN P, ZHAO X, HU J, et al. Multi-energy load forecasting incorporating AI algorithms: research status and trends in integrated energy systems[J]. Renewable and Sustainable Energy Reviews, 2026, 229: 116611. DOI: 10.1016/j.rser.2025.116611.

[2] ZHANG W, CAI Y, ZHAN H, et al. Multi-energy load forecasting for small-sample integrated energy systems based on neural network Gaussian process and multi-task learning[J]. Energy Conversion and Management, 2024, 321: 119027. DOI: 10.1016/j.enconman.2024.119027.

[3] KIM H J, KIM D, TAK H, et al. Global-local attention-enabled multiple decoder Transformer for multi-energy load forecasting in user-level integrated energy system[J]. Applied Energy, 2025, 396: 126255. DOI: 10.1016/j.apenergy.2025.126255.

[4] GU Z, SHEN Y, WANG Z, et al. Load forecasting model considering dynamic coupling relationships using structured dynamic-inner latent variables and broad learning system[J]. Engineering Applications of Artificial Intelligence, 2024, 133: 108180. DOI: 10.1016/j.engappai.2024.108180.

[5] HUANG N, REN S, LIU J, et al. Multi-task learning and single-task learning joint multi-energy load forecasting of integrated energy systems considering meteorological variations[J]. Expert Systems with Applications, 2025, 288: 128269. DOI: 10.1016/j.eswa.2025.128269.

[6] SONG C, YANG H, CAI J, et al. Multi-energy load forecasting via hierarchical multi-task learning and spatiotemporal attention[J]. Applied Energy, 2024, 373: 123788. DOI: 10.1016/j.apenergy.2024.123788.

[7] ZENG X, JI G, ZHOU Y, et al. Multi-load forecasting for integrated energy systems based on GAT-MTL[J]. The Journal of Engineering, 2025, 2025(1): e70050. DOI: 10.1049/tje2.70050.

[8] LIU Y, HU T, ZHANG H, et al. iTransformer: Inverted Transformers Are Effective for Time Series Forecasting[C/OL]//The Twelfth International Conference on Learning Representations. 2024[2026-07-24]. https://proceedings.iclr.cc/paper_files/paper/2024/hash/2ea18fdc667e0ef2ad82b2b4d65147ad-Abstract-Conference.html.

[9] ZHAO L, SHEN Y. Rethinking Channel Dependence for Multivariate Time Series Forecasting: Learning from Leading Indicators[C/OL]//The Twelfth International Conference on Learning Representations. 2024[2026-07-24]. https://proceedings.iclr.cc/paper_files/paper/2024/hash/b52b07a239a7afa155ca25cf17a55074-Abstract-Conference.html.

[10] HAN L, CHEN X Y, YE H J, et al. SOFTS: Efficient Multivariate Time Series Forecasting with Series-Core Fusion[C/OL]//Advances in Neural Information Processing Systems 37. 2024: 64145–64175. Available: https://proceedings.neurips.cc/paper_files/paper/2024/hash/754612bde73a8b65ad8743f1f6d8ddf6-Abstract-Conference.html.

[11] WANG C, WANG Y, ZIO E, et al. Short-term load forecasting method for integrated energy systems based on graph neural network and multi-task balance[J]. International Journal of Electrical Power & Energy Systems, 2025, 172: 111195. DOI: 10.1016/j.ijepes.2025.111195.

[12] SHAO Y, JI M, JIANG C, et al. A Frequency-enhanced Multi-task Mixture-of-Experts Network for Multi-energy Load Forecasting under Small-sample Scenarios[C]//2025 10th International Conference on Power and Renewable Energy. Piscataway: IEEE, 2025: 752–758. DOI: 10.1109/ICPRE67300.2025.11274001.

[13] BAN H, JI K. Fair Resource Allocation in Multi-Task Learning[C/OL]//Proceedings of the 41st International Conference on Machine Learning. PMLR, 2024, 235: 2715–2731[2026-07-24]. https://proceedings.mlr.press/v235/ban24a.html.

[14] NOCHUMSOHN L, ZISLING H, AZENCOT O. A Multi-Task Learning Approach to Linear Multivariate Forecasting[C/OL]//Proceedings of the 28th International Conference on Artificial Intelligence and Statistics. PMLR, 2025, 258: 2638–2646[2026-07-24]. https://proceedings.mlr.press/v258/nochumsohn25a.html.

[15] LIAO W, JIN X, RAN Y, et al. A twenty-year dataset of hourly energy generation and consumption from district campus building energy systems[J]. Scientific Data, 2024, 11: 1400. DOI: 10.1038/s41597-024-04244-6. Dataset DOI: 10.6084/m9.figshare.24978645.
