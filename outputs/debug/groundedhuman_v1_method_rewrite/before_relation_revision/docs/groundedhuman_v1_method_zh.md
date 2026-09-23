# GroundedHuman v1：重写后方法的中文对照稿

对应正文：`.paper/my_paper/versions/v1/tex/03_method_v7.tex`。用于讨论与逐段审阅；正式论文仍以 v1 英文 LaTeX 为准。方法图由作者重新设计，本稿不依赖旧图。

## 方法概述

给定单目视频 \(\mathcal I=\{\mathbf I_t\}_{t=1}^{T}\)，GroundedHuman 在共享的米制世界坐标系中恢复场景几何、相机运动和人体网格。我们的核心思想是将人体作为**读取场景的结构化接口**：人体的米制尺寸为场景校准提供参照，其空间构型则决定哪些场景观测与人体放置相关。我们通过查询与注意力编码这些人体—场景关系，并从中读出相应的几何修正。交互隐含在这些关系及其产生的更新中，无需中间的接触标签预测。

这一原则形成两个模块。**Body-Anchored Metric Calibration（人体锚定的度量校准）**通过 **Metric Alignment Query（度量对齐查询）**关联人体锚点与场景特征，恢复共享的度量参照。**Dynamic Interaction Grounding（动态交互定位）**通过 **Placement-Conditioned Interaction Attention（以当前放置为条件的交互注意力）**，根据人体周围经过校准的几何推断平移修正。

两者的执行顺序针对一个关键歧义：相同的局部深度差，可能来自场景尺度错误，也可能来自人体位置错误。我们先校准公共参照，再将剩余差异作为位置精修的证据。人体移动后，在更新后的位置重新构建注意力上下文。

### 初始观测

冻结的 VGGT-Ω 提供相对 Z-depth \(D_t^r\)、相机内参 \(\mathbf K_t\)、外参 \((\mathbf R_{cw,t},\mathbf t_{cw,t}^r)\) 和多层场景特征 \(\mathbf F_t\)。在共享内参下，冻结的 NLF 为第 \(q\) 个人提供米制 SMPL 观测：姿态 \(\boldsymbol\theta_{tq}\)、形状 \(\boldsymbol\beta_{tq}\)、初始根平移 \(\boldsymbol\tau_{tq}^{(0)}\) 以及置信度 \(c_{tq}\)。

记 \(\overline{\mathbf V}_{tq}\) 为施加姿态、尚未加入根平移的顶点，则当前相机坐标系人体为

\[
\mathbf V_{tq}^{c,(k)}=\overline{\mathbf V}_{tq}+\boldsymbol\tau_{tq}^{(k)}.
\]

平移对所有顶点广播。这些估计提供人体的米制先验；两个模块均保留人体姿态与形状。

## 1. Body-Anchored Metric Calibration

度量校准回答：场景几何应该如何表达在人体提供的度量参照中？人体表面对应初始化这一关系，结构化的人体查询随后读取场景特征，估计剩余修正。

### 人体表面初始化

将初始人体顶点投影到相对深度图，获得同一相机射线上的一对深度。汇总多个人体的有效深度比：

\[
s_t^c=\operatorname{median}_{(q,i)\in\mathcal A_t}
\frac{[\mathbf V_{tqi}^{c,(0)}]_z}
{D_t^r(\pi(\mathbf K_t,\mathbf V_{tqi}^{c,(0)}))},
\qquad D_t^{\mathrm{coarse}}=s_t^cD_t^r.
\]

其中 \(\pi\) 表示透视投影，\(\mathcal A_t\) 保留通过投影与深度筛选的对应；投影重合时保留最近的人体表面采样。中位数减弱孤立误差的影响。采样细节及观测缺失时的处理见论文附录。

### Metric Alignment Query

人体锚点同时指定**去哪里读取场景证据**，以及**相对于人体怎样解释这些证据**。每个人构造 24 个锚点 token，编码人体状态、锚点位置和投影，以及从粗校准深度取得的局部场景观测，包括三维位置、法向、相对偏移、距离和深度残差。每个锚点在场景特征中读取一个 \(3\times3\) 邻域。

Self-attention 首先汇集身体锚点之间的证据，然后各锚点查询自己的场景邻域。省略帧与人物索引：

\[
\begin{aligned}
\widetilde{\mathbf H}^{(l)}&=\mathbf H^{(l)}+\operatorname{SelfAttn}(\operatorname{LN}(\mathbf H^{(l)})),\\
\mathbf Q_j&=\operatorname{LN}(\widetilde{\mathbf h}_j^{(l)})\mathbf W_Q,\quad
\mathbf K_j=\mathbf S_j\mathbf W_K,\quad
\mathbf U_j=\mathbf S_j\mathbf W_V,\\
\mathbf o_j&=\operatorname{softmax}(\mathbf Q_j\mathbf K_j^{\mathsf T}/\sqrt d)\mathbf U_j.
\end{aligned}
\]

最后一行展示维度为 \(d\) 的单个注意力头，\(\mathbf U_j\) 表示 value，以区别于人体顶点。多头输出、残差连接与前馈模块更新锚点 token。最终响应经过池化，预测残余对数尺度 \(\Delta\ell_{tq}\) 和深度偏置 \(\Delta b_{tq}\)。解析初始化承担主要的尺度对齐；查询以人体—场景差异为条件，读取估计残差所需的场景信息。

### 共享度量读出

使用归一化的 NLF 置信度汇总人物级预测，得到帧级残余尺度 \(\rho_t\) 与偏置 \(b_t\)。假设输入片段具有公共场景尺度，跨帧聚合得到

\[
\begin{aligned}
s^{\mathrm{vid}}&=\exp[\operatorname{median}_{t}\log(s_t^c\rho_t)],&
b^{\mathrm{vid}}&=\operatorname{median}_{t}b_t,\\
D_t^m&=s^{\mathrm{vid}}D_t^r+b^{\mathrm{vid}},&
\mathbf t_{cw,t}^m&=s^{\mathrm{vid}}\mathbf t_{cw,t}^r.
\end{aligned}
\]

加性偏置仅作用于深度，相机旋转与内参保持不变。根据内参反投影得到相机坐标系的米制点 \(\mathbf P_t^m\)，供后续位置精修使用。

## 2. Dynamic Interaction Grounding

度量参照建立后，交互定位回答：人体应如何根据周围环境改变放置位置？人体移动会改变邻近观测、几何残差及其相关性。因此，我们构建依赖当前位置的交互表示，从中读出平移更新，再在新位置重新计算该表示。

### Placement-Conditioned Interaction Attention

将 SMPL 表面划分为 96 个对应区域，每个区域由中心点和 8 个采样顶点表示。第 \(k\) 次更新时，每个区域从四种空间支持域采集场景点：固定 \(3\times3\) 与 \(7\times7\) 窗口、随投影区域范围变化的自适应窗口，以及外围环形区域。这些支持域同时提供细粒度对齐线索和更大范围的环境结构。

每个样本通过相对当前人体区域的三维偏移、图像偏移和深度残差进行编码。将场景深度与带人物实例标识的人体深度缓冲比较，分离自身表面与环境证据。其他人体样本从两类注意力通道中排除，仅作为干扰统计保留。

记 \(\mathbf z_{ri}^{(k)}\) 为样本 \(i\) 相对区域 \(r\) 的编码，\(\mathcal N_{rsc}^{(k)}\) 为支持域 \(s\) 和通道 \(c\in\{\mathrm{self},\mathrm{env}\}\) 内的有效样本集合，则

\[
\mathbf u_{rsc}^{(k)}=
\sum_{i\in\mathcal N_{rsc}^{(k)}}
\frac{\exp(a(\mathbf z_{ri}^{(k)}))}
{\sum_{j\in\mathcal N_{rsc}^{(k)}}\exp(a(\mathbf z_{rj}^{(k)}))}
\mathbf z_{ri}^{(k)}.
\]

\(a\) 为学习得到的标量评分，空集合返回零 token。四种支持域乘以两类通道，形成每区域 8 个上下文 token，并保留有效支持比例。关系编码与采样集合均依赖当前位置，因此相同场景会随着人体移动提供不同的交互证据。

### 从交互证据到人体位置

将上下文 token 与区域几何、身体区域嵌入、支持比例及干扰统计融合，得到区域表示 \(\mathbf h_r^{(k)}\)。每个区域预测有界平移建议 \(\mathbf v_r^{(k)}\)、门控 \(g_r^{(k)}\) 和标量对数方差 \(\ell_r^{(k)}\)。建议沿区域的视线方向及两个切平面方向表达，再转换到相机坐标系；每个建议都作用于整个身体。

用 \(m_r^{(k)}\) 表示是否有有效几何支持：

\[
\alpha_r^{(k)}=m_r^{(k)}g_r^{(k)}\exp(-\ell_r^{(k)}),\qquad
\widehat{\mathbf v}^{(k)}=
\frac{\sum_r\alpha_r^{(k)}\mathbf v_r^{(k)}}
{\max(\sum_r\alpha_r^{(k)},\varepsilon)}.
\]

这一读出优先采用具有有效支持且可靠的证据。人物级门控进一步控制有界根平移更新：

\[
\begin{aligned}
\Delta\boldsymbol\tau_{tq}^{(k)}&=g_{tq}^{\mathrm{person},(k)}
\operatorname{clip}_{[-d_{\max},d_{\max}]^3}(\widehat{\mathbf v}_{tq}^{(k)}),\\
\boldsymbol\tau_{tq}^{(k+1)}&=\boldsymbol\tau_{tq}^{(k)}+\Delta\boldsymbol\tau_{tq}^{(k)}.
\end{aligned}
\]

两次前馈更新共享参数，每次更新后重建区域位置、样本归属与注意力上下文。这里的 **dynamic** 指依赖人体放置状态的重新计算。场景几何、相机参数、人体姿态和形状均保持固定。模型通过人—场景关系支持的位移学习交互，不显式分类接触，也不进行测试时优化。

### 世界坐标系重建

外参采用 \(\mathbf X^c=\mathbf R_{cw,t}\mathbf X^w+\mathbf t_{cw,t}^m\) 约定。最终人体网格为

\[
\mathbf V_{tq}^{w}=\mathbf R_{cw,t}^{\mathsf T}
(\overline{\mathbf V}_{tq}+\boldsymbol\tau_{tq}^{(2)}-\mathbf t_{cw,t}^m),
\]

该变换逐顶点应用。相同逆变换将场景点映射到世界坐标系，使人体与场景共享同一米制参照。

## 训练目标

两阶段独立训练，VGGT-Ω 和 NLF 均冻结。度量校准使用带米制监督的序列与合成尺度扰动，先进行尺度训练，再在解析初始化产生的状态上监督剩余修正：

\[
\mathcal L_{\mathrm{calib}}=
\lambda_s\mathcal L_{\mathrm{scale}}+
\lambda_b\mathcal L_{\mathrm{bias}}+
\lambda_d\mathcal L_{\mathrm{depth}}.
\]

三个损失分别监督残余对数尺度、深度偏置和校准深度。在残差头训练阶段，锚点交互主干冻结，仅训练尺度与偏置头。该尺度扰动训练路径的偏置目标为零，用来约束模型对乘性错误采用加性补偿。

固定校准模块后，在正确状态、尺度扰动、平移扰动和混合状态上训练动态交互定位：

\[
\mathcal L_{\mathrm{ground}}=
\lambda_v\mathcal L_{\mathrm{vote}}+
\lambda_t\mathcal L_{\mathrm{trans}}+
\lambda_g\mathcal L_{\mathrm{gate}}+
\lambda_r\mathcal L_{\mathrm{reg}}+
\mathcal L_{\mathrm{state}}.
\]

区域建议的目标是剩余位移除以剩余更新次数。最终平移监督训练位置读出，门控监督训练更新有效性，正则项控制更新幅度。状态相关损失鼓励纠正较大错误、保留准确输入，并减少迭代退化。完整目标及训练配置状态见论文附录。
