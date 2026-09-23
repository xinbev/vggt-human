# GroundedHuman v1：隐式交互表示与几何读出（中文对照稿）

对应正文：.paper/my_paper/versions/v1/tex/03_method_v7.tex。本稿同步本轮思想整合、小标题恢复与公式精简。方法图由作者重绘，正文不依赖旧图。

## 方法概述

给定单目视频 \(\mathcal I=\{\mathbf I_t\}_{t=1}^{T}\)，GroundedHuman 在共享的米制世界坐标系中恢复场景几何、相机运动和人体网格。我们将人—场景定位建模为**隐式交互表示的学习，并从这些表示中推断度量校准与人体位置**。这些表示编码场景观测与人体米制尺寸、当前空间构型之间的关系。人体结构决定在哪里、以什么条件查询场景；注意力将相关证据整合为关系表示；几何读出将表示转换为修正量。其统一原则为：

\[
(\mathcal H,\mathcal S)
\xrightarrow{\text{Query / Attention}}
\mathbf Z_{\mathrm{int}}
\xrightarrow{\text{Geometric readout}}
(\text{Metric},\text{Placement}).
\]

其中 \(\mathcal H\) 和 \(\mathcal S\) 分别表示人体与场景观测。交互隐含于学习得到的关系特征 \(\mathbf Z_{\mathrm{int}}\) 及其支持的修正中，无需中间的接触标签预测。

我们用两个顺序执行的模块及其各自的表示和读出实现这一原则。**Body-Anchored Metric Calibration** 通过 **Body-Anchored Scene Query** 编码人体—场景对应关系，读出残余度量修正。**Dynamic Interaction Grounding** 通过 **Hierarchical Context Attention** 编码区域性的人体—环境关系，读出人体位置更新。

两个阶段的顺序针对一个歧义：相同的局部深度差，可能来自场景尺度错误，也可能来自人体位置错误。先建立公共度量参照，区域关系才具有用于位置推理的可比性。每次位置更新又改变这些关系，因此需要重新计算交互表示。

### 初始观测

冻结的 VGGT-Ω 提供相对 Z-depth \(D_t^r\)、内参 \(\mathbf K_t\)、外参 \((\mathbf R_{cw,t},\mathbf t_{cw,t}^r)\) 和多层场景特征 \(\mathbf F_t\)。共享相机内参的冻结 NLF 为人物 \(q\) 提供米制 SMPL 观测：姿态 \(\boldsymbol\theta_{tq}\)、形状 \(\boldsymbol\beta_{tq}\)、初始根平移 \(\boldsymbol\tau_{tq}^{(0)}\) 和置信度 \(c_{tq}\)。

记 \(\overline{\mathbf V}_{tq}\) 为施加姿态、尚未加入根平移的顶点，则当前相机坐标系人体为 \(\mathbf V_{tq}^{c,(k)}=\overline{\mathbf V}_{tq}+\boldsymbol\tau_{tq}^{(k)}\)，平移对所有顶点广播。这些观测提供人体的米制先验；两个模块均保留人体姿态与形状。

## 1. Body-Anchored Metric Calibration

第一个模块编码米制人体观测与相对场景几何之间的对应关系。人体表面的深度比建立粗参照，人体锚定查询随后形成关系表示，用于预测剩余度量修正。

### Analytic Scale Initialization

将初始人体顶点投影到 \(D_t^r\)，得到同一相机射线上的一对深度。汇总多个人体的有效深度比：

\[
s_t^c=\operatorname{median}_{(q,i)\in\mathcal A_t}
\frac{[\mathbf V_{tqi}^{c,(0)}]_z}
{D_t^r(\pi(\mathbf K_t,\mathbf V_{tqi}^{c,(0)}))},
\qquad D_t^{\mathrm{coarse}}=s_t^cD_t^r.
\tag{1}
\]

\(\pi\) 为透视投影，\(\mathcal A_t\) 保留通过投影与深度筛选的对应；投影重合时保留最近的人体表面采样。中位数减弱孤立误差的影响。采样与观测缺失时的处理见论文附录。

### Body-Anchored Scene Query

锚点同时指定**去哪里读取场景证据**，以及**相对于人体怎样解释证据**。每个人构造 24 个锚点 token，编码人体状态、锚点位置和投影，以及粗深度中的局部场景观测：三维位置、法向、相对偏移、距离和深度残差。每个锚点从多层场景特征中读取一个 \(3\times3\) 邻域，得到 \(\mathbf S_{tqj}\)。

自注意力关联身体不同锚点的证据，交叉注意力关联每个锚点及其局部场景上下文。记第 \(l\) 层锚点 token 为 \(\mathbf H_{tq}^{(l)}\)：

\[
\widetilde{\mathbf H}_{tq}^{(l)}
=\operatorname{SelfAttn}(\mathbf H_{tq}^{(l)}),
\qquad
\mathbf h_{tqj}^{(l+1)}
=\operatorname{CrossAttn}(\widetilde{\mathbf h}_{tqj}^{(l)},\mathbf S_{tqj}).
\tag{2}
\]

交叉注意力中，人体锚点提供 query，场景 token 提供 key 和 value。公式省略归一化、残差连接和前馈模块。得到的 token 编码以人体为条件的场景证据。池化最终响应，形成**度量交互表示** \(\mathbf z_{tq}^{\mathrm{met}}=\operatorname{Pool}(\mathbf H_{tq}^{(L)})\)。度量读出将其映射为残余对数尺度 \(\Delta\ell_{tq}\) 和深度偏置 \(\Delta b_{tq}\)。

这里明确建立了“人体—场景关联 → 交互表示 → 几何修正”的联系；解析初始化承担主要尺度误差，关系表示支持剩余修正。

### Temporal Aggregation

归一化的 NLF 置信度将人物级预测汇总为帧级残余尺度 \(\rho_t\) 和偏置 \(b_t\)。假设输入片段具有公共场景尺度，则：

\[
\begin{aligned}
s^{\mathrm{vid}}&=\exp[\operatorname{median}_{t}\log(s_t^c\rho_t)],&
b^{\mathrm{vid}}&=\operatorname{median}_{t}b_t,\\
D_t^m&=s^{\mathrm{vid}}D_t^r+b^{\mathrm{vid}},&
\mathbf t_{cw,t}^m&=s^{\mathrm{vid}}\mathbf t_{cw,t}^r.
\end{aligned}
\tag{3}
\]

偏置仅作用于深度，相机旋转与内参保持不变。根据内参反投影得到相机坐标系中的米制点 \(\mathbf P_t^m\)，作为后续位置精修的参照。

## 2. Dynamic Interaction Grounding

公共度量参照建立后，第二个模块编码人体与周围环境之间的空间关系，并读出平移修正。人体移动会改变相关观测与几何残差，因此交互表示依赖当前位置，并随位置精修而重新计算。

### Hierarchical Context Attention

将 SMPL 表面划分为 96 个对应区域，每个区域由中心点及 8 个采样顶点表示。第 \(k\) 次更新时，各区域从四种空间支持域采样：固定 \(3\times3\) 与 \(7\times7\) 窗口、随投影范围变化的自适应窗口，以及外围环形区域。这些支持域同时提供细粒度对齐线索与更大范围的环境结构。

每个样本通过相对当前身体区域的三维偏移、图像偏移和深度残差编码。将场景深度与带人物实例标识的人体深度缓冲比较，分离自身表面与环境证据。其他人体样本从两个注意力通道排除，仅保留为干扰统计。

带掩码的注意力在每个支持域、每类通道内选择和汇聚关系特征，得到每区域 8 个上下文 token 及其有效支持比例；空支持域返回零 token。将上下文与区域几何、身体区域嵌入及干扰统计融合，形成**区域交互表示** \(\mathbf z_r^{\mathrm{place},(k)}\)。

该表示编码身体与可见人体表面的一致性，以及身体与环境之间的空间关系，为位置推理提供互补证据。具体池化公式移至附录。

### Interaction-Aware Placement

位置读出将 \(\mathbf z_r^{\mathrm{place},(k)}\) 映射为有界平移建议 \(\mathbf v_r^{(k)}\)、门控 \(g_r^{(k)}\) 和标量对数方差 \(\ell_r^{(k)}\)，把隐式交互表示连接到具体的几何动作。

平移建议沿区域视线方向与两个切平面方向表达，再转换到相机坐标系。各区域的建议均作用于整个身体。用 \(m_r^{(k)}\) 表示有效支持：

\[
\alpha_r^{(k)}=m_r^{(k)}g_r^{(k)}\exp(-\ell_r^{(k)}),
\qquad
\widehat{\mathbf v}^{(k)}
=\frac{\sum_r\alpha_r^{(k)}\mathbf v_r^{(k)}}
{\max(\sum_r\alpha_r^{(k)},\varepsilon)}.
\tag{4}
\]

读出优先采用有有效支持且可靠的证据。对这些修正量的监督训练交互表示，无需显式接触标签。

### Translation Refinement

对聚合建议按相机坐标分量裁剪，再施加人物级门控，得到根平移更新 \(\Delta\boldsymbol\tau_{tq}^{(k)}\)，具体处理见附录：

\[
\boldsymbol\tau_{tq}^{(k+1)}
=\boldsymbol\tau_{tq}^{(k)}+\Delta\boldsymbol\tau_{tq}^{(k)}.
\tag{5}
\]

两次前馈更新共享参数。每次更新后，重建区域位置、样本归属和注意力上下文，得到新的 \(\mathbf z_r^{\mathrm{place},(k+1)}\)。这形成**关系编码与几何读出之间的循环**：更新改变人体—场景关系，下一次更新依赖这个新关系。

这里的 dynamic 指依赖人体位置的重新计算。场景几何、相机参数、人体姿态和形状均保持固定。最终使用经过校准的相机逆变换，将人体与米制场景点映射到共享世界坐标系；展开公式见附录坐标约定部分。

## 训练目标

两阶段独立训练，VGGT-Ω 与 NLF 冻结。度量校准使用带米制监督的序列与合成尺度扰动；初始尺度训练后，在解析初始化产生的状态上监督所需残差：

\[
\mathcal L_{\mathrm{calib}}
=\lambda_s\mathcal L_{\mathrm{scale}}
+\lambda_b\mathcal L_{\mathrm{bias}}
+\lambda_d\mathcal L_{\mathrm{depth}}.
\tag{6}
\]

三个损失监督残余对数尺度、偏置和校准深度。残差头训练阶段冻结锚点交互主干，仅更新尺度与偏置头。该尺度扰动路径中的偏置目标为零，约束针对乘性错误的加性补偿。

固定校准模块后，在正确状态、尺度扰动、平移扰动和混合状态上训练 Dynamic Interaction Grounding：

\[
\mathcal L_{\mathrm{ground}}
=\lambda_v\mathcal L_{\mathrm{vote}}
+\lambda_t\mathcal L_{\mathrm{trans}}
+\lambda_g\mathcal L_{\mathrm{gate}}
+\lambda_r\mathcal L_{\mathrm{reg}}
+\mathcal L_{\mathrm{state}}.
\tag{7}
\]

区域建议以剩余目标位移除以剩余更新次数作为监督。最终平移监督训练位置读出，门控监督训练更新有效性，正则项控制更新幅度。状态相关损失鼓励恢复较大误差、保留准确输入，并减少迭代退化。具体目标及训练配置状态见附录。

