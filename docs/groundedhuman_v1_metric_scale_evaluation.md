# B5：如何验证 Body-Anchored Metric Calibration 的尺度恢复

日期：2026-09-23。本文是评测设计及写作建议，**不是已经运行的新实验**。当前论文 Fig. 4(a) 的数值、图片与 Sim(3) 协议尚未替换。仅依据 v1 方法和项目现有评测实现分析，未检查附录。

## 结论

若要让 Fig. 4(a) 证明米制尺度校准有效，推荐将主指标改成 **SE(3)-aligned camera ATE (m)**，并增加 **残余尺度误差**。最关键的控制变量比较是：**解析人体尺度初始化 → 完整 Body-Anchored Metric Calibration**，两者使用完全相同的 VGGT-Ω 轨迹。原始未校准轨迹可作为诊断基线。

可以保留原 Sim(3)-ATE 作为轨迹形状和既有方法对比指标，但不能将其改善归因于统一米制缩放。新协议须用原始预测重新评估；若保存了预测相机中心与模型输出的校准尺度，通常只需重新算指标，不必重新训练或重新运行整个模型。

## 为什么原实验看不出尺度贡献

当前 v1 方法对相机保留旋转，只对平移施加一个 clip 共享尺度。令原始相机中心为 p_t，校准后为 ŝ p_t。Sim(3) 评估又从真值求最优尺度 a、旋转 R 和平移 b，实际比较：

`a R (ŝ p_t) + b` 与 GT。

因为 a 可以自由拟合，它会抵消 ŝ。只要原始轨迹非退化、共享尺度为正且关联帧相同，纯统一缩放前后的最优 Sim(3)-ATE 在数值精度内应相同。**算法有没有找到正确尺度，被评测本身抹掉了。**

例子：真值移动 10 m，预测只移动 1 m。Sim(3) 可以在评测时替预测乘 10，使尺度错十倍的轨迹仍得到很小的误差。SE(3) 仅允许转动和平移坐标系，不能替预测补这个比例。

代码证据：

- [TUM evaluate_ate.py:94](C:/Users/ROG/PycharmProjects/vggt-omega/benchmarks/tum_dynamics_ate/evaluate_ate.py:94)：`sim3_align` 显式拟合 scale。
- [TUM evaluate_ate.py:119](C:/Users/ROG/PycharmProjects/vggt-omega/benchmarks/tum_dynamics_ate/evaluate_ate.py:119)：`compute_ate` 固定调用此对齐函数；当前评测入口不是尺度保留的 ATE。
- [EMDB 可视化实现:253](C:/Users/ROG/PycharmProjects/vggt-omega/scripts/vis/plot_emdb_camera_trajectory_comparison.py:253)：已有 `fixed_scale=True` 的 SE(3) 和 `fixed_scale=False` 的 Sim(3) 两种路径，可供后续适配；它目前是 EMDB 可视化工具，不能直接当成已经完成的 TUM 新基准。

## 推荐评测设计

### 1. 保留米制尺度的相机误差：主指标

计算相机中心 C_t，与 GT 只拟合一个序列级刚体变换：

`ATE_SE3 = min_(R ∈ SO(3), b) sqrt(mean_t ||R C_t + b − C_t^GT||²)`。

这里没有可优化的缩放项。允许 R、b 消除重建坐标系的任意原点与方向；不得先做 Sim(3)、GT scale / shift 校准，再声称得到 scale-preserving ATE。单位为米，越小越好。

若预测使用 world-to-camera 外参，应先用 `C_t = −R_cw,t^T t_cw,t` 转成相机中心；GT 也用同一方向和单位。时间关联、有效帧、序列长度、聚合方式对所有变体保持一致。

SE(3)-ATE 同时受轨迹形状和尺度影响，因此必须比较**相同 backbone 输出的不同尺度处理**来隔离校准贡献；仅与不同模型横向比较不足以完成该归因。

### 2. 残余尺度误差：辅助指标

对已经校准的预测与 GT 另外求一个最优 Sim(3) 尺度 a*，**只把它作为诊断量，不把它用于主指标的轨迹修正**。理想情况下 a*=1，可报告：

`E_scale = |log(a*)|`，越小越好。

用对数可以对“差两倍”和“差一半”作对称评价。也可同时列 a*，直观看它是否更接近 1。此量是相对于 GT 轨迹的最佳统一尺度诊断，不等同于有独立真值的网络 scale 参数误差。

对于相同原始轨迹，若其对 GT 的最佳尺度为 s*，则可等价比较 `|log(ŝ / s*)|`。s* 仅用于评估，不得在模型输出中应用。

**近静止 / 近纯旋转片段**的相机平移不足以稳定识别尺度。TUM 包含 static / rpy 类型，建议保留它们的 ATE 结果，但将尺度诊断按预先固定、与模型无关的 GT 平移激励规则单列或标记 N/A，并报告样本数。不要用所有序列混合后的单一尺度数字掩盖退化情况。若 GT 平移几乎为零，深度和人体–场景尺寸是更合适的尺度验证载体。

### 3. 控制变量表与曲线

| 变体 | 相机轨迹处理 | 用途 |
|---|---|---|
| VGGT-Ω (uncalibrated) | 原始轨迹，不使用 GT 归一化 | 显示尺度歧义的诊断基线；其原始单位任意，不把“击败未校准输出”作为唯一贡献证据 |
| Analytic scale initialization | 人体表面解析尺度 + 相同跨帧汇总口径 | 最关键的校准基线 |
| Body-Anchored Metric Calibration | 解析初始化 + learned residual metric readout + 跨帧汇总 | 检验 Body-Anchored Scene Query 的增益 |

建议表列：`ATE_SE3 (m) ↓`、`E_scale ↓`；可另列 `ATE_Sim3 (m) ↓` 说明形状性能保持。若改 Fig. 4(a)，可保留输入帧数横轴，纵轴改为 `SE(3)-aligned ATE (m) ↓`，图例改为以上三种尺度处理。动态精修不改变相机，在此图中无需单独加一条“完整 GroundedHuman”来重复相同相机结果。

每个输入长度使用同一组帧、同一条 backbone 轨迹，再施加各变体的输出尺度；沿用当前逐序列计算后 macro-average 的方式，避免不同长度改变样本集却未说明。若添加外部方法，必须在相同 SE(3) 协议下重算，不能搬用已发表的 Sim(3)-ATE 数值。

## 深度是配套的第二条证据

Fig. 4(b) 若确实直接评估模型输出的 metric depth，使用无 GT scale/shift 的 Abs Rel 和 δ<1.25，本身可以支持米制深度质量。建议也补充“解析初始化 / 完整校准”的同源消融，避免只做跨模型比较。

当前项目同时存在两种 Bonn 入口，**不能只凭文件名推断现有曲线用了哪一种**：

- [evaluate.py:107](C:/Users/ROG/PycharmProjects/vggt-omega/benchmarks/bonn_depth/evaluate.py:107) 在 `alignment=scale` 时拟合 GT 尺度，在 `alignment=metric` 时不拟合；该基础入口默认 scale。
- [evaluate_curve.py:59](C:/Users/ROG/PycharmProjects/vggt-omega/benchmarks/bonn_depth/evaluate_curve.py:59) 的曲线入口默认 metric。
- 输出 JSON / CSV 记录 alignment、fitted_scale / applied_scale；应以当前论文图对应的日志和数据为准。

这不是认定 Fig. 4(b) 的声明有误，而是新实验设计中需要确认的具体开关。GT 拟合尺度及其再乘一个系数的诊断模式，均不能作为模型自行恢复尺度的结果。深度的 affine bias 与 scale 都会影响误差；若进一步声称“纯尺度”改进，可补充 scale-only 与 scale+bias 区分，而不能把所有深度收益都归给尺度。

## 推荐论文表述

以下是**新协议实际运行后**可采用的方法描述，不包含尚未获得的性能结论：

> **Metric Camera Trajectory Evaluation.** We evaluate camera trajectories in metric units using ATE after SE(3) alignment, which resolves the arbitrary rotation and translation of the reconstruction frame while preserving the predicted scale. No ground-truth scale correction is applied to the trajectories used for this metric. To isolate metric calibration, we compare the same VGGT-Ω trajectories before calibration, after analytic scale initialization, and after Body-Anchored Metric Calibration. We additionally report residual log-scale error on sequences with sufficient translational motion; the fitted scale is used only as an evaluation diagnostic.

配套图注候选：

> **Metric calibration across input lengths.** (a) Camera trajectory ATE after SE(3) alignment on TUM-Dynamic. All calibration variants share the same backbone trajectories, and alignment preserves their predicted scale. (b) Metric video-depth evaluation on Bonn without ground-truth scale or shift fitting.

只有在新结果确实支持时，结果段才可写：

> Compared with analytic initialization, the learned metric readout reduces scale-preserving trajectory error and residual scale error, supporting the benefit of body-conditioned scene queries for metric calibration.

如果残余尺度改善但 ATE 不明显，则应分别描述这两个观察，不能强行宣称二者都改进。若新协议尚未运行，当前 Sim(3) 结果仍应按轨迹形状/相机重建质量解释，不能把 caption 提前替换为上述新协议。

## 本轮边界

已分析现有实现并给出协议、控制变量和写作方案。本轮未修改 TUM / Bonn 评测代码，未运行服务器推理，未改 Fig. 4 数据或 caption。后续实施应在保留原 Sim(3) baseline 的基础上新增 SE(3) 评测选项，并按项目规范提供 `scripts/eval/` 下的服务器 `.sh` 入口；不覆盖原结果。
