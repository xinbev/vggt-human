# SMPL Temporal Stabilizer v2：对齐后序列的保守稳定器

## 1. 目标与边界

本模块不重新做视频 HMR，不估计世界坐标，不替代 NLF、HSI 或 TRSTR。它位于既有单帧链路**全部完成之后**：

```text
RGB -> NLF SMPL -> HSI scale -> TRSTR spatial alignment
    -> aligned per-frame SMPL sequence
    -> Temporal Stabilizer v2 (optional, offline)
    -> stable aligned SMPL sequence
```

目标只有两个：

1. 减少 root 位置和整体朝向的逐帧闪动；
2. 在不恶化单帧对齐精度的前提下，让最终 NLF/HSI/TRSTR 序列指标更好。

V2 的第一版**不修改 body pose 和 betas**。body pose 的真实快速动作最容易被误平滑；先把最影响视觉稳定性的 `transl` 和 `global orient` 做好，再决定是否扩展。

这也意味着当前 `TemporalSMPLRefiner` v1 不继续训练、不接入主链。它保留为独立实验记录，不删除。

## 2. 核心思想：观测与时序预测的受限融合

每一帧有两种候选答案：

```text
O_t: 当前帧的单帧观测
     = NLF + HSI + TRSTR 已对齐输出

M_t: 只依据邻帧运动估计得到的时序候选
     = 不能读取当前帧 O_t
```

最终不是自由预测新 SMPL，而是只在两者之间融合：

```text
X_final,t = O_t + alpha_t * (M_t - O_t)
```

其中：

- `X` 是 translation 或 root rotation；
- `alpha_t` 是 `[0, alpha_max]` 的融合比例；第一版 `alpha_max = 0.5`；
- `alpha_t = 0` 时严格保持当前单帧输出；
- `alpha_t = 0.5` 时最多只向时序候选移动一半，不能凭空产生大修正。

因此它的能力边界明确：只在单帧观测与邻帧运动明显矛盾时，把当前结果适度拉回时序一致的位置；它不能取代单帧检测器，也不能通过大幅改 pose "编造" 动作。

## 3. 模型内部构造

### 3.1 按 track 处理

输入仍遵循项目的 frame-major 结构：

```text
pose_6d       [B,S,Q,144]
transl_cam    [B,S,Q,3]
track_ids     [B,S,Q]
valid_mask    [B,S,Q]
```

先按 `track_id` 取出同一人。每个轨迹独立稳定，绝不在不同人之间混合 SMPL 参数。track gap、重复 ID 或视频边界时不融合，保持 `O_t`。

### 3.2 两个小分支，不使用全自由 Transformer residual

```text
邻帧 O_(t-k...t-1), O_(t+1...t+k)
          |
          +-- Motion Proposal TCN --> M_t

当前帧 O_t + deviation(O_t, M_t) + base confidence
          |
          +-- Fusion Gate MLP -----> alpha_t

O_t + alpha_t * (M_t - O_t) -> X_final,t
```

`Motion Proposal TCN` 是轻量的 1D temporal convolution：

- 翻译分支输入邻帧 `transl`、velocity、acceleration；
- root 朝向分支输入邻帧 root rotation 的相对旋转（rotation log）；
- 对目标帧 `t` 做中心遮挡，确保 `M_t` 不复制 `O_t`；
- 预测 `M_t`，即“如果不看第 t 帧，邻帧推断的合理状态”。

`Fusion Gate MLP` 的输入为：

```text
O_t
M_t
O_t - M_t
base confidence（若当前 NLF/TRSTR 能提供）
有效邻帧数量 / track gap
```

它只输出 `alpha_t`，没有任意 `delta_transl` 或 `delta_pose` head。

### 3.3 两类状态必须分开

| 状态 | V2 处理方式 |
| --- | --- |
| `transl_cam` | Euclidean 融合；仅由 `alpha_trans` 向 `M_trans` 移动 |
| global orientation | 在 SO(3) 上相对旋转插值，不能直接加 axis-angle |
| body pose | V2 固定 passthrough |
| `betas` | 固定 passthrough |

如果训练/推理已验证可靠的 camera-to-world 变换，translation 分支可以在 world / motion-compensated 坐标做稳定，再转换回 camera。没有明确验证前，V2 仅在当前一致的 `transl_cam` 坐标内工作，不猜测外参方向。

## 4. 训练设计

### 4.1 输入、target 和两阶段训练

训练 target 是 3DPW/EMDB GT SMPL。第一阶段仍可用合成单帧闪动，但输入要明确区分：

```text
O_t = GT_t + 模拟单帧误差
target = GT_t
```

所有帧都有扰动。只使用 small / medium 两档：

| 档位 | drift | jitter | outlier |
| --- | --- | --- | --- |
| small | 6 cm | 2.5 cm | 10 cm |
| medium | 12 cm | 5 cm | 25 cm |

第二阶段才用冻结的真实 NLF + HSI + TRSTR 在训练视频上的缓存输出替换 `O_t`，GT 仍为 target。这是最终改善真实 NLF 指标的必要步骤。

### 4.2 gate 的直接监督：避免 no-op 塌缩

V1 gate 没有 target，只靠 no-worse 间接约束，导致 `alpha=0` 成为安全解。V2 使用 GT 给 gate 一个明确的最优融合标签。

对于 translation，给定当前观测 `O_t`、邻帧候选 `M_t` 和 GT `G_t`：

```text
direction = M_t - O_t
alpha_oracle = clamp(
    dot(G_t - O_t, direction) / (||direction||^2 + eps),
    0,
    alpha_max,
)
```

含义：沿着“从单帧结果走向时序候选”的方向，GT 最希望走多远。真实快速动作导致 `M_t` 不可靠时，oracle 会自然接近 0；当前帧闪动而邻帧连贯时，oracle 才变大。

root orientation 对相对旋转的 log vector 使用同样的投影方式。

### 4.3 loss

```text
L_final:     X_final 对 GT 的 translation / root rotation 误差
L_alpha:     alpha 对 alpha_oracle 的 Huber loss
L_velocity:  final velocity 对 GT velocity 的误差
L_accel:     final acceleration 对 GT acceleration 的误差
L_no_worse:  只在 alpha_oracle > 0 的候选帧上启用
L_identity:  alpha_oracle = 0 时，约束 alpha 接近 0
```

不要再使用 "全样本都不许变差" 的强 no-worse。它在模型尚未学会修正时会逼迫 gate 永远关闭。

## 5. 验收指标与最小实验顺序

### E0：固定 64-window overfit

固定样本、固定 small 噪声。验收：

```text
translation final L1 < base L1
alpha 与 alpha_oracle 有明显相关性
alpha_oracle > 0 的帧，final improvement > 0
alpha_oracle = 0 的帧，final displacement 接近 0
```

未通过 E0 前不跑全数据。

### E0 当前实现

实现仅覆盖 translation，文件为：

- `vggt_omega/models/smpl_temporal_stabilizer.py`
- `vggt_omega/training/smpl_temporal_stabilizer_noise.py`
- `scripts/train/overfit_smpl_temporal_stabilizer_v2.py`
- `scripts/smoke/run_smpl_temporal_stabilizer_v2_e0.sh`

E0 从 3DPW 与 EMDB 的 train partition 各随机取 32 个窗口，组成固定 64-window batch；一次性加入固定 small translation corruption，并对同一 batch 训练 1000 step。它不读取图像、不调用 NLF/HSI/TRSTR，也不写入主链。

通过条件固定为：

```text
final translation L1 < base translation L1
且 mean improvement > 0.002 m
```

结果写入 `outputs/debug/smpl_temporal_stabilizer_v2_translation_e0/e0_summary.json`，并保存 `checkpoint_e0.pt`。失败时仅说明这个最小设计或训练目标仍有问题；不得直接进入全数据训练。

### E1：3DPW + EMDB synthetic

对 small、medium 分开评估，报告：

```text
base / final translation L1
root rotation geodesic error
velocity / acceleration error
improvement rate
oracle-positive recall
oracle-zero false-apply rate
```

### E2：真实单帧输出微调与回接

缓存 NLF -> HSI -> TRSTR 的真实连续输出，与 GT track 匹配。只有在真实输入上同时满足：

```text
final translation error <= base
轨迹 acceleration/jitter 下降
视觉检查没有真实动作被拉平
```

才在推理可视化链路加 `enable_smpl_temporal_stabilizer_v2` 开关。默认 false，保留 baseline 回退。

## 6. 借鉴 GVHMR 的范围

参考论文是 GVHMR（Shen et al., SIGGRAPH Asia 2024）。V2 只借鉴两个原则：

1. 用结构化中间运动量（velocity / stationary-like reliability）而非直接自由回归整段 SMPL；
2. 把时序结果作为受约束的后处理，而不是让网络任意改动作。

V2 不引入该论文的 Gravity-View、视觉主干、相机 VO、长序列 RoPE、脚接触 IK 或 world trajectory recovery；这些超出“稳定已对齐输出序列”的任务边界。

## 7. Pose E0：必须独立验收

translation E0 不能证明 body pose 可安全稳定。新增独立的 pose E0：所有 24 个 SMPL 关节都在 SO(3) 上处理，邻帧 proposal 对中心帧做严格遮挡，最终只允许在当前 rotation 与 proposal rotation 之间插值最多 50%。它不是 axis-angle 的直接加法平滑。

运行：

```bash
bash scripts/smoke/run_smpl_pose_stabilizer_v2_e0.sh
```

通过条件为：

```text
final mean joint geodesic error < base error
且 mean improvement > 0.01 rad
```

Pose E0 通过后仍不直接说明可以发布：下一关必须包含 clean windows、small/medium noise 和真实快速动作的 held-out E1，以验证真实动作没有被过平滑。
