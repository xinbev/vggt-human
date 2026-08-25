# HSI Module Figure Explanation

本文档用于画论文中的 HSI 模块图。目标是把 `HSIRefinementHead` 究竟做了什么讲清楚：它不是一个普通后处理模块，也不是直接照搬 GRAFT，而是在 VGGT-Omega 的 scene/depth/token 表示上做的 geometry-grounded human-scene interaction refinement。

## 1. 一句话概括

HSI 模块接收 base SMPL、VGGT-Omega depth、camera prediction 和 scene patch tokens，围绕人体构造 24 个 body-anchored interaction tokens，让这些 token 探测局部场景几何，然后预测人体参数残差和场景深度 affine 校正：

```text
Base SMPL + VGGT depth/camera/tokens
  -> 24 body-anchored HSI tokens
  -> local scene probing
  -> HSI Transformer
  -> refined SMPL + human-scale scene depth calibration
```

论文图里的核心公式可以写成：

```text
D_hsi = s_hsi * D_vggt + b_hsi
```

其中 `s_hsi` 是 HSI 预测的 scene scale，`b_hsi` 是 HSI 预测的 scene depth bias。

## 2. HSI 在整条 pipeline 中的位置

HSI 位于 SMPLHead 之后，属于 refinement stage。

```text
RGB frames
  -> VGGT-Omega Aggregator
  -> CameraHead: pose_enc / camera
  -> DenseHead: depth
  -> SMPLHead: base SMPL
  -> HSIRefinementHead: refined SMPL + scene affine
```

它不替代 VGGT-Omega，也不替代 SMPLHead。它做的是：拿已有的 base human prediction 和 scene geometry，判断二者是否在 3D 空间里对得上，然后预测小幅修正。

## 3. HSI 的输入

### 3.1 来自 SMPLHead 的 base human state

HSI 需要 SMPLHead 先输出一版人体：

```text
pred_pose_6d        [B, S, Q, 144]
pred_poses          [B, S, Q, 72]
pred_betas          [B, S, Q, 10]
pred_transl_cam     [B, S, Q, 3]
pred_confs          [B, S, Q, *]
```

含义：

```text
B: batch size
S: sequence length / frame number
Q: human query number, current configs commonly use Q=20
```

这部分提供人体本身的初始状态：姿态、体型、相机坐标系平移和置信度。

### 3.2 来自 VGGT-Omega 的 scene geometry

HSI 使用 VGGT-Omega 的 scene 输出：

```text
depth               [B, S, H, W]
pose_enc            camera encoding, converted to intrinsics
aggregated tokens   cached VGGT-Omega token features
```

其中：

```text
depth: 用于根据人体锚点读取场景深度。
pose_enc: 用于把 3D 人体锚点投影到 depth map，也用于 depth pixel -> camera-space scene point。
aggregated tokens: 用于提供局部 scene feature，不只是 raw depth。
```

### 3.3 可选 temporal 输入

HSI forward 接口支持：

```text
track_ids
track_mask
```

但这部分只用于 optional temporal memory。当前论文主图不应把 temporal identity 画成核心模块。如果要画，只能用灰色虚线表示 optional extension。

## 4. HSI 的输出

HSI 输出两类东西：人体 refinement 和场景 calibration。

### 4.1 人体 refinement 输出

```text
hsi_refined_pred_pose_6d
hsi_refined_pred_poses
hsi_refined_pred_betas
hsi_refined_pred_transl_cam
```

这些是 base SMPL 经过 HSI 修正后的姿态、体型和平移。

### 4.2 场景 calibration 输出

```text
hsi_scene_scale
hsi_scene_depth_bias
```

它们作用在 VGGT-Omega 原始 depth 上：

```text
D_hsi = hsi_scene_scale * depth + hsi_scene_depth_bias
```

直观含义：人体有大致稳定的真实尺度，HSI 用人体和局部场景的几何关系去校正场景深度的尺度和偏移。

### 4.3 诊断/辅助输出

```text
hsi_contact_logits
hsi_anchor_depth_residual
hsi_per_query_scene_log_scale
hsi_per_query_scene_depth_bias
hsi_refine_gate
```

这些可以放在补充图或 debug 图中。主图可以只画 `contact logits` 作为 auxiliary signal，不要把它画成“contact 已完全解决”。

## 5. HSI 的核心步骤

### Step 1: 从 VGGT-Omega token 中构造 scene feature stream

代码从 VGGT-Omega Aggregator 的 cached layers 中取最近 4 层 patch tokens，并分别投影到 `hidden_dim / 4`，然后拼接成 HSI 使用的 scene features。

```text
last 4 cached VGGT patch-token layers
  -> linear projections
  -> concatenate
  -> scene_features
```

画图建议：

```text
VGGT-Omega patch tokens -> "multi-layer scene token features"
```

不要画成单纯 depth map。HSI 同时使用 depth 和 token features。

### Step 2: 从 base SMPL 生成 24 个 body anchors

HSI 会根据 base SMPL 的 pose、betas、translation 运行 SMPLLayer，得到 vertices 和 joints，然后构造 24 个人体锚点：

```text
21 non-root body joints
+ left hand center
+ right hand center
+ full-body anchor
= 24 body anchors
```

更具体地说：

```text
body anchors: joints[:, 1:22]
left hand center: average(joint 20, joint 22)
right hand center: average(joint 21, joint 23)
full-body anchor: average of deterministic FPS-selected body vertices
```

画图建议：

```text
在人体 mesh / skeleton 上画 24 个小点。
脚、膝、髋、手、躯干附近的点可以画得更明显。
最后一个 full-body anchor 可以画在人体中心。
```

### Step 3: 将 body anchors 投影到 depth map

每个 3D anchor 都在 camera coordinate 中。HSI 用 `pose_enc` 转出的 intrinsics，把每个 anchor 投影到 depth map 上：

```text
anchor_3d + intrinsics -> projected 2D pixel
```

然后 HSI 在该 pixel 位置读取 VGGT depth：

```text
projected pixel -> scene depth z_scene
```

再通过 intrinsics 把这个 pixel/depth 反投影成 camera-space scene point：

```text
(u, v, z_scene) -> scene point xyz
```

画图建议：

```text
人体锚点 -> 投影线 -> depth map 上的对应点 -> 反投影成 scene point。
```

### Step 4: 估计局部 scene normals 和几何残差

HSI 根据 depth map 的局部梯度估计 scene normal：

```text
depth gradients -> approximate scene normals
```

然后对每个 anchor 计算人体点和场景点之间的几何关系：

```text
offset = scene_point - anchor
distance = ||offset||
depth_residual = scene_point.z - anchor.z
```

这些量告诉 HSI：

```text
人体点是否在场景表面前后？
是否离场景太远？
是否可能浮空？
是否可能穿进场景？
```

画图建议：

```text
在一个局部放大 inset 里画：
anchor point、scene point、offset arrow、depth residual。
```

### Step 5: 可选 local-nearest scene probing

HSI 有两种 probe mode：

```text
projected: 直接使用 anchor 投影点处的 depth。
local_nearest: 在 anchor 投影点附近的局部窗口中搜索离 anchor 最近的 scene point。
```

`local_nearest` 用于处理 anchor 投影点不正好落在正确 scene surface 的情况。它会在一个局部 window 中读取多点 depth，反投影为 scene xyz，然后选离 anchor 最近的有效点。

画图建议：

```text
主图可以只画 projected probing。
如果要画 local_nearest，用一个小窗口网格表示 "search nearest local scene point"。
```

### Step 6: 构造每个 HSI token 的输入特征

每个 HSI token 对应一个 body anchor。它不是单一的关节点 embedding，而是拼接了人体状态和局部场景状态：

```text
HSI token input =
  global human parameters:
    pose_6d, betas, translation
  body-anchor geometry:
    anchor xyz
  projection:
    normalized projected uv
  local scene geometry:
    scene point xyz
    offset vector
    distance
    scene normal
    depth residual
```

代码中的 token input 维度是：

```text
144 pose + 10 betas + 3 translation
+ 3 anchor xyz
+ 2 projected uv
+ 3 scene point xyz
+ 3 offset
+ 1 distance
+ 3 scene normal
+ 1 depth residual
= 173 dims
```

然后通过一个 MLP 投影到 HSI hidden dimension：

```text
173-d geometric descriptor -> token_mlp -> 512-d HSI token
```

当前典型配置：

```text
24 HSI tokens per human query
hidden_dim = 512
```

画图建议：

```text
画一个 HSI token composition card：
[Human params] + [Anchor xyz] + [Projected uv] + [Scene xyz] + [Offset/distance] + [Normal/residual]
```

### Step 7: 为每个 anchor 收集局部 scene tokens

除了 depth 几何量，HSI 还从 VGGT-Omega patch token 中收集局部 scene token features。

流程：

```text
anchor projection -> patch-grid coordinate
patch-grid coordinate -> scene_window x scene_window neighborhood
local patch tokens -> local_scene_tokens
```

当前典型配置：

```text
hsi_scene_window = 3
```

也就是每个 anchor 周围取 `3 x 3 = 9` 个局部 scene patch tokens。

画图建议：

```text
在 depth/patch grid 上画一个 3x3 小窗口，箭头指向 HSI token 的 cross-attention memory。
```

### Step 8: HSI Transformer 做 anchor self-attention 和 scene cross-attention

HSI Transformer layer 包含三部分：

```text
1. self-attention among 24 body-anchor tokens
2. cross-attention from each body-anchor token to its local scene tokens
3. feed-forward network
```

直观解释：

```text
self-attention: 让身体各部位互相协调，比如脚的修正不能让膝盖/髋部变得不合理。
cross-attention: 让每个身体部位看自己附近的局部场景 geometry/token evidence。
FFN: 更新 token 表示，用于预测 residual。
```

当前典型配置：

```text
hsi_num_layers = 5
hsi_num_heads = 8
hsi_num_iters = 3
```

画图建议：

```text
24 body tokens -> HSI Transformer block
local scene tokens -> cross-attention arrow into HSI Transformer block
```

### Step 9: 从 HSI tokens 预测 residual updates

Transformer 后，HSI 对 24 个 token 做 mean pooling，得到每个 human query 的 pooled HSI feature。然后预测：

```text
pose_delta     -> refine pose_6d
betas_delta    -> refine betas
transl_delta   -> refine camera-space translation
scale_delta    -> per-query scene log scale
bias_delta     -> per-query scene depth bias
contact_logits -> per-anchor contact probability logits
```

代码中的 residual 缩放：

```text
pose update:  pose += gate * 0.01 * pose_delta
betas update: betas += gate * 0.01 * betas_delta
transl update: transl += gate * 0.05 * transl_delta
```

`gate` 是 optional delta gate。如果 `hsi_use_delta_gate = false`，gate 直接等于 1。

输出 head 使用 zero-last initialization，因此训练初期 HSI 更接近 no-op refinement，不会一开始大幅破坏 base SMPL。

画图建议：

```text
HSI Transformer -> residual heads:
  Delta pose
  Delta shape
  Delta translation
  Scene scale/bias
  Contact logits
```

### Step 10: 将 per-query scene affine 聚合成 per-frame scene affine

每个 human query 都会预测自己的 scene log scale 和 depth bias。最后 HSI 用 `pred_confs` 做 confidence-weighted average，聚合成每帧一个 scene scale/bias：

```text
per-query log_scale, bias
  -> confidence-weighted average over Q humans
  -> per-frame hsi_scene_scale, hsi_scene_depth_bias
```

scale 的计算：

```text
scene_scale = exp(clamp(log_scale, -3, 3))
```

画图建议：

```text
多个 human queries -> weighted average -> frame-level scene scale/bias。
```

这部分是 human-scale metric calibration 的关键：人体不是只被场景修正，人体也反过来给场景 depth 提供尺度锚点。

## 6. HSI 的可选 temporal memory

代码中有 temporal HSI 分支：

```text
enable_temporal_momentum = true
```

当 sequence length 大于 1 时，HSI 可以逐帧处理并维护 memory。memory 存储：

```text
previous HSI tokens
previous contact probabilities
previous refined translation
previous scene scale/bias
```

memory key 优先使用 `track_id`；如果没有有效 track_id，则 fallback 到负 query index：

```text
valid track_id -> memory key = track_id
otherwise -> memory key = -(query_idx + 1)
```

当前画论文主图时建议：

```text
不要把 temporal memory 画成核心模块。
如需展示，把它画成灰色虚线 optional extension。
```

原因：当前论文最稳妥主张是 geometry-grounded HSI refinement 和 human-scale scene calibration；temporal identity / kinematic temporal refinement 仍应谨慎表达。

## 7. HSI scene affine 的后处理选择

HSI 原生预测的是 per-frame scale/bias。视频推理时可以选择：

```text
per_frame: 使用每帧 HSI 原始预测
clip_median: 使用整段 clip 的 median scale/bias，减少帧间抖动
ema: 使用 EMA 平滑 scale/bias
```

画图时可以不画这一步。如果要画，建议画成 scene affine 之后的小型 stabilizer：

```text
per-frame scale/bias -> optional clip median / EMA -> calibrated depth
```

## 8. 训练时 HSI 被哪些信号约束

图中不需要画所有 loss，但理解上 HSI 主要被以下几类信号约束：

### 8.1 人体监督

```text
hsi refined pose
hsi refined betas
hsi refined translation
hsi refined joints
hsi refined vertices
hsi projected 2D joints
```

作用：确保 HSI refinement 不只是让人贴近场景，也不能破坏人体姿态和体型。

### 8.2 场景 depth / affine 监督

```text
hsi_depth_teacher
hsi_teacher_scene_affine
scene scale/bias temporal smoothness
```

作用：约束 `D_hsi = sD + b` 后的 depth 更接近 GT 或 teacher depth。

### 8.3 Anchor / contact 相关监督

```text
hsi_anchor_depth
hsi_anchor_scene_xyz
hsi_contact
foot contact / sole contact / support plane contact
```

作用：让人体锚点和局部场景几何更一致，减少浮空和穿插。

### 8.4 稳定性约束

```text
hsi_delta_reg
hsi_no_worse
hsi_gate_reg
velocity / acceleration smoothness
```

作用：防止 HSI 修正过猛，或把 base prediction 修坏。

## 9. 适合放进论文图的模块名称

推荐命名：

```text
Geometry-Grounded HSI Refinement
Body-Anchored HSI Tokens
Local Scene Probing
HSI Transformer
Human Residual Heads
Human-Scale Scene Affine
```

不推荐命名：

```text
GRAFT module
Contact solver
Physics engine
Tracker refinement
Post-processing
```

## 10. 主图应该画什么

建议主 HSI 图从左到右画 5 个部分：

```text
1. Base SMPL + VGGT Scene
2. 24 Body Anchors
3. Local Scene Probing
4. HSI Transformer
5. Refined Human + Calibrated Scene
```

箭头关系：

```text
Base SMPL -> Body Anchors
VGGT depth/camera -> Project anchors into scene
VGGT patch tokens -> Local scene tokens
Body anchors + local scene -> HSI tokens
HSI tokens -> HSI Transformer
HSI Transformer -> human residuals
HSI Transformer -> scene scale/bias
```

主图中最该强调的双向关系：

```text
scene helps refine human:
  local depth / scene tokens guide pose and translation updates

human helps calibrate scene:
  human-scale geometry predicts depth scale and bias
```

## 11. 不应该画什么

不要画成：

```text
1. HSI = 普通 SMPL 后处理。
2. HSI = 只预测 contact。
3. HSI = 直接把 GRAFT 原代码接进来。
4. HSI = temporal tracker。
5. HSI = 只修 depth，不修人体。
6. HSI = 只修人体，不修 scene scale。
7. HSI = 使用外部 tracker ID 才能工作。
```

更准确的说法是：

```text
HSI is a VGGT-Omega-native geometry refinement head that uses body-anchored local scene probes to jointly refine human states and calibrate scene depth.
```

## 12. 图注候选

英文：

```text
Geometry-grounded HSI refinement. Starting from base SMPL predictions, OmegaHSR constructs 24 body-anchored interaction tokens for each human query. Each token projects its body anchor into the VGGT-Omega depth map, probes local scene geometry and patch-token features, and encodes body-scene residuals. An HSI Transformer performs body-token self-attention and local scene cross-attention, then predicts residual updates for SMPL pose, shape, and translation, together with a human-scale affine calibration for scene depth.
```

中文：

```text
HSI 几何细化模块。给定 base SMPL，OmegaHSR 为每个人构造 24 个身体锚点 token。每个 token 将人体锚点投影到 VGGT-Omega 深度图中，探测局部场景点、法线和 patch-token 特征，并编码人体与场景之间的几何残差。HSI Transformer 通过身体 token 自注意力和局部场景 cross-attention 预测人体姿态、体型和平移残差，同时输出用于场景深度校正的 human-scale affine 参数。
```

## 13. 代码依据

主要代码：

```text
vggt_omega/models/heads/hsi_refinement_head.py
```

相关辅助：

```text
vggt_omega/models/vggt_omega.py
vggt_omega/utils/hsi_affine.py
vggt_omega/training/hungarian_losses.py
configs/train_smpl_hsi_refine.yaml
```

