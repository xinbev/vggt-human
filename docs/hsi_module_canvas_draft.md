# HSI Module Canvas Draft

这份是画图草稿，不是论文文字。你可以把它当成一张画布的元素清单，后续自己重新画成正式论文图。

## Canvas 1: HSI Refinement Main Figure

### 画布规格

```text
Canvas size: 1600 x 900
Reading direction: left -> right
Main blocks: 5
Color theme:
  Base human: orange
  VGGT scene/depth: blue
  HSI tokens/refinement: green
  Outputs: teal
  Optional temporal: gray dashed
```

### Block A: Base Inputs

位置：

```text
x=60, y=160, w=260, h=520
```

元素：

```text
A1. Base SMPL mesh / skeleton
    label: "Base SMPL"
    sublabel: "pose, shape, translation"

A2. VGGT depth map
    label: "VGGT-Omega Depth"

A3. VGGT patch-token grid
    label: "Scene Patch Tokens"

A4. Camera icon / intrinsics symbol
    label: "Camera / pose_enc"
```

画法：

```text
Base SMPL 放在上半部分，用橙色人体骨架。
Depth map 放在下半部分，用蓝色热力图或灰度深度图。
Patch-token grid 可以叠在 depth map 后面，用小方格表示。
Camera 放在左侧或 SMPL 和 depth 之间。
```

从 Block A 发出的箭头：

```text
Base SMPL -> Body Anchors
Depth + Camera -> Local Scene Probing
Patch Tokens -> Local Scene Probing
```

### Block B: Body-Anchored Tokens

位置：

```text
x=380, y=160, w=250, h=520
```

元素：

```text
B1. Human skeleton with 24 dots
    label: "24 Body Anchors"

B2. Anchor legend
    text:
      "21 body joints"
      "+ 2 hand centers"
      "+ 1 full-body anchor"

B3. Token chips
    24 small green circles or chips
    label: "HSI Tokens"
```

画法：

```text
在人体骨架上画一串小绿点。
从骨架上的点拉出到一排/一组 token chips。
```

建议强调：

```text
每个 token 对应一个身体锚点。
这些 token 是 HSI 的核心表示。
```

### Block C: Local Scene Probing

位置：

```text
x=680, y=120, w=310, h=600
```

元素：

```text
C1. Projection arrows
    anchor -> depth pixel

C2. Depth-map local window
    3x3 patch window
    label: "Local Scene Window"

C3. Scene point
    label: "scene xyz"

C4. Offset arrow
    anchor point -> scene point
    label: "offset / distance"

C5. Normal vector
    small arrow at scene point
    label: "normal"

C6. Depth residual marker
    vertical/depth-axis difference
    label: "z residual"
```

画法：

```text
做一个局部放大 inset：
左边是一个人体 anchor，右边是 depth map 上对应的局部窗口。
从 depth pixel 反投影到 3D scene point。
anchor 和 scene point 之间画一条红/紫色差值箭头。
```

图中小公式：

```text
r_z = z_scene - z_anchor
d = ||p_scene - p_anchor||
```

### Block D: Token Composition Card

位置：

```text
x=680, y=740, w=310, h=120
```

这是一个小卡片，用来解释一个 HSI token 由什么组成。

卡片内容：

```text
HSI token =
  human params
  + anchor xyz
  + projected uv
  + scene xyz
  + offset / distance
  + normal
  + depth residual
```

画法：

```text
用小 pill chips 拼接：
[theta,beta,tau] [anchor] [uv] [scene] [offset] [normal] [residual]
然后箭头指向 token_mlp。
```

注意：

```text
不要在主图里写 173-d，太工程。
可以在 supplementary 或 caption 里提。
```

### Block E: HSI Transformer

位置：

```text
x=1060, y=180, w=260, h=430
```

元素：

```text
E1. HSI Transformer block
    label: "HSI Transformer"

E2. Self-attention icon
    label: "body-token self-attn"

E3. Cross-attention icon
    label: "local scene cross-attn"

E4. Iterative refinement marker
    label: "x3 iterations"
```

画法：

```text
绿色大矩形。
上方输入 24 HSI tokens。
侧边输入 local scene tokens。
内部画两条小箭头：
  self-attn among anchors
  cross-attn to scene window
```

可以写：

```text
5 layers, 8 heads
```

如果图太挤，别写这些数字，把它们放 caption。

### Block F: Residual Heads and Outputs

位置：

```text
x=1380, y=140, w=180, h=620
```

元素：

```text
F1. Human residual heads
    Delta pose
    Delta shape
    Delta translation

F2. Scene affine heads
    scale s_hsi
    bias b_hsi

F3. Contact auxiliary head
    contact logits

F4. Refined SMPL output
    label: "Refined SMPL"

F5. Calibrated depth output
    label: "Calibrated Depth"
    formula: "D_hsi = sD + b"
```

画法：

```text
HSI Transformer 输出分成两条粗箭头：
  upper arrow -> Refined SMPL
  lower arrow -> Scene scale/bias -> Calibrated Depth
contact logits 用细虚线输出，标为 auxiliary。
```

核心视觉：

```text
Refined SMPL 应该比 base SMPL 更贴合地面/场景。
Calibrated depth 可以用颜色变化或尺度标尺表示。
```

## Canvas 2: Compact Paper Version

如果论文版面不够，可以压成 4 个大块：

```text
[Base SMPL + VGGT Scene]
      |
      v
[24 Body-Anchored Tokens]
      |
      v
[Local Scene Probing + HSI Transformer]
      |
      v
[Refined Human + Calibrated Scene]
```

每个块里只写一句：

```text
Base SMPL + VGGT Scene:
  pose, shape, translation + depth, camera, patch tokens

24 Body-Anchored Tokens:
  joints, hand centers, full-body anchor

Local Scene Probing + HSI Transformer:
  project anchors, sample scene, self/cross attention

Refined Human + Calibrated Scene:
  Delta theta, Delta beta, Delta tau, s_hsi, b_hsi
```

## Canvas 3: Token-Level Inset

这个 inset 可以放在 Figure 4 右下角。

### 元素布局

```text
One anchor token card:

┌──────────────────────────────────────────────┐
│ HSI Token k                                  │
├──────────────────────────────────────────────┤
│ Human state: theta, beta, tau                │
│ Body anchor: p_anchor                        │
│ Projection: u, v                             │
│ Scene probe: p_scene, normal                 │
│ Residual: p_scene - p_anchor, z difference   │
└──────────────────────────────────────────────┘
       |
       v
 token MLP -> z_k
```

### 适合说明的问题

```text
为什么 HSI token 是 human-scene token，而不是普通 body token？
因为它同时编码人体状态和局部场景几何。
```

## Canvas 4: Scene Calibration Inset

这个 inset 用来强调 human-scale metric calibration。

### 元素布局

```text
Multiple people / queries
  q1 -> log scale, bias
  q2 -> log scale, bias
  q3 -> log scale, bias
       ...
       weighted by confidence
       v
Frame-level scale and bias
       v
D_hsi = s_hsi * D_vggt + b_hsi
```

### 画法

```text
多个小人或者多个 query token 汇聚到一个 scale/bias 仪表盘。
仪表盘再指向 depth map。
```

### 图中文字

```text
Confidence-weighted human-scale scene affine
```

## Canvas 5: Optional Temporal Extension

只在需要时画，必须虚线。

### 元素布局

```text
Refined HSI tokens at frame t-1
  + contact
  + translation
  + scene scale/bias
       |
       v
Temporal memory
       |
       v
Frame t HSI tokens
```

### 视觉规则

```text
Use gray dashed box.
Label: optional temporal memory.
Do not put it in the main contribution path.
```

### 不要画

```text
external tracker ID -> fixed query slot
temporal module -> final result
```

## 推荐最终 Figure 4 构图

我建议正式论文里的 Figure 4 这样布局：

```text
Top row:
  Base SMPL + VGGT depth
    -> 24 body anchors
    -> local scene probing
    -> HSI Transformer
    -> refined SMPL

Bottom row:
  token composition card
    -> scene affine formula
    -> calibrated depth
```

优点：

```text
1. 上面讲 forward pipeline。
2. 下面讲 token 组成和 scene calibration。
3. reviewer 能快速看到 HSI 同时修 human 和 scene。
```

## 图中推荐短标签

```text
Base SMPL
VGGT Depth
Scene Patch Tokens
24 Body Anchors
Project to Depth
Local Scene Probe
Body-Scene Residual
HSI Tokens
Self-Attn across Body
Cross-Attn to Scene
Residual Heads
Refined SMPL
Scene Scale/Bias
Calibrated Depth
```

## 图中不推荐的标签

```text
GRAFT
Physics Solver
Contact Solver
Tracker
Post-processing
External ID
YOLO/SAM2
```

## Caption 草稿

```text
HSI refinement module. OmegaHSR first derives 24 body-anchored interaction tokens from the base SMPL prediction. Each token projects its body anchor into the VGGT-Omega depth map, probes local scene points, normals, and patch-token features, and encodes body-scene residuals. A lightweight Transformer performs body-token self-attention and local scene cross-attention. The resulting representation predicts residual SMPL updates and a human-scale affine correction for scene depth.
```

