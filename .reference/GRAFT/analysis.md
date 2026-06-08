# GRAFT 论文完整整理版

论文：**GRAFT: Geometric Refinement and Fitting Transformer for Human Scene Reconstruction**

本文件整合了当前目录下 `paper.pdf`、原始 `analysis.md`，以及本轮对话里补充讨论过的内容，包括：

- GRAFT 整体架构
- 24 个 HSI tokens 的含义
- GRAFT 使用的 backbone / foundation model
- MapAnything 和 VGGT 的关系
- MapAnything 双图联合推理
- MoGe-2 为什么只做尺度辅助
- geometry-only plug-in prior 如何接到 Human3R 后面
- OmniEraser 论文与仓库下载信息

相关本地文件：

- GRAFT 论文：`paper.pdf`
- 整理版笔记：`analysis_clean.md`
- 架构图：`graft_architecture_diagram.svg`
- OmniEraser 论文：`OmniEraser_paper.pdf`
- OmniEraser 仓库：`Omnieraser/`

---

## 1. GRAFT 解决什么问题

GRAFT 解决的是：**从一张 RGB 图片里同时恢复人体 3D mesh 和场景 3D geometry，并让人体和场景之间的接触关系更合理**。

传统方法的问题：

- 只估计人体时，人体可能漂浮在地面上。
- 人体和场景分开估计时，人体可能穿进椅子、墙、沙发等物体。
- 优化式方法能做出较好的 contact，但每张图都要慢慢优化，速度很慢。
- 纯 feed-forward 网络速度快，但缺少显式的人体-场景交互推理。

GRAFT 的核心想法是：

> 不在测试时手工最小化 contact / penetration energy，而是训练一个 Transformer，让它直接预测人体参数应该怎么修正。

也就是说，它把传统几何优化过程学成了一个前向网络。

---

## 2. 一句话理解 GRAFT

GRAFT 不是从零生成人体，而是先拿到一个粗糙的人体和场景，然后反复做 3 次 refinement：

1. 看当前人体和场景哪里接近、哪里穿透、哪里悬空。
2. 把这些几何关系编码成 24 个 HSI tokens。
3. 用 Transformer 推理身体各部位之间的关系。
4. 输出人体姿态、位置、体型和尺度的修正量。
5. 更新人体 mesh，再重新检查人体和场景的几何关系。

论文默认训练和推理都使用：

```text
T = 3 refinement steps
```

---

## 3. 整体流程

输入是一张有人和场景的 RGB 图片 `Ih`。

### 3.1 初始化人体和场景

GRAFT 自己不是从图片直接估计所有东西，它先调用几个外部模型：

1. **OmniEraser**：把图片里的人抹掉，得到只包含场景的图片 `Is`。
2. **MapAnything**：联合处理 `Is` 和 `Ih`，输出场景 pointmap、交互图 pointmap，以及两路视觉特征。
3. **NLF**：从原图 `Ih` 里估计初始人体 SMPL-X mesh。
4. **MoGe-2 / monocular depth estimator**：帮助恢复 metric scale。

初始化后的结果只是粗对齐，人体和场景仍然可能有悬空、穿透或尺度错误。

### 3.2 GRAFT 做迭代修正

每次迭代时，GRAFT 会拿当前人体 mesh 去场景点云里做查询：

- 对身体关节、手部、身体表面采样点，找最近的场景点。
- 计算人体点到场景点的相对位移。
- 读取场景表面法向量。
- 把这些局部几何信息编码成 HSI token。

然后 Transformer 根据这些 token 判断：

- 哪些部位应该接触地面或物体。
- 哪些部位正在穿透场景。
- 哪些身体部位之间的修正应该联动。
- 是否需要调整全局位置、姿态、体型或尺度。

### 3.3 输出修正后的人体

每一步 GRAFT 输出：

- body pose 的修正
- hand pose 的修正
- global orientation 的修正
- translation 的修正
- body shape 的修正
- uniform scale 的修正

最后通过 SMPL-X forward 得到 refined mesh。

---

## 4. 24 个 HSI tokens 是什么

**HSI = Human-Scene Interaction**，中文就是 **人体-场景交互**。

这里的 HSI 不是高光谱图像里的 Hyperspectral Imaging，而是指人体和三维场景之间的空间关系，比如：

- 脚踩在地面上
- 人坐在椅子上
- 手撑在桌子上
- 身体靠在墙上
- 腿穿进沙发里
- 人悬浮在地面上方

这些都属于 Human-Scene Interaction。

GRAFT 论文里的 **24 个 HSI tokens** 指的是：

> 不把整个人体 mesh 和整张场景点云都塞进 Transformer，而是把“人体和场景的交互状态”压缩成 24 个身体相关的信息块。

这 24 个 token 分成三类：

| token 类型 | 数量 | 表示什么 |
|---|---:|---|
| body joint tokens | 21 | 非 root 身体关节，每个关节一个 token |
| hand tokens | 2 | 左手一个、右手一个 |
| full-body token | 1 | 全身整体状态，包括全局朝向、平移、shape 和身体表面采样点 |

加起来：

```text
21 + 2 + 1 = 24
```

每个 token 可以理解成一个“身体观察点”。例如脚踝 token 会记录：

- 这个脚踝附近最近的场景点在哪里。
- 脚踝离地面多远。
- 地面的法向量是什么方向。
- 当前脚踝关节的旋转是什么。
- 这个部位是否可能应该接触地面。

所以 **HSI token** 可以理解为：

> 用一个 token 表示某个身体部位和周围场景的交互关系。

这样 Transformer 就可以推理：

- 脚是不是应该落地。
- 手是不是应该扶着桌子。
- 身体是不是穿进了椅子。
- 躯干和腿的姿态是否符合坐姿。
- 哪些身体部位的修正要一起发生。

---

## 5. GRAFT 的 Transformer 结构

GRAFT 自己的核心网络是一个轻量 Transformer。

| 项目 | 设置 |
|---|---|
| Transformer 层数 | 5 层 |
| hidden width | 512 |
| attention heads | 8 |
| 总参数量 | 约 16.2M |
| 主要 attention | HSI token self-attention + geometry-aware visual cross-attention |
| decoder heads | 两层 MLP，hidden dim 256，GELU |

每层主要做两件事：

1. **Self-attention**：让身体各部位 token 之间互相通信。例如脚和躯干的支撑关系、手和桌面的接触关系。
2. **Geometry-aware cross-attention**：每个身体 token 只关注自己 anchor 附近的 MapAnything 视觉特征，而不是全图乱看。

这种 cross-attention 是稀疏、局部、身体锚定的，因此适合做人和场景的接触修正。

---

## 6. GRAFT 用到的 backbone / foundation model

下面把“GRAFT 自己的主干”和“外部初始化模型 / baseline backbone”分开。

### 6.1 GRAFT 正式使用的模型

| 名称 | 类型 | 在 GRAFT 里的作用 | 是否是 GRAFT 自己训练的核心 |
|---|---|---|---|
| **GRAFT Transformer** | 内部 refinement backbone | 根据 HSI tokens 预测人体参数修正量 | 是 |
| **MapAnything** | geometry foundation model / visual feature backbone | 生成 `Ps`、`Ph`，并提供多层视觉特征 `Fs`、`Fh` | 否，作为外部模型使用 |
| **NLF** | pose foundation model | 初始化人体 SMPL-X mesh | 否，作为外部模型使用 |
| **OmniEraser** | image inpainting / human removal model | 从原图中去掉人体，得到 `Is` | 否，预处理模型 |
| **MoGe-2** | monocular geometry / metric depth estimator | 帮助恢复 metric scale | 否，尺度辅助模型 |
| **SMPL-X** | parametric human body model | 把 pose、translation、shape 参数变成人体 mesh | 不是神经 backbone |

### 6.2 只在评估或 baseline 中出现的模型

| 名称 | 出现场景 | 注意 |
|---|---|---|
| **CUT3R** | 某些 baseline 的 depth backbone | 不是 GRAFT 自己的 backbone |
| **Pi3** | 某些 baseline 的 depth backbone | 不是 GRAFT 自己的 backbone |
| **Human3R** | feed-forward baseline，也可接 GRAFT geometry-only refinement | 不是 GRAFT 主体 |
| **UniSH** | feed-forward baseline | 不是 GRAFT 主体 |
| **PhySIC / PROX / HolisticMesh** | 优化式或传统对比方法 | 不是 GRAFT 主体 |

---

## 7. MapAnything 和 VGGT 的关系

你问过：**MapAnything 的 backbone 是 VGGT 吗？**

结论：

> 不能简单说 MapAnything 的 backbone 是 VGGT。MapAnything 是独立的 universal metric 3D reconstruction model；VGGT 是 MapAnything 官方代码框架支持的一个可互换 external model / wrapper。

更准确地说：

- GRAFT 论文引用和使用的是 **MapAnything**。
- MapAnything 负责输出 pointmaps 和多层视觉特征。
- VGGT 在 MapAnything 官方仓库中是一个可接入模型 key / wrapper。
- 因此 **MapAnything ≠ VGGT**。

可以在笔记里这样写：

```text
MapAnything: 独立的 universal metric 3D reconstruction transformer。
注意：MapAnything repo 支持 VGGT 作为 external model wrapper，
但 GRAFT 论文中引用的是 MapAnything，不是 VGGT。
```

相关链接：

- MapAnything arXiv: https://arxiv.org/abs/2509.13414
- MapAnything GitHub: https://github.com/facebookresearch/map-anything

---

## 8. MapAnything 双图联合推理是什么意思

GRAFT 论文里写的是：

```text
(Ps, Ph, Fs, Fh) = MapAnything(Is, Ih)
```

也就是说，按论文语义是 **两张图一起进入 MapAnything 做联合推理**，不是一张一张独立推理后再后处理。

输入两张图：

- `Is`：去掉人体后的场景图。
- `Ih`：原始人-场景图。

输出四个东西：

| 输出 | 含义 | 用途 |
|---|---|---|
| `Ps` | human-removed scene pointmap | 作为主要场景几何，用于 Geometric Probes |
| `Ph` | original interaction image pointmap | 用于人体初始 metric alignment |
| `Fs` | `Is` 对应的 MapAnything features | scene stream visual features |
| `Fh` | `Ih` 对应的 MapAnything features | interaction stream visual features |

关键点是论文说 `Ps` 和 `Ph` 在 **shared camera frame** 里。

如果单独跑：

```text
Ps = MapAnything(Is)
Ph = MapAnything(Ih)
```

两个结果可能各自有自己的尺度、深度偏移、坐标对齐误差。后续 GRAFT 既要用 `Ps` 做几何 probe，又要用 `Ph` 做人体初始对齐，如果二者坐标系不一致，就会很麻烦。

联合推理的直观理解：

```text
Ih 原图:
  有人，有人体和场景交互的真实视觉证据

Is 去人图:
  人被抹掉，可以补出人体遮挡住的背景、地面、椅子等结构

MapAnything 联合处理:
  把两张图的几何放到同一个 camera frame

GRAFT 使用:
  Ps 用作主要场景几何
  Ph 用作人体初始 metric alignment
  Fs/Fh 用作 Transformer cross-attention 的视觉特征
```

所以答案是：

> 论文语义上是 `Is` 和 `Ih` 一起输入 MapAnything 联合推理；不是两个独立推理结果再拼起来。

当前 GRAFT 仓库还没有源码，因此具体 API 形式无法从代码核对，但论文正文和算法伪代码表达的是 joint inference。

---

## 9. MapAnything 特征具体怎么用

论文补充材料写得比较具体：GRAFT 从 MapAnything 取 **4 个层级的视觉特征**：

1. post-ViT output
2. alternating transformer 中的一个 intermediate activation
3. alternating transformer 中的另一个 intermediate activation
4. final output

每个层级的特征先线性投影到 128 维，4 个层级拼接后变成 512 维，再通过一个 `512 -> 512` 的线性层。

GRAFT 用两路 MapAnything features：

- **scene stream**：来自去人后的场景图 `Is`
- **interaction stream**：来自原始人-场景图 `Ih`

对每个 HSI token，GRAFT 会定义 visual anchor：

- body token：每个身体关节一个 anchor。
- hand token：每只手一个 anchor，通常取 distal joints 的均值。
- full-body token：27 个身体表面 anchor。

然后在这些 anchor 附近采样 MapAnything 特征：

- body / hand anchor：采样 `3 x 3` 邻域。
- full-body anchor：采样 `1 x 1` 单 token。

这些局部视觉特征再通过 geometry-aware cross-attention 融入 HSI tokens。

---

## 10. 为什么不用 MoGe-2 一个模型

这里容易误解成“MapAnything 和 MoGe-2 都是环境重建模型，为什么不只用 MoGe-2”。

更准确的分工是：

```text
MapAnything:
  主场景几何来源
  输出 pointmap
  输出多层 visual features
  支持 Is + Ih 双图联合推理

MoGe-2:
  单目 metric geometry / depth estimator
  主要帮助恢复 metric scale
  不是 GRAFT 的主几何 backbone
```

为什么不用 MoGe-2 一个模型：

1. **GRAFT 不只需要 depth，还需要 MapAnything 的视觉特征。**

   GRAFT 的 geometry-aware cross-attention 会用 MapAnything 的多层 features。MoGe-2 不能直接替代这一套 visual token pipeline。

2. **GRAFT 需要两路一致的 pointmap。**

   MapAnything 联合处理 `Is` 和 `Ih`，得到共享 camera frame 下的：

   ```text
   Ps, Ph, Fs, Fh
   ```

   这对后续几何 probe 和人体初始对齐很重要。

3. **MoGe-2 在这里更像尺度辅助。**

   论文写的是用 monocular depth estimator 恢复 metric scale，引用的是 MoGe-2。它不是替代 MapAnything，而是帮助解决尺度问题。

---

## 11. 尺度优化 / metric alignment 是怎么实现的

论文里有两个尺度概念，容易混在一起。

### 11.1 初始化阶段的 metric alignment scale

这是粗对齐阶段用的尺度。

流程：

```text
NLF 估计人体 mesh
MapAnything 估计场景 pointmap
MoGe-2 辅助恢复 metric scale
然后用人体 head joint 做粗对齐
```

论文算法里的公式是：

```text
s0_i = p_scene_i,z / p_head_i,z
```

含义：

- 把第 `i` 个人的 head joint 投影到图像上。
- 在这个 2D 像素位置，从 `Ph` 里取对应的 3D scene point。
- 取 scene point 的深度 `p_scene_i,z`。
- 取当前人体 head joint 的深度 `p_head_i,z`。
- 用二者深度比值作为人体 mesh 的初始尺度修正。

直观理解：

> 如果人体头部当前深度是 2m，但场景 pointmap 认为这个位置应该在 3m，那就按比例把人体推到 / 缩放到更接近场景尺度的位置。

### 11.2 GRAFT refinement 阶段的 uniform scale update

GRAFT 每一步不仅预测 pose、translation、shape 的修正，还预测一个 uniform scale `s_t`。

这个 scale 会作用到 SMPL-X mesh 顶点上，用来修正初始化时残留的尺度误差。

论文为了让这个 scale 可训练、可反传，用了一个近似公式把 uniform scale 吸收到 SMPL-X 的 shape 参数里：

```text
β_s = s · (β + c) - c
```

其中：

- `β` 是 SMPL-X shape 参数。
- `s` 是预测的 uniform scale。
- `c` 是预先离线算好的常量。
- rotation 不变。
- translation 直接按尺度调整。

一句话总结：

```text
MapAnything: 主要负责点云和特征
MoGe-2: 主要补米制尺度
NLF: 负责人初始人体
GRAFT Transformer: 负责人-场景几何关系的迭代修正
```

---

## 12. geometry-only plug-in prior 如何接到 Human3R 后面

GRAFT 有两种用法：

| 模式 | 输入 | 是否使用图像特征 | 作用 |
|---|---|---:|---|
| full mode | `Ps, Θ, Fs, Fh` | 使用 | 作为完整的人体-场景重建系统 |
| geometry-only mode | `Ps, Θ` | 不使用 | 作为后处理 prior，接到其他方法后面 |

关键是：GRAFT 的核心输入不是某个特定模型的内部特征，而是：

```text
当前人体参数 Θ
场景几何 Ps
```

所以只要一个外部方法能输出：

- 初始人体 mesh 或 SMPL-X 参数 `Θ`
- 场景点云 / pointmap / scene geometry `Ps`

GRAFT 就可以把它当成初始状态继续修正。

以 Human3R 为例：

```text
输入图片
  -> Human3R
     得到:
       初始人体 SMPL-X / mesh
       初始场景几何 / pointmap / depth / scene mesh

  -> GRAFT geometry-only mode
     输入:
       scene geometry Ps
       current human parameters Θ
     不输入:
       MapAnything visual features Fs, Fh

  -> GRAFT 迭代 refinement
     Geometric Probes
     24 HSI tokens
     Transformer self-attention
     MLP heads 输出 ΔΘ 和 scale

  -> refined human mesh
```

geometry-only 模式下，GRAFT 主要依靠 Geometric Probes：

```text
人体关节 / 表面点 p
  -> 找最近的场景点 p*
  -> 计算相对位移 p* - p
  -> 读取场景法向量 n*
  -> 编码成 HSI token
  -> Transformer 预测人体参数修正量
```

这些几何信息已经能告诉模型：

- 脚是不是离地太远。
- 身体是不是穿进了物体。
- 手和桌面是不是接近。
- 人体整体位置是不是和支撑面不一致。

因此，GRAFT 可以关掉 visual cross-attention，或者不提供 `Fs, Fh`，只用几何 token 做 refinement。

伪代码：

```text
Θ0, Ps = Human3R(image)

for t in range(T):
    probes = GeometricProbe(Ps, Θt)
    tokens = HSI_Tokenize(probes, SMPLX_params=Θt)

    # geometry-only:
    # 不使用 VisualAnchors(Fs, Fh)
    # 不依赖 Human3R 的内部特征

    ΔΘt, st = GRAFT_Transformer(tokens)
    Θt+1 = Θt + ΔΘt
    mesh_t+1 = st * SMPLX(Θt+1)
```

为什么不需要重新训练 Human3R：

- Human3R 只负责提供初始人体和场景。
- GRAFT 在 Human3R 后面独立运行。
- Human3R 不参与反向传播。
- Human3R 的网络权重不需要修改。

所以 `Human3R + Ours` 的含义是：

```text
Human3R 输出初始重建
GRAFT geometry-only refinement 修正人体-场景交互
```

一句话理解：

> GRAFT 学到的是通用的人体-场景交互修正 prior。给它一个人体和一个场景，它就判断当前交互哪里不合理，并预测应该怎么改人体参数。

---

## 13. 训练方法

GRAFT 使用 InHabitants pseudo-labeled HSI dataset 训练：

- 75k images
- 97k human instances
- 单张 H100 训练
- Adam optimizer
- 150k iterations
- batch size：16
- peak learning rate：1e-4
- warmup：前 2k iterations
- cosine decay 到 2e-7

训练时不是只监督最后一步，而是监督每一次 refinement。

训练样本包括两种 query：

1. **NLF 初始化结果**：模拟真实测试时的错误初始化。
2. **GT 附近扰动结果**：让模型学会在接近正确答案时不要乱改。

扰动设置：

- clean GT 概率：0.2
- 加扰动概率：0.8
- translation noise：0.1 m
- shape noise：0.03
- pose rotation noise：7 度
- global orientation noise：3 度

rollout 设置：

- 默认 `T = 3`
- 训练和推理都用 3 次 refinement
- full rollout supervision 从 10k iterations 后开始
- global scale augmentation：`[0.85, 1.15]`
- visual-anchor dropout：每个 token 概率 0.35

---

## 14. Loss 怎么理解

论文没有用显式 contact loss，也没有用显式 interpenetration loss。

原因是单目场景 pointmap 是不完整的，遮挡区域会导致 contact / SDF 类 loss 不稳定。

GRAFT 使用的是每步监督：

- **rotation loss**：监督人体关节旋转。
- **camera-relative vertex loss**：监督人体整体位置、尺度和 mesh 顶点。
- **mean-normalized vertex loss**：去掉整体平移后监督体型和姿态。

直观理解：

> 模型不是被硬性告诉“这里必须接触、那里不能穿透”，而是通过大量正确的人-场景配置，学会什么样的人体和场景关系是合理的。

loss 权重补充材料给出：

- camera-relative vertex loss weight：`λv = 7.0`
- centered-vertex loss weight：`λn = 5.0`
- global orientation rotation weight：`5.0`
- body pose rotation weight：`2.0`
- left-hand pose rotation weight：`0.5`
- right-hand pose rotation weight：`0.5`

---

## 15. 结果和结论

GRAFT 的主要结果：

- 相比 feed-forward baseline，contact F1 最多提升 113%。
- 接近 PhySIC 这类优化方法的交互质量，但速度约快 50 倍。
- 推理时间约 0.38s，而 PhySIC 约 20s。
- geometry-only 模式可以作为 plug-in prior 接到 Human3R 后面，不需要重新训练 Human3R。
- 在 Human3R 后面接 GRAFT，contact F1 最多提升约 44%。

论文中 `Human3R + Ours` 的实验说明：

- Human3R 先输出初始重建。
- GRAFT geometry-only mode 只用几何信息做 refinement。
- Human3R 本身不需要重新训练。

---

## 16. 当前代码仓库状态

当前目录下 `repo/README.md` 显示：

- project page 已发布
- arXiv paper 已发布
- demo release 未发布
- code release 未发布

所以目前本地 GRAFT 仓库里还没有可核对的模型源码、训练脚本或 loss 实现。

本文件中的架构、backbone、训练细节主要来自：

- GRAFT 论文正文
- GRAFT 补充材料
- 当前对话中对 MapAnything / VGGT / MoGe-2 / Human3R plug-in 模式的整理

后续如果源码发布，最值得核对的是：

- GRAFT Transformer 的实际模块实现。
- MapAnything feature 的具体抽取层名。
- HSI token tokenizer 的 MLP 结构。
- visual-anchor dropout 的实现。
- loss weights 和 rollout schedule。
- 数据预处理和 pseudo-label 生成流程。

---

## 17. OmniEraser 论文和仓库

本轮已经把 OmniEraser 的论文和仓库拉到当前目录：

```text
OmniEraser_paper.pdf
Omnieraser/
```

论文：

```text
OmniEraser: Remove objects and their effects in images with paired video-frame data
arXiv: 2501.07397
```

仓库：

```text
https://github.com/PRIS-CV/Omnieraser
```

本地仓库顶层包含：

- `README.md`
- `requirements.txt`
- `train_control_lora_flux.py`
- `test_control_lora_flux.py`
- `pipeline_flux_control_removal.py`
- `ControlNet_version/`
- `example/`
- `static/`

注意：检查 `git status` 时 Git 报过 `dubious ownership`，这是当前沙箱用户和文件所有者不一致导致的安全检查，不影响文件已下载。如果后续需要在 `Omnieraser/` 里执行 git 命令，可以把该目录加入 Git 的 `safe.directory`。

---

## 18. 架构图

已经生成完整架构图：

```text
graft_architecture_diagram.svg
```

图中包含：

- `Ih` 原始人-场景图
- `OmniEraser` 去人，生成 `Is`
- `MapAnything(Is, Ih)` 联合推理，输出 `Ps / Ph / Fs / Fh`
- `NLF` 初始化 SMPL-X 人体
- `MoGe-2` 辅助 metric scale
- `Metric Alignment`，用 `s0 = z_scene / z_head` 做初始尺度对齐
- `Geometric Probes`
- `24 个 HSI Tokens`
- `Fourier + MLP` 编码
- `Visual Anchors`
- `5-layer Transformer`
- `MLP Heads`
- `SMPL-X Update`
- `T = 3` 迭代 refinement loop
- 每步训练监督 loss

SVG 可以直接用浏览器打开，也可以导入 PPT / draw.io / Figma 继续编辑。
