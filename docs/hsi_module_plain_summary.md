# HSI 模块白话版

这份是给自己快速理解用的，不是正式论文文字。

## 1. HSI 到底在干什么

可以把 HSI 想成一个“人体-场景对齐检查员”。

前面的 SMPLHead 已经预测了一版人体：这个人在哪里、什么姿态、什么体型。但是这版人体可能和 VGGT-Omega 的场景深度不完全对齐，比如：

```text
脚看起来悬空
身体穿进场景
人体 translation 偏了一点
场景 depth 整体尺度和人体尺度不一致
```

HSI 的作用就是拿这版人体去问场景：

```text
我的脚、膝盖、手、身体中心这些位置，附近的场景深度在哪里？
我离场景表面是太远了，还是穿进去了？
如果不对，应该怎么小幅修正人体？
同时，场景 depth 的整体尺度是不是也应该用人体尺度校正一下？
```

## 2. 它不是从零重建人体

HSI 不是重新预测一个人。它是 refinement。

流程是：

```text
先有 base SMPL
再根据 base SMPL 找身体锚点
再看这些锚点附近的场景
最后预测残差修正
```

所以 HSI 更像：

```text
base prediction + local scene evidence -> correction
```

而不是：

```text
image -> new SMPL from scratch
```

## 3. 为什么是 24 个 token

HSI 在每个人身上放了 24 个“探针”。

这些探针来自：

```text
21 个非 root 身体关节
2 个手部中心点
1 个全身中心点
```

你可以理解成：在人体骨架和身体表面放一些传感器。每个传感器都去看自己附近的场景，判断人体和场景是不是对得上。

## 4. 每个探针看什么

每个 HSI token 会看两类信息。

第一类是人体自己的信息：

```text
当前姿态
当前体型
当前 3D 平移
这个身体锚点的 3D 位置
```

第二类是场景的信息：

```text
这个锚点投影到 depth map 的哪个位置
该位置的场景深度
反投影出来的场景 3D 点
局部场景法线
人体点到场景点的距离
人体点和场景点的深度差
附近 VGGT patch token 的特征
```

白话说：每个身体点都拿着尺子去量自己和附近场景的关系。

## 5. HSI Transformer 做什么

HSI Transformer 让这些身体探针互相交流，也让它们看附近的 scene tokens。

它有两件事：

```text
身体内部协调：
  脚要往下修，膝盖和髋部也要合理，不能只动一个点。

身体-场景交互：
  每个身体点看自己附近的深度和 scene tokens，决定要不要修正。
```

所以它不是只看单个脚点，而是看整个人和周围场景的局部关系。

## 6. HSI 最后输出什么

HSI 输出两类结果。

第一类是修人体：

```text
pose residual
shape residual
translation residual
```

也就是把 base SMPL 变成 HSI-refined SMPL。

第二类是修场景 depth：

```text
scene scale
scene depth bias
```

也就是：

```text
HSI depth = scale * VGGT depth + bias
```

这点很重要：HSI 不只是“场景帮助人体”，也是“人体尺度反过来帮助场景”。

## 7. contact logits 是什么

HSI 还会对每个身体锚点输出一个 contact logit。

它可以理解成：

```text
这个身体点是不是应该和场景接触？
```

但论文里不要把它吹成完整 contact reasoning。当前更稳妥的说法是：contact logits 是 HSI 的辅助几何信号，主贡献是 geometry-grounded refinement 和 human-scale calibration。

## 8. temporal memory 是什么

代码里有一个可选的 temporal memory 分支。它可以把上一帧的 HSI token、contact、translation、scale/bias 存起来，下一帧再用。

但是这个不应该画成当前论文主贡献。现在最稳的主线还是单帧/短序列的 HSI geometry refinement。

如果要画 temporal，画成灰色虚线：

```text
optional temporal memory
```

## 9. 一句话给你讲明白

HSI 就是：

```text
先在人体上放 24 个几何探针，
把这些探针投到 VGGT 的深度图里，
看看人体点和附近场景点差多少，
再用 Transformer 综合全身和局部场景信息，
输出人体的小修正和场景 depth 的尺度/偏移校正。
```

## 10. 画图时最关键的表达

图上一定要画出这四件事：

```text
1. Base SMPL 不是最终结果，HSI 在它后面做 refinement。
2. 24 body anchors 是 HSI token 的来源。
3. 每个 anchor 都投影到 depth map / local scene patch 中取信息。
4. HSI 同时输出 refined human 和 calibrated scene depth。
```

不要画成：

```text
HSI = tracker
HSI = contact solver
HSI = 普通后处理
HSI = 只修人体
HSI = 只修场景
```

