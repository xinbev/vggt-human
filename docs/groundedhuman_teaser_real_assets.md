# 从 woman_sofa.jpg 导出 GroundedHuman teaser 的真实素材

## 目标和实现边界

输入为服务器上的 `/home/zhw/lab_users/xyb/home/projects/vggt-human/assets/image/teaser/woman_sofa.jpg`。
导出人体、场景、anchor、局部 scene tokens、几何搜索候选点、区域四尺度采样和真实平移建议，供后续 teaser 排版使用。没有调用图像生成服务，没有修改 `.paper/`、模型源码、训练配置或 baseline 路径。

参考的是当前 GroundedHuman 的 Anchor-Scene Cross Attention 和 Regional Translation Prediction；具体复用本仓库 `HSIRefinementHead`、`RegionalSceneProbe`、`HSIRegionalTranslationRefiner`。属于可视化适配，不是移植第三方模型。沿用 `infer_smpl_hsi_v3_trstr_spatial.yaml` 的结构，不新增模型分支。

仓库指令提到的 `.reference/stage2_body_j2d1e-4.sh` 在当前 checkout 不存在，脚本风格改为参考现有 `scripts/vis/export_paper_architecture_real_assets.sh` 和 `serve_walking_hsi_v3_trstr_spatial.sh`。

## 运行

本地入口：`C:\Users\ROG\PycharmProjects\vggt-omega\scripts\vis\export_groundedhuman_teaser_assets.sh`。

同步后入口：`/home/zhw/lab_users/xyb/home/projects/vggt-human/scripts/vis/export_groundedhuman_teaser_assets.sh`。

进入服务器项目并激活已有项目环境，然后执行：

```bash
cd /home/zhw/lab_users/xyb/home/projects/vggt-human
bash scripts/smoke/check_groundedhuman_teaser_assets.sh
PREFLIGHT_ONLY=true bash scripts/vis/export_groundedhuman_teaser_assets.sh
bash scripts/vis/export_groundedhuman_teaser_assets.sh
```

smoke 不需要 ckpt 或 SMPL 文件；默认使用 torch 在 CPU 上把独立导出掩码与真正的 `RegionalSceneProbe` 对照。输出在 `outputs/debug/groundedhuman_teaser_smoke/`，其中图片明确标记为合成测试，不是 sofa 推理结果。

资源预检查不构建模型，读取配置并检查输入、VGGT、NLF、SMPL 和两个 head ckpt。正式推理和绘图输出在 `outputs/vis/groundedhuman_teaser/woman_sofa/`，与之前 sofa.png 的输出目录分开。

多 GPU 服务器可指定可见卡，进程内部仍使用 `cuda:0`：

```bash
CUDA_VISIBLE_DEVICES_VALUE=7 bash scripts/vis/export_groundedhuman_teaser_assets.sh
```

可覆盖 ckpt、输入、输出和人员选择，不用改源码：

```bash
CHECKPOINT=/absolute/path/to/accepted_trstr.pt \
SCALE_CHECKPOINT=/absolute/path/to/accepted_scale.pt \
OUTPUT_DIR=outputs/vis/groundedhuman_teaser/woman_sofa_run02 \
bash scripts/vis/export_groundedhuman_teaser_assets.sh
```

默认 ckpt 来自已有 walking 推理脚本：

- TRSTR：`outputs/train/smpl_hsi_stage2_trstr_v3_scale_spatial/checkpoint_latest.pt`
- HSI scale：`outputs/train/smpl_hsi_coarse_residual_stratified_v3/checkpoint_top_train_epoch_0005_loss_total_0.009242.pt`

这两个训练结果路径没有在 `configs/path.yaml` 中声明，而是沿用已存在的推理入口；必须在服务器确认文件仍存在，缺失时通过环境变量指定。VGGT、NLF 和 SMPL 优先从 `configs/path.yaml` 读取。head 加载会严格检查所需 tensor 与 shape；不接受随机初始化的缺失权重。scale ckpt 只覆盖 HSI head，TRSTR ckpt 只覆盖 TRSTR head，避免两个完整训练 checkpoint 互相覆盖。

`PERSON_RANK=0` 默认选置信度最高的人，其他人体仍参与遮挡／实例归属判断。`PERSON_RANK=1` 可选第二人。图片必须包含可识别的人；纯沙发或检测失败时明确报错，不生成虚构人体。

已完成目录和包含残留文件的失败目录均不覆盖；重新运行时选择新的 `OUTPUT_DIR`。默认不会安装任何依赖。

## 素材索引

| 文件／目录 | 用途 |
| --- | --- |
| `asset_contact_sheet.png` | 首先打开它，检查人体、采样部位是否适合选图 |
| `input_original.png` / `input_processed.png` | 原图与模型预处理后的图像 |
| `human_scene_overlay.png` | 场景深度遮挡处理后的人体与输入 RGB 合成 |
| `human_scene_white.png` | 白底场景／人体素材；不是背景修复结果 |
| `layers/human_base_rgba.png` | NLF 原始人体，透明底、珊瑚色 |
| `layers/human_refined_rgba.png` | TRSTR 后完整人体，透明底、青色 |
| `layers/human_scene_occluded_rgba.png` | 根据场景深度做了遮挡的青色人体 |
| `layers/environment_rgba.png` | 几何人体过滤后的可见环境；遮挡后方不补画 |
| `layers/anchors_rgba.png` | 与完整画布对齐的 24 body anchors |
| `layers/translation_arrows_1x_rgba.png` | 原始位移幅度，区域建议和整体平移箭头 |
| `layers/translation_arrows_5x_rgba.png` | 默认 5 倍显示的替代箭头图层；仅用于看清小位移 |
| `anchors/anchor_XX.png` | 一个 body anchor 查询其实际 token 邻域；连线强度取真实 attention |
| `anchors/anchor_XX_background.png` / `_overlay.png` | 分离的局部 RGB 和 token／attention 图层 |
| `anchors/anchor_XX_geometry_search.png` | 实际深度像素候选窗口，橙点是最近的有效 3D scene point |
| `anchors/all_anchors.npz` | 24 anchors、投影、nearest scene、offset、token 索引和逐 head attention |
| `regions/{group}_rXX.png` | 同一区域的四种真实采样支持；包含 self/environment/other/invalid 标记 |
| `regions/{group}_rXX_fixed_3.png` 等 | 每一种支持域独立导出，含 background／overlay 分层 |
| `regions/{group}_rXX.npz` | sample 像素、3D 点、有效性、四尺度掩码、通道、attention pooling 权重、实际 vote |
| `regions/probe_state_XX.npz` | 每次 probe 的输入人体、centers/reps、depth/owner、logits、tokens 等 |
| `regions/translation_iterations.npz` | 全 96 区域、每轮 vote/gate/logvar/valid/weight/person_gate 和 root translations |
| `regions/selection.json` | 选中的手／骨盆／脚区域及其真实可靠性、位移 |
| `geometry/human_base.ply` / `human_refined.ply` | 可在 Blender、MeshLab、Viser 重新取景的人体网格 |
| `geometry/scene_metric_environment.ply` | 校准后且按几何归属过滤人体的可见环境点云 |
| `geometry/scene_metric_all.ply` | 校准后的完整可见点云 |
| `geometry/scene_raw_scale_ambiguous.ply` | 原始非度量点云，不能直接当成米制坐标 |
| `geometry/body_anchors.ply` / `anchor_nearest_scene_links.ply` | 实际 3D anchor、nearest scene 及连接线 |
| `geometry/{group}_rXX_{scale}_{channel}.ply` | 区域不同支持域、不同通道的真实 3D 采样点 |
| `camera_geometry.npz` | K、原始外参、三阶段深度、人体网格和图像尺寸 |
| `scale_correspondences.npz` | 解析尺度初始化的表面像素对应及比值 |
| `manifest.json` | 输入 hash、模型路径、坐标说明、检查结果、全部文件索引 |

所有全画布 RGBA 图层使用同样的 `render_scale`（默认 3）和 processed RGB 的相机投影，能够直接对齐叠加。局部 crop 的坐标保存在相邻 JSON。人体网格渲染使用 CPU z-buffer，不需要额外安装 OpenGL/pyrender。

## 方法和坐标一致性

1. 输入 `images` 为 `[1,1,3,H,W]`，RGB float32、0–1，在配置 device 上执行；分辨率沿用已有 balanced resize。
2. NLF 产生 pose6d `[1,1,Q,144]`、betas `[1,1,Q,10]`、translation `[1,1,Q,3]`。人体坐标单位为米。
3. 基于置信度合格的人体表面作解析尺度估计，然后 HSI 预测残差尺度和深度偏置。明确组合 `D_metric = D_raw * coarse_scale * residual_scale + bias`，再单独调用 TRSTR。
4. 该顺序与现有序列 viewer 的 clip-depth cascade 对齐；单帧的 median 等于本帧值，但不能把它说成具有多帧共识。这里不直接把 model 默认返回的 `hsi_translation_depth` 当成已经组合好的最终深度。
5. anchor 是 `[24,3]`；3×3 token 邻域来自 HSI 实际特征网格，当前 gather 按 `H_depth//16,W_depth//16` 映射。它不是 3×3 深度像素。图像边缘 token 重复索引按实际实现保留。
6. HSI 几何证据另用当前 `probe_window=9` 的深度像素邻域选最近有效 3D scene point；有单独的数值和 PNG 素材，避免和 token patch 混淆。
7. TRSTR centers `[Q,96,3]`、representatives `[Q,96,8,3]`。固定支持域为 3×3、7×7；adaptive radius 取代表顶点投影范围加 1，再向上取整并限幅；外环用 Chebyshev 距离，所以是方形环带。
8. self/environment/other 标签复用真正的模型人体 depth/owner rasterizer，并使用模型的 depth tolerance。other-human 不参加 self/environment pooling；越界、无效深度不会通过坐标 clamp 被误画成有效观测。
9. 区域建议通过 `valid * sigmoid(gate) * exp(-logvar)` 汇总，随后 person gate 和截断给出每轮全身平移。运行时验证导出的 votes/weights 能还原实际 translations；pose 和 shape 始终沿用 base。
10. `probe_state_00..num_iters-1` 是各轮更新之前的输入；最后一个 snapshot 是更新完后的 audit。用于解释 vote 的采样图固定取最后一轮更新前的状态，避免混用最终位置与前一轮 vote。
11. PLY 使用 camera-space：x 向右、y 向下、z 向前。除明确命名为 raw 的点云外单位为米。这里不导出未经校准的 world transform；存储的原始外参平移仍是原始尺度。
12. attention 通过观察原 forward 的 Q/K/V，在最后一层使用同一 MHA 权重重算逐 head 权重，检查 replay output 与原 forward 一致；不替换原 forward 输出。局部区域 pooling 权重由实际 logits 和掩码重现。所有 hook 在异常时也移除。

## 已完成和待服务器验证

本地已做 Python AST 语法检查、CLI help、YAML 解析、两个 `.sh` 的 `bash -n` 检查；已执行 NumPy/Pillow smoke，覆盖窗口点数、方环互斥、其他人体排除、无效／越界深度、空 attention、非方形坐标、token 边缘重复、局部最近点和 z-buffer/RGBA 输出。已目视检查合成采样图。

本机没有 torch、模型环境和服务器图片／ckpt，未执行实际 sofa forward、GPU/device 检查、live sampler 对照或最终素材视觉验证。服务器先跑上述 smoke，再预检查资源和正式导出。正式导出内置 live valid-ratio 对照、attention replay 对照、全身 translation aggregation 对照、pose/shape 不变检查和位移后网格一致性检查。

科研风险：单图场景仅包含可见几何；人体预测错误、遮挡和深度误差可能造成采样不可靠、接触未改善或轻微位移。无有效证据的区域标为 `UNSUPPORTED`，零更新不画成虚构的非零改善。四个窗口有时大小重合，这是实际采样结果。环境去人依赖几何，可能留下残影／孔洞；当前不新增 SAM 或修复网络。正式投稿前应看 `manifest.json`、数值与原图，不能只挑好看的箭头。
