# 参考项目映射表

本文件用于说明 `.reference/` 下每个外部论文项目或参考实现的用途边界。

`.reference/` 中的代码不是主项目运行代码，默认只作为阅读、分析和适配参考。

主项目代码位于 `vggt_human/`。主项目不得直接 import `.reference/` 中的代码。

## 总体规则

参考项目只用于理解论文思想、张量流、模型结构、损失函数、训练流程、数据处理或评估方法。

不要直接复制整个参考项目的工程结构。

不要引入参考项目的完整训练框架、数据集加载器、日志系统或命令行工具，除非任务明确要求。

如果需要使用参考项目中的实现，应当把核心思想适配到 `vggt_human/` 中，并符合当前项目代码结构、命名规范、配置方式和验证方式。

使用任何参考项目前，必须先明确：

1. 当前参考的是哪个项目。
2. 当前参考的是哪个具体模块、函数、类或论文思想。
3. 当前实现属于直接移植、适配改写，还是概念重写。
4. 是否保留 VGGT 原始 baseline。
5. 是否需要配置开关控制实验模块。
6. 是否会改变输入输出 shape、坐标约定、dtype 或 device 行为。

## 当前参考项目

| 参考项目 | 路径 | 主要用途 | 不应使用的部分 |
|---|---|---|---|
| vggt_omega | `.reference/vggt_omega/` | VGGT 相关参考实现，用于理解 baseline、模型结构、张量流、训练或推理逻辑 | 不直接作为主项目运行路径，不直接 import，不整体复制工程结构 |
| SAT-HMR-smpl | `.reference/SAT-HMR-smpl/` | SMPL regression 参考实现，重点参考从 decoder hidden states 回归 SMPL pose、betas、camera、confidence 的 head 设计，以及 SMPL layer、旋转转换、相机投影和可选训练损失 | 不直接引入完整 SAT-HMR 工程，不直接 import，不整体复制训练框架、数据加载器、日志系统或 CLI |
| SAT-HMR-query | `.reference/SAT-HMR-query/` | query-based decoding 参考实现，重点参考 SAT-HMR/DAB-DETR 风格的 learnable content query、reference box query、reference-position encoding、query decoder 和逐层 bbox/reference refinement | 不直接替换 VGGT encoder/backbone，不直接引入完整 SAT-HMR decoder 工程，不直接 import，不迁移 denoising training 逻辑，除非任务明确要求 |
| data_loading | `.reference/data_loading/` | portable data-loading 参考实现，重点参考 BEDLAM 数据索引、图像预处理、相机内参变换、SMPL/SMPL-X 标注转换、batch collate 和 slot supervision 构建 | 不直接作为主项目 dataloader 包 import，不引入 Scal3R 工程路径，不提交数据集、SMPL 模型资产或本机绝对路径 |

## 使用 `.reference/vggt_omega/` 的规则

使用 `.reference/vggt_omega/` 时，需要先明确：

1. 当前参考的是 baseline 行为、模型结构、训练逻辑、推理逻辑，还是工具函数。
2. 当前要适配到 `vggt_human/` 的具体模块。
3. 原始 VGGT 路径是否需要保留。
4. 新实现是否需要配置开关。
5. 是否会改变输入输出 shape、坐标约定、dtype 或 device 行为。

默认只做最小适配，不进行大范围重写。

## 使用 `.reference/SAT-HMR-smpl/` 的规则

`.reference/SAT-HMR-smpl/` 是从 SAT-HMR 中抽取出的 SMPL regression 参考布局，用于把人体参数回归能力适配到当前 VGGT 改造项目中。

SAT-HMR-smpl 的核心参考思想包括：

1. decoder 输出 hidden states，形状为 `(num_decoder_layers, batch_size, num_queries, hidden_dim)`。
2. 每个 decoder layer 使用独立的 `pose_head` 和 `shape_head`，这些 head 来自同一种 MLP 结构的 clone。
3. pose 和 shape 从 `mean_pose`、`mean_shape` 出发做 residual refinement。
4. pose head 预测 6D rotation，再转换为 SMPL axis-angle 后调用 SMPL layer。
5. head 输出可包含 `pred_poses`、`pred_betas`、`pred_confs`、`pred_cam`，以及可选的中间 decoder layer 辅助输出。

默认用途是参考 SMPL regression head 和几何后处理设计，而不是替换 VGGT backbone、encoder、decoder 或完整训练系统。

### SAT-HMR-smpl 模块边界

| 模块 | 路径 | 可参考内容 | 迁移注意事项 |
|---|---|---|---|
| 配置路径 | `.reference/SAT-HMR-smpl/config/` | `SMPL_MODEL_PATH`、`SMPL_MEAN_PARAMS_PATH` 的组织方式 | 不要硬编码本机绝对路径；应接入本项目配置方式；SMPL checkpoint 或资产由用户准备 |
| 回归头 | `.reference/SAT-HMR-smpl/heads/` | `SMPLRegressionHead`，从 decoder hidden states 回归 pose、betas、confidence、camera | 需要匹配 `hidden_dim`、`num_decoder_layers`、`num_queries`；优先作为实验 head 接入，保留原 VGGT 输出 head |
| SMPL layer | `.reference/SAT-HMR-smpl/smpl_layer/` | `smplx` wrapper，输入 axis-angle pose 和 betas，输出 vertices 和 joints | 引入 `smplx` 前必须确认依赖需求；SMPL 模型文件由用户提供；不要把模型资产提交进仓库 |
| 几何工具 | `.reference/SAT-HMR-smpl/geometry/` | 6D rotation 到 axis-angle 的转换，相机参数到 3D/2D 投影的后处理 | 必须检查 VGGT 当前相机约定、坐标系、尺度、归一化方式和 image size/focal 定义 |
| 集成示例 | `.reference/SAT-HMR-smpl/integration_example/` | 如何把 decoder hidden states 和 camera intrinsics 接到 SMPL head 与 SMPL layer | 只能作为连接方式参考，不能直接作为主项目入口；需要改成 `vggt_human/` 的接口和配置风格 |
| 训练可选项 | `.reference/SAT-HMR-smpl/training_optional/` | Hungarian matcher、confidence focal loss、pose/betas/2D joints/3D joints/depth L1 loss | 只有在训练多人体 query 且任务明确需要时才使用；必须适配本项目 target keys、box 格式、joint 定义和 mask 约定 |

### SAT-HMR-smpl 推理迁移优先级

如果任务目标是 inference-only 或先做最小 forward smoke test，优先参考：

1. `.reference/SAT-HMR-smpl/heads/smpl_regression_head.py`
2. `.reference/SAT-HMR-smpl/geometry/rotation_conversions.py`
3. `.reference/SAT-HMR-smpl/geometry/camera_projection.py`
4. `.reference/SAT-HMR-smpl/smpl_layer/smpl_layer.py`
5. `.reference/SAT-HMR-smpl/config/smpl_paths.py`
6. `.reference/SAT-HMR-smpl/integration_example/decoder_to_smpl_example.py`

迁移时应先打通 decoder hidden states 到 SMPL 参数输出的最小路径，再考虑 SMPL vertices、joints、camera projection 和可视化。

### SAT-HMR-smpl 训练参考使用条件

`.reference/SAT-HMR-smpl/training_optional/` 只在以下条件同时满足时使用：

1. 当前任务明确要求训练 SMPL regression 或 multi-query human prediction。
2. 本项目已有或将新增对应 target 字段，例如 `boxes`、`poses`、`betas`、`j3ds`、`j2ds`、`j2ds_mask`、`depths`。
3. 已确认 bbox 格式是 `cxcywh` 还是 `xyxy`。
4. 已确认 2D joints 是否归一化、3D joints 是否 root-aligned、SMPL joint 数量和 root index。
5. 已确认 unmatched queries 是否作为有效 negative 参与 confidence loss。

如果这些条件不明确，不要直接迁移 matcher 或 losses。

### SAT-HMR-smpl 与 VGGT 集成要求

将 SAT-HMR-smpl 思路适配到 `vggt_human/` 时，需要遵守：

1. 先检查当前 VGGT decoder 或 head 的输出 tensor shape。
2. 如果当前 VGGT decoder 输出不是 `(num_decoder_layers, batch_size, num_queries, hidden_dim)`，需要设计明确的 adapter，而不是强行改动整个 decoder。
3. 新 SMPL head 应优先作为实验模块新增，并通过配置开关启用。
4. 原 VGGT baseline head 和推理路径必须保留，除非用户明确要求替换。
5. 不允许主项目直接从 `.reference/SAT-HMR-smpl/` import 代码。
6. 若需要使用 rotation conversion、camera projection 或 SMPL wrapper，应复制并适配到 `vggt_human/` 的合适模块位置，并注明来源思想。
7. 所有 SMPL 模型文件、mean params、checkpoint 和人体模型资产由用户提供，不要自动下载或提交到仓库。
8. 如果新增依赖 `smplx`，必须先说明原因，并确认不会破坏 baseline 的 import 和运行。

### SAT-HMR-smpl 验证要求

涉及 SAT-HMR-smpl 适配的代码改动完成后，优先做以下验证：

1. import 检查。
2. 构造最小 dummy decoder hidden states，检查 SMPL head forward 输出 shape。
3. 检查 `pred_poses` 是否为 `(B, Q, 72)`，`pred_betas` 是否为 `(B, Q, 10)`，`pred_confs` 是否为 `(B, Q, 1)`，`pred_cam` 是否为 `(B, Q, 3)`。
4. 如果接入 SMPL layer，检查 vertices 和 joints 的 batch 展平与还原逻辑。
5. 如果接入 camera projection，检查坐标系、depth、focal、image size 和归一化约定。
6. 如果 full SMPL asset 或 checkpoint 缺失，需要明确说明哪些验证无法完成。

## 使用 `.reference/SAT-HMR-query/` 的规则

`.reference/SAT-HMR-query/` 是从 SAT-HMR 中抽取出的 query-based decoding 参考布局，用于理解和迁移 SAT-HMR/DAB-DETR 风格的 query 机制。

SAT-HMR-query 的核心参考思想包括：

1. 每个 object/person slot 同时具有 learnable content embedding `tgt_embed` 和 learnable reference box embedding `refpoint_embed`。
2. `refpoint_embed` 的维度为 4，表示 inverse-sigmoid 空间中的 normalized box `(cx, cy, w, h)`。
3. decoder 将 reference boxes 转成 sine positional embeddings，并作为 query position 使用。
4. 每层 decoder 通过 self-attention、cross-attention 和 FFN 更新 query hidden states。
5. bbox heads 在 inverse-sigmoid 空间逐层 residual refinement reference boxes，最终 boxes 约束在 `[0, 1]`。
6. 最终 decoder hidden states 可以接入 SMPL head 或其他 task heads。

默认用途是参考 query 初始化、reference-position encoding、query decoder 连接方式和 iterative reference refinement，而不是直接替换 VGGT 的 backbone、encoder 或完整 decoder 堆栈。

### SAT-HMR-query 模块边界

| 模块 | 路径 | 可参考内容 | 迁移注意事项 |
|---|---|---|---|
| Query 初始化 | `.reference/SAT-HMR-query/query_init/` | `tgt_embed` content query 和 `refpoint_embed` reference box query 的初始化方式 | 需要确认 `num_queries`、`hidden_dim`、`query_dim=4` 是否适合当前 VGGT 任务；不要无条件改变 VGGT 原始 token/query 组织 |
| Reference position | `.reference/SAT-HMR-query/position/` | 将 `(cx, cy, w, h)` reference boxes 转成 sine/cosine embedding，再投影为 `query_pos` | 如果当前项目只需要 point query `(cx, cy)`，必须修改 `query_dim` 和位置编码；必须确认坐标归一化约定 |
| Query decoder | `.reference/SAT-HMR-query/decoder/` | 标准 PyTorch `nn.MultiheadAttention` 版本的 query decoder，输出 hidden states 和逐层 references | 输入约定是 flattened `memory: (sum_tokens, hidden_dim)`、`memory_lens`、`memory_pos`；需要适配 VGGT 当前 encoder/aggregator 输出 |
| BBox refinement | `.reference/SAT-HMR-query/bbox_update/` | 在 inverse-sigmoid 空间对 reference boxes 做 residual update | 如果 VGGT 任务不需要显式 bbox，可以只保留 reference update 作为 attention prior，或在明确分析后移除 |
| Denoising queries | `.reference/SAT-HMR-query/denoising_optional/` | DINO 风格训练 denoising queries，训练时把 noisy ground-truth queries 拼到 regular queries 前面 | 仅训练时使用；需要 target 中有 `labels`、`boxes`，如使用参数 query 还需要 `poses`、`betas`；不要用于 inference-only 迁移 |
| 集成示例 | `.reference/SAT-HMR-query/integration_example/` | 从 encoder tokens 和 pos embeds 到 query decoder，再到 bbox/head 输出的最小连接流程 | 只能作为连接方式参考；需要改成 `vggt_human/` 的接口、tensor layout 和配置风格 |

### SAT-HMR-query 推理迁移优先级

如果任务目标是先接入 query 机制或做最小 forward smoke test，优先参考：

1. `.reference/SAT-HMR-query/query_init/query_initializer.py`
2. `.reference/SAT-HMR-query/position/reference_position_encoding.py`
3. `.reference/SAT-HMR-query/decoder/query_decoder.py`
4. `.reference/SAT-HMR-query/bbox_update/bbox_refinement.py`
5. `.reference/SAT-HMR-query/integration_example/query_pipeline_example.py`

迁移时应先打通 encoder features 到 query decoder hidden states 的最小路径，再考虑 bbox refinement、SMPL head、loss 或 denoising training。

### SAT-HMR-query 训练参考使用条件

`.reference/SAT-HMR-query/denoising_optional/` 只在以下条件同时满足时使用：

1. 当前任务明确要求训练 query-based detector、multi-person query 或 DINO-style denoising。
2. 本项目已有或将新增 target 字段 `labels` 和 normalized `boxes`。
3. 如果使用 `tgt_embed_type='params'`，本项目 target 还需要 `poses` 和 `betas`。
4. 已确认 bbox 格式为 normalized `cxcywh`，或已经设计好格式转换。
5. 已确认 attention mask、denoising groups 和 regular queries 的隔离逻辑与训练目标一致。

如果这些条件不明确，不要直接迁移 denoising queries。

### SAT-HMR-query 与 VGGT 集成要求

将 SAT-HMR-query 思路适配到 `vggt_human/` 时，需要遵守：

1. 先检查当前 VGGT encoder、aggregator、decoder 或 head 的 token layout 和 tensor shape。
2. 明确当前 VGGT 是否已有 query/token 机制，避免重复引入一套互相冲突的 query 表示。
3. 如果当前 VGGT 输出是 `(B, N, C)`，而参考 decoder 期望 `memory: (sum_tokens, hidden_dim)` 和 `memory_lens`，需要设计明确 adapter。
4. 新 query 机制应优先作为实验模块新增，并通过配置开关启用。
5. 原 VGGT baseline decoder/head 路径必须保留，除非用户明确要求替换。
6. 不允许主项目直接从 `.reference/SAT-HMR-query/` import 代码。
7. 如果将 SAT-HMR-query 与 SAT-HMR-smpl 联合使用，需要先保证 query decoder 输出 `hidden_states` 与 SMPL head 期望的 `(num_decoder_layers, batch_size, num_queries, hidden_dim)` 一致。
8. 如果移植 bbox refinement，必须明确 bbox 是否是最终监督目标，还是只作为 query reference prior。

### SAT-HMR-query 验证要求

涉及 SAT-HMR-query 适配的代码改动完成后，优先做以下验证：

1. import 检查。
2. 构造最小 dummy encoder memory 和 position embedding，检查 query decoder forward。
3. 检查 `hidden_states` 是否为 `(num_layers, B, Q, C)`。
4. 检查 `references` 是否为 `(num_layers, B, Q, 4)`，且经过 sigmoid 后位于 `[0, 1]`。
5. 如果接入 bbox refinement，检查 `pred_boxes` 的 shape、格式和归一化范围。
6. 如果接入 SAT-HMR-smpl head，检查 query hidden states 与 SMPL head 输入维度一致。

## 使用 `.reference/data_loading/` 的规则

`.reference/data_loading/` 是从 Scal3R 工作区抽取出的 portable data-loading 参考布局，用于理解和适配 BEDLAM 数据加载、图像预处理、SMPL 标注转换、batch schema 和 slot supervision 构建。

data_loading 的核心参考思想包括：

1. `BedlamDataset` 读取预处理后的 BEDLAM sequence，单个样本带有序列维度 `[S, ...]`。
2. `bedlam_collate_fn` 将 batch 组织为 `[B, S, ...]`，并将 Multi-HMR 分支输入展平为 `[B*S, ...]`。
3. 预处理同时支持 direct square resize 和 Multi-HMR contain+pad letterbox，并分别维护变换后的相机内参。
4. SMPL-X person dict 被转换为 padded SMPL tensors，包括 6D pose、betas、camera-space translation、joint slots 和 valid-person mask。
5. `SMPLGTBuilder` 将 3D joints 和 intrinsics 转换为 slot UV、patch score maps、center 3D 和 SMPL joint supervision。
6. `ImageFolderDataset` 提供轻量级图片目录推理输入列表。

默认用途是参考数据接口、预处理和监督构建方式，而不是直接把参考目录作为主项目运行包使用。

### data_loading 模块边界

| 模块 | 路径 | 可参考内容 | 迁移注意事项 |
|---|---|---|---|
| BEDLAM dataset | `.reference/data_loading/bedlam/` | 预处理 BEDLAM sequence indexing、低层文件 IO、`BedlamDataset` 样本组织 | 必须适配本项目数据目录配置；不要硬编码本机数据路径；不要假设原 Scal3R import 路径存在 |
| 图像预处理 | `.reference/data_loading/preprocessing/` | ImageNet normalization、direct square resize、Multi-HMR letterbox、camera intrinsics transform | 必须确认 VGGT 当前输入分辨率、归一化均值方差、resize/letterbox 策略和相机内参约定 |
| SMPL 转换 | `.reference/data_loading/smpl/` | BEDLAM SMPL-X person dict 到 SMPL pose/betas/transl/joints 的转换，axis-angle/6D rotation conversion，optional `smplx.SMPL` helper | `smplx` 和 SMPL 模型文件为可选本地依赖；缺失时只能做 smoke test 级 fallback，不代表最终训练质量 |
| Batch collate | `.reference/data_loading/batching/` | `bedlam_collate_fn`，包括 `[B, S] -> [B*S]` 的 Multi-HMR 分支展平 | 必须明确当前模型期望 `[B, S, ...]` 还是 `[B*S, ...]`；不要无分析地改变训练 batch contract |
| Supervision builder | `.reference/data_loading/supervision/` | `SMPLGTBuilder`、`perspective_projection`、`resize_K`，构建 slot UV 和 patch score maps | 必须匹配 patch size、图像分辨率、相机内参、joint 数量、slot mask 和坐标系 |
| Image folder dataset | `.reference/data_loading/image_folder_dataset.py` | 推理阶段图片文件列表和 manifest 风格输入 | 仅作为简单 inference 输入参考；不要替代正式训练 dataset |

### data_loading 预期 BEDLAM 数据布局

`BedlamDataset` 参考实现期望预处理后的 BEDLAM 数据大致为：

```text
processed_bedlam/
  Training/
    <scene>_<seq>/
      rgb/    frame_*.png
      depth/  frame_*.npy
      cam/    frame_*.npz
      smpl/   frame_*.pkl
  Test/
    <scene>_<seq>/
      rgb/    frame_*.png
      depth/  frame_*.npy
      cam/    frame_*.npz
      smpl/   frame_*.pkl
```

数据集路径应通过 `configs/path.yaml`、本地配置文件或环境变量提供，不要写死在核心代码中。

### data_loading batch schema 参考

如果迁移 BEDLAM loader，需要特别保护以下 batch contract：

| Key | Shape | 含义 |
|---|---|---|
| `images` | `[B, S, 3, img_size, img_size]` | VGGT/Scal3R 风格输入，direct square resize 后 ImageNet normalized |
| `img_mhmr` | `[B*S, 3, mhmr_size, mhmr_size]` | Multi-HMR 风格输入，contain+pad letterbox 后 ImageNet normalized |
| `gt_depth` | `[B, S, 1, img_size, img_size]` | metric depth，resize 到 `img_size` |
| `K_scal3r` | `[B, S, 3, 3]` | direct square resize 后的内参 |
| `K_mhmr` | `[B, S, 3, 3]` | Multi-HMR letterbox 后的内参 |
| `mhmr_letterbox_scale` | `[B*S, 2]` | letterbox scale 元数据 |
| `mhmr_letterbox_pad` | `[B*S, 2]` | letterbox pad 元数据 |
| `mhmr_orig_hw` | `[B*S, 2]` | 原图 `(H, W)` |
| `joints3d_cam` | `[B, S, M, 24, 3]` | camera coordinates 中 padding 后的 SMPL joints |
| `gt_pose` | `[B, S, M, 144]` | 24 joint 的 6D pose |
| `gt_betas` | `[B, S, M, 10]` | SMPL betas |
| `gt_cam_trans` | `[B, S, M, 3]` | camera-space translation |
| `smpl_mask` | `[B, S, M]` | valid-person slot mask |

如果本项目后续训练接口采用不同 key 或 shape，必须显式写 adapter，不要让模型代码隐式依赖参考项目命名。

### data_loading 与 VGGT 集成要求

将 data_loading 思路适配到 `vggt_human/` 或训练脚本时，需要遵守：

1. 先检查当前项目已有 dataset、dataloader、preprocessing 和 batch contract。
2. 明确当前任务是训练、可视化、导出、smoke test，还是 inference-only。
3. 数据路径、SMPL 模型路径和 ckpt 路径必须来自配置或环境变量，不要写死本机路径。
4. 如果新增真实数据加载代码，优先放到项目已有数据模块或 `vggt_human/` 下合适位置，不要直接从 `.reference/data_loading/` import。
5. 如果只需要图像目录推理，不要迁移完整 BEDLAM loader。
6. 如果只需要 SMPL supervision，不要迁移无关的 image folder 或 BEDLAM indexing 代码。
7. 如果引入 `smplx`，必须说明它只在构建真实 SMPL joints/mesh 时需要，并保证 baseline import 不受影响。
8. 不要提交 BEDLAM 数据、SMPL 模型文件、checkpoint 或本机路径配置。

### data_loading 验证要求

涉及 data_loading 适配的代码改动完成后，优先做以下验证：

1. import 检查。
2. 使用最小本地样本或 mock 数据检查 dataset `__getitem__`。
3. 检查 `DataLoader` 加 `collate_fn` 后的 batch keys 和 shape。
4. 检查 `images`、`img_mhmr`、`K_scal3r`、`K_mhmr`、`smpl_mask` 的 dtype、device 和 shape。
5. 如果使用 SMPL fallback，需要明确说明它只适合 smoke test，不代表真实 joint supervision。
6. 如果使用 `SMPLGTBuilder`，检查 slot UV、score map、joint 坐标和 mask 的 shape 与取值范围。

## 冲突处理

如果不同参考项目给出互相冲突的设计：

1. 默认保留当前 VGGT 实现。
2. 优先做 isolated experiment，而不是改动共享基础结构。
3. 如果冲突会影响研究方向或导致大范围重写，需要先向用户说明风险。
4. 不要在没有明确理由的情况下混合多个参考项目的设计。
