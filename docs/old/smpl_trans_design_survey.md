# SMPL `trans` 设计调研：MetricHMSR、GVHMR、Human3D、NLF、PromptHMR

本文整理以下论文/代码库中 SMPL/SMPL-X 模块里的 `trans` / `transl` / translation 设计方式，并重点关注：它表示什么坐标系下的位置、网络是否直接回归它、如何从相机参数或中间量解码、训练时如何监督，以及对 `vggt-omega` 后续科研改造的启发。

调研对象：

- `C:\Users\ROG\PycharmProjects\vggt-omega\.paper\MetricHMSR.pdf`
- `C:\Users\ROG\PycharmProjects\GVHMR-main`
- `C:\Users\ROG\PycharmProjects\Human3D`
- `C:\Users\ROG\PycharmProjects\nlf-main`
- `C:\Users\ROG\PycharmProjects\PromptHMR`

## 总览结论

| 项目 | `trans` 的核心含义 | 坐标系 | 主要获得方式 | 是否直接回归绝对 3D 平移 | 对 VGGT/HSI 类方法的启发 |
|---|---|---|---|---|---|
| MetricHMSR / MetricHMR | 人体全局 metric 3D 位置 `t_global` | 相机/场景 metric 空间，论文强调全局 metric 位置 | 用 bbox、crop、相机内参形成 bounding camera ray map，再由 MLP head 输出 `t_global` | 是，论文把 `t_global in R^3` 作为显式输出 | 若要做 metric translation，应把内参、bbox/crop、ray cue 显式输入，而不是只让图像特征猜尺度 |
| GVHMR | 两套 translation：相机内 `transl_c` 与世界/重力对齐轨迹 `transl_w` | camera space 与 global/world/gravity-view | 相机内平移由 `pred_cam=[s, tx, ty]` + bbox + intrinsics 解码；全局轨迹用 local velocity rollout | 相机内不直接监督 `transl`，而监督可逆的 `pred_cam`；全局用速度表示再 rollout | 单帧可采用 camera-param 中间量；视频全局位移不要朴素回归绝对坐标，优先速度/残差/rollout |
| PromptHMR | SMPL-X 相机内 metric translation，之后可转世界系并优化 | camera space -> world space | location token 分别输出 2D offset 与 depth，再用 focal 解码成 `tx,ty,tz` | 不是直接输出 xyz，而是 `2D offset + depth` 解码 | 深度和横向偏移拆开更稳定；后处理可做 residual trans 优化 |
| NLF / MNLF | 官方 NLF 更偏 absolute 3D points；相关脚本里 `trans` 是 SMPL 初始平移并做 residual refinement | camera space 为主，也有用户脚本转 tracker/world-like 坐标 | 通过 full-perspective absolute reconstruction 或数据加载时把 SMPL `trans` rototranslate 到 camera；实验脚本中 `init_trans + delta_trans` | 官方 NLF 不以 HMR 式 SMPL `trans` head 为核心；用户脚本是 residual 修正 | 可借鉴“绝对 3D 点/点云约束 + 初值残差修正”，但要先统一坐标和尺度 |
| Human3D | 未发现 SMPL/SMPL-X `trans` 模块 | 不适用 | `rg` 未检索到 `SMPL/smpl/transl/global_orient/body_pose/betas` 相关实现 | 不适用 | 当前不适合作为 SMPL translation 设计参考 |

一句话归纳：这些项目几乎都避免“裸回归一个不带几何约束的 3D trans”。MetricHMSR 直接输出 metric `t_global`，但前提是显式引入相机 ray/bbox/crop metric cue；GVHMR 和 PromptHMR 更倾向先预测相机/深度中间量再解码；视频全局位移则倾向速度、rollout、后处理优化。

## 1. MetricHMSR / MetricHMR：把 `trans` 定义成 metric global position

### 设计定位

MetricHMSR 的目标不是传统弱透视 HMR 那种只得到相对人体形状和弱透视相机，而是恢复 metric human mesh 和 metric 3D scene。论文明确指出单目尺度歧义和 weak-perspective camera assumptions 会导致 metric scale 难恢复，因此它把人体局部姿态和全局 metric 平移解耦。

论文中的 SMPL 参数输出包括：

- pose：`theta in R^72`
- shape：`beta in R^10`
- translation：`t_global in R^3`

其中 `t_global` 被描述为人体的 metric 3D position / global metric position，而不是单纯的 crop 内弱透视相机参数。

### `trans` 如何获得

论文核心机制是引入 bounding camera ray map。这个 ray map 显式编码：

- human image position
- camera intrinsics
- bbox/crop 引入的裁剪和缩放影响

然后 HumanMoE / MLP heads 共同预测：

- local pose
- metric body shape
- metric 3D translation `t_global`

因此 MetricHMSR 里的 `trans` 更接近“有相机几何先验约束的直接 metric translation regression”。它不是没有几何输入地直接从图像 token 猜 xyz，而是把 bbox、crop、intrinsics 转成 ray cue 后再让网络学习。

### 设计启发

如果 `vggt-omega` 后续想借鉴 MetricHMSR 的 translation 设计，关键不是简单加一个 `Linear(..., 3)` 输出 `trans`，而是要同步引入 metric cue：

- full-image intrinsics：`fx, fy, cx, cy`
- crop/bbox 到 full image 的映射
- 每个目标人体的 ray / normalized camera coordinate
- 输出 `trans` 的坐标系定义：camera-space root translation，还是 scene/world global translation

风险是：如果训练数据没有可靠 metric supervision，或者 bbox/crop 与相机内参在数据管线里不一致，`t_global` 可能会学成 dataset bias，而不是稳定的 metric 平移。

## 2. GVHMR：`pred_cam` 解码相机内平移，local velocity 表示全局轨迹

GVHMR 的 translation 设计有两层，不能混成一个概念：

- `pred_smpl_params_incam["transl"]`：相机坐标系下的人体 SMPL 平移。
- `pred_smpl_params_global["transl"]`：全局/世界/重力对齐坐标系下的人体轨迹。

### 2.1 相机内 `transl`：由弱透视式 `pred_cam` 解码

GVHMR 不让网络直接输出相机内 xyz translation，而是输出 `pred_cam=[s, tx, ty]`。关键函数在：

- `C:\Users\ROG\PycharmProjects\GVHMR-main\hmr4d\utils\geo\hmr_cam.py:124`

逻辑为：

```python
def compute_transl_full_cam(pred_cam, bbx_xys, K_fullimg):
    s, tx, ty = pred_cam[..., 0], pred_cam[..., 1], pred_cam[..., 2]
    focal_length = K_fullimg[..., 0, 0]
    ...
    sb = s * bbx_xys[..., 2]
    cx = 2 * (bbx_xys[..., 0] - icx) / (sb + 1e-9)
    cy = 2 * (bbx_xys[..., 1] - icy) / (sb + 1e-9)
    tz = 2 * focal_length / (sb + 1e-9)
    cam_t = torch.stack([tx + cx, ty + cy, tz], dim=-1)
```

也就是说：

- `s` 控制深度：`tz = 2 * focal / (s * bbox_size)`
- bbox center 与 full-image principal point 的偏移修正 `x/y`
- 最终输出 camera-space `cam_t`

在 pipeline 中，相机内 SMPL 参数直接把这个解码结果作为 `transl`：

- `C:\Users\ROG\PycharmProjects\GVHMR-main\hmr4d\model\gvhmr\pipeline\gvhmr_pipeline.py:82`

```python
"transl": compute_transl_full_cam(model_output["pred_cam"], inputs["bbx_xys"], inputs["K_fullimg"])
```

### 2.2 训练监督：不直接监督 `transl_c`，而监督 `pred_cam`

GVHMR 的训练代码里保留了直接 `transl` L1 的注释，但实际采用了更稳定的方式：

- 先用 GT `transl` 反推出 GT `pred_cam`
- 再监督网络输出的 `pred_cam`

位置：

- `C:\Users\ROG\PycharmProjects\GVHMR-main\hmr4d\model\gvhmr\pipeline\gvhmr_pipeline.py:185`

关键逻辑：

```python
# Instead of supervising transl, we convert gt to pred_cam (prevent divide 0)
gt_pred_cam = get_a_pred_cam(gt_transl, inputs["bbx_xys"], inputs["K_fullimg"])
transl_c_loss = F.mse_loss(pred_cam, gt_pred_cam, reduction="none")
```

这说明 GVHMR 认为相机内绝对 translation 直接监督可能数值不稳定，尤其深度和 bbox scale 强耦合。用 `pred_cam` 中间量监督，可以把相机几何和 bbox 尺度关系显式约束起来。

### 2.3 全局/world `transl`：不是直接回归位置，而是 local translation velocity

GVHMR 的 global trajectory 表示使用 `local_transl_vel`：

- `C:\Users\ROG\PycharmProjects\GVHMR-main\hmr4d\model\gvhmr\utils\endecoder.py:127`
- `C:\Users\ROG\PycharmProjects\GVHMR-main\hmr4d\utils\geo\hmr_global.py:132`

编码时把绝对世界平移转换成 SMPL/local 坐标系下的速度；解码时使用 `rollout_local_transl_vel` 重新积分出全局轨迹：

- `C:\Users\ROG\PycharmProjects\GVHMR-main\hmr4d\model\gvhmr\pipeline\gvhmr_pipeline.py:378`

```python
transl = rollout_local_transl_vel(local_transl_vel, global_orient)
global_orient, transl, _ = get_tgtcoord_rootparam(global_orient, transl, tsf="any->ay")
```

训练中 `transl_w_loss` 也是先由 local velocity rollout 出 `pred_transl_w`，再与 GT world translation 做 L1：

- `C:\Users\ROG\PycharmProjects\GVHMR-main\hmr4d\model\gvhmr\pipeline\gvhmr_pipeline.py:290`

### 设计启发

GVHMR 给出的信息很清楚：

- 单帧/相机内 `transl`：适合用 `pred_cam + bbox + intrinsics` 解码。
- 视频/全局 `transl`：适合用 velocity representation，再 rollout。
- 训练稳定性：不要急着直接对 xyz translation 做强监督，尤其深度方向容易不稳定。

对 `vggt-omega` 来说，如果当前任务是单帧人体相机内平移，GVHMR 的 `compute_transl_full_cam` 类设计很适合作为 baseline-preserving experimental branch；如果是视频跨帧全局轨迹，则应考虑 local velocity / residual displacement，而不是每帧独立回归世界坐标。

## 3. PromptHMR：location token 输出 2D offset + depth，再解码成 metric `transl`

PromptHMR 的 SMPL-X decoder 明确把 location 预测拆成两个 head：

- `transl_head`：输出 2D translation / image-plane offset，维度为 2
- `depth_head`：输出 depth，维度为 1

位置：

- `C:\Users\ROG\PycharmProjects\PromptHMR\prompt_hmr\models\components\smpl_decoder.py:47`

```python
self.transl_head = MLP(transformer_dim, smpl_head_hidden_dim, 2, smpl_head_depth)
self.depth_head = MLP(transformer_dim, smpl_head_hidden_dim, 1, smpl_head_depth)
```

forward 中：

- `loc_token` 负责 location
- `depth_c = depth_head(loc_token) + init_depth`
- `transl_c = transl_head(loc_token) + init_transl`
- `transl = decode_transl(cam_int, transl_c, depth_c)`

位置：

- `C:\Users\ROG\PycharmProjects\PromptHMR\prompt_hmr\models\components\smpl_decoder.py:85`

### 解码公式

`decode_transl` 的逻辑在：

- `C:\Users\ROG\PycharmProjects\PromptHMR\prompt_hmr\models\components\smpl_decoder.py:93`

```python
px, py = transl.unbind(-1)
pz = depth.unbind(-1)[0]
if self.inverse_depth:
    pz = 1 / (pz + 1e-6)
tx = px * pz
ty = py * pz
tz = pz * focal / 1000
t_full = torch.stack([tx, ty, tz], dim=-1)
```

含义是：

- 网络预测的是 normalized 横向/纵向偏移 `px, py` 和深度变量 `pz`
- 横向平移随深度缩放：`tx=px*pz`, `ty=py*pz`
- z 平移再按 focal 缩放：`tz=pz*focal/1000`

这与 GVHMR 的 `pred_cam -> cam_t` 思路相似：translation 不作为完全自由的 xyz 变量，而是通过相机几何相关的中间量解码。

### SMPL-X root/transl 约定

PromptHMR 还有一个重要开关：

- `C:\Users\ROG\PycharmProjects\PromptHMR\prompt_hmr\models\phmr.py:104`

如果 `cfg.MODEL.TRANSL == 'root'`，它会在传入 SMPL-X 前从 `transl` 里减去 SMPL-X root joint。这说明 PromptHMR 区分了：

- `transl` 表示模型整体平移
- 或者让 `transl` 表示 root/pelvis 对齐后的平移

这类 root/pelvis 约定必须在集成时写清楚，否则同一个 `trans` 会在不同 body model wrapper 中差一个 pelvis offset。

### camera -> world 与后处理 residual

视频 pipeline 中，PromptHMR 先保存 camera-space `smplx_cam["trans"]`，再通过相机世界位姿转换到 `smplx_world["trans"]`：

- `C:\Users\ROG\PycharmProjects\PromptHMR\pipeline\world.py:13`
- `C:\Users\ROG\PycharmProjects\PromptHMR\pipeline\world.py:97`
- `C:\Users\ROG\PycharmProjects\PromptHMR\pipeline\world.py:122`

后处理阶段不是重新从零估计 translation，而是以 world translation 为初值，优化 residual：

- `C:\Users\ROG\PycharmProjects\PromptHMR\pipeline\postprocessing.py:42`
- `C:\Users\ROG\PycharmProjects\PromptHMR\pipeline\postprocessing.py:82`
- `C:\Users\ROG\PycharmProjects\PromptHMR\pipeline\postprocessing.py:214`

```python
transl_init = ...
transl = torch.zeros_like(transl_init)
final_transl = transl_init + transl
```

### 设计启发

PromptHMR 适合作为“单帧 translation head 的更现代版本”参考：

- 用单独 location token 管 translation/depth。
- 把 image-plane offset 和 depth 拆开预测。
- 用 focal/intrinsics 解码 camera-space metric translation。
- 视频和 world 场景下，先 transform，再 residual refinement。

对 VGGT 改造而言，PromptHMR 的风险在于 `tz=pz*focal/1000` 里的尺度单位要和数据集、相机内参、SMPL mesh 单位完全一致；否则模型可能表面收敛，实际 metric scale 漂移。

## 4. NLF / MNLF：官方 NLF 是 absolute 3D reconstruction，相关脚本中 `trans` 多作为初值残差修正

NLF 与 HMR 系列不完全同构。它的核心更偏向从 2D/3D 点预测中重建 absolute 3D coordinates，而不是标准 HMR 那样输出 pose、shape、camera/trans。

### 4.1 官方 NLF：用 full perspective reconstruct absolute coordinates

关键函数：

- `C:\Users\ROG\PycharmProjects\nlf-main\nlf\pt\ptu3d.py:9`
- `C:\Users\ROG\PycharmProjects\nlf-main\nlf\pt\models\nlf_model.py:320`

`reconstruct_absolute` 输入包括：

- `coords2d`
- `coords3d_rel`
- `intrinsics`
- weak/full perspective 标志

它先用 intrinsics 把 2D 点转成 normalized camera coordinates，再求 reference point 的绝对位置，最后得到 absolute 3D coordinates：

```python
inv_intrinsics = torch.linalg.inv(intrinsics.to(coords2d.dtype))
coords2d_normalized = (to_homogeneous(coords2d) @ inv_intrinsics.transpose(1, 2))[..., :2]
...
ref = reconstruct_ref_fullpersp(...)
coords_abs_3d_based = coords3d_rel + ref.unsqueeze(1)
```

这说明 NLF 官方主体并不是“回归一个 SMPL `trans` head”，而是先恢复绝对 3D 点。若后续要接 SMPL，`trans` 更像由 absolute geometry 或数据加载转换得到的参数。

### 4.2 parametric 数据加载：把 SMPL `trans` 转到 camera

在 parametric loader 中，SMPL 参数会经过 body model 的 `rototranslate` 转换：

- `C:\Users\ROG\PycharmProjects\nlf-main\nlf\pt\loading\parametric.py:166`
- `C:\Users\ROG\PycharmProjects\nlf-main\nlf\pt\loading\parametric.py:209`

核心逻辑：

```python
pose_cam, trans_cam = bm.rototranslate(
    cam.R, cam.t / 1000 / scale, pose, shape, trans, kid_factor, post_translate=False
)
...
trans=np.float32(trans_cam)
```

也就是说，这里的 `trans` 是经过相机外参、单位缩放等处理后的 camera-space SMPL translation。

### 4.3 MNLF / new_try 用户脚本：`init_trans + delta_trans`

`nlf-main` 里有大量 MNLF / test 脚本会读取 `nlf_data["trans"]` 作为初始 SMPL 平移，然后预测或优化残差：

- `C:\Users\ROG\PycharmProjects\nlf-main\new_try\net.py:123`
- `C:\Users\ROG\PycharmProjects\nlf-main\new_try\net.py:129`
- `C:\Users\ROG\PycharmProjects\nlf-main\MNLF\combined_centroid_optimization.py:309`
- `C:\Users\ROG\PycharmProjects\nlf-main\MNLF\combined_centroid_optimization.py:312`

典型模式：

```python
delta_trans = self.head_trans(x)
output_trans = init_trans + delta_trans
```

或：

```python
refined_trans = initial_trans + delta_trans
```

这些脚本更像“拿 NLF/已有结果作为初值，再用点云、投影、Chamfer、centroid 等约束做残差修正”的实验，而不是 NLF 官方主干的 translation head 设计。

### 设计启发

NLF 对 VGGT 的启发主要不是 head 结构，而是几何约束方式：

- 若已有 absolute 3D points 或 point cloud，可以把 SMPL `trans` 作为待修正变量。
- 用 `init_trans + delta_trans` 比从零预测绝对 translation 更稳。
- 但必须先统一坐标系：camera、world、tracker、SMPL canonical 的原点和单位。

风险是 `nlf-main` 里用户实验脚本很多，命名和坐标约定不统一，不能把这些脚本整体视为官方 NLF 设计。真正可借鉴的是 residual refinement 思想，而不是直接移植某个 `MNLF\t*.py`。

## 5. Human3D：未发现 SMPL translation 设计

对 `C:\Users\ROG\PycharmProjects\Human3D` 做了关键词检索：

- `SMPL`
- `smpl`
- `transl`
- `global_orient`
- `body_pose`
- `betas`

未检索到相关实现。该仓库从文件结构看更像 Human3D / Mask3D 风格的 3D 点云人体实例分割或场景理解项目，而不是 SMPL/HMR 参数回归项目。

因此当前不能从 Human3D 中整理出 SMPL `trans` 模块设计。若后续要继续查它对人体实例中心、点云坐标或场景坐标的处理，可以作为“点云空间人体定位”参考，但不应把它列为 SMPL translation 参考。

## 横向比较：几种 `trans` 设计范式

### A. 直接 metric translation regression

代表：MetricHMSR。

特点：

- 输出 `t_global in R^3`
- 目标是 metric position
- 必须有 bbox/crop/intrinsics/ray map 这类 metric cue

适用条件：

- 数据有可靠 metric GT
- 相机内参与 crop 映射可信
- 模型输入里有足够几何提示

不建议在缺少 metric supervision 时裸用。

### B. camera-parameter intermediate -> camera-space translation

代表：GVHMR。

特点：

- 预测 `pred_cam=[s, tx, ty]`
- 用 bbox size 和 focal length 解码 `tz`
- 用 bbox center 和 principal point 修正 `x/y`

适合单帧 HMR 或 crop-based pipeline。优点是兼容传统 HMR/CLIFF 思路，也容易保留 baseline。

### C. image-plane offset + depth -> camera-space translation

代表：PromptHMR。

特点：

- `transl_head` 预测 2D offset
- `depth_head` 预测 depth
- 用 focal 解码成 `tx, ty, tz`

适合希望显式解耦横向位置和深度的模型。若 VGGT/HSI 特征已经有较强几何感知能力，这种设计可能比单个 `pred_cam` 更灵活。

### D. global trajectory as velocity + rollout

代表：GVHMR。

特点：

- 不逐帧独立回归全局位置
- 表示 local translation velocity
- 通过 rollout 恢复全局轨迹

适合视频人体运动。优点是更平滑、更符合运动连续性；缺点是初始位置和累积误差需要额外处理。

### E. residual translation refinement

代表：PromptHMR postprocessing、NLF/MNLF 实验脚本。

特点：

- 先有 `transl_init`
- 优化或预测 `delta_trans`
- 最终 `transl = transl_init + delta_trans`

适合已有粗定位但要利用 2D/3D/点云/接触/地面约束进一步修正的场景。

## 对 `vggt-omega` 的建议

### 1. 先明确本项目 `trans` 要表示什么

在实现任何新 head 前，建议先固定下面四个约定：

- `trans` 是 camera-space 还是 world-space？
- `trans` 对齐 SMPL root joint、pelvis，还是 body model 原点？
- 单位是 meter、millimeter，还是归一化 crop 单位？
- 输入 bbox/crop 是 full-image 坐标还是 resize 后坐标？

这些约定应写在配置或模块 docstring 中，否则后续损失、可视化、评估会互相打架。

### 2. 保留 VGGT baseline，不直接替换

符合本仓库实验纪律，建议新增配置开关，例如：

- `translation_head: baseline`
- `translation_head: pred_cam_decode`
- `translation_head: offset_depth_decode`
- `translation_head: ray_metric`
- `translation_refine: none/residual/temporal_velocity`

不要直接删除或覆盖已有 baseline translation 路径。

### 3. 单帧优先比较两个轻量实验

如果当前目标是单帧 SMPL 平移估计，建议先做两个最小可验证分支：

1. GVHMR-style：预测 `pred_cam=[s,tx,ty]`，通过 bbox + intrinsics 解码 camera `transl`。
2. PromptHMR-style：预测 `2D offset + depth`，通过 focal 解码 camera `transl`。

这两个分支都比直接 xyz regression 更有几何约束，也更容易做 shape/dtype/forward smoke test。

### 4. 若追求 metric/global，必须补 ray/intrinsics/crop 特征

MetricHMSR 的关键不是“输出 `t_global`”，而是：

- bounding camera ray map
- intrinsics
- bbox/crop 几何
- local pose 与 global metric position 解耦

如果 `vggt-omega` 要做类似 metric translation，需要先确认数据管线里这些信息能稳定提供。

### 5. 视频/跨帧不要每帧裸回归世界坐标

如果目标包含视频序列或跨帧轨迹，建议优先参考 GVHMR：

- local translation velocity
- rollout
- static joint / static camera postprocess
- residual smoothing

绝对世界位置可以作为 rollout 后的监督目标，但不建议作为每帧独立 head 的唯一训练目标。

## 科研和工程风险

1. 坐标系风险：camera/world/tracker/gravity-view/root/pelvis 混用会让 `trans` 数值看似合理但投影错误。
2. 尺度风险：`focal/1000`、`cam.t / 1000 / scale`、SMPL meter/mm 单位不统一会造成深度整体偏移。
3. bbox/crop 风险：GVHMR/MetricHMSR 类方法高度依赖 bbox center、bbox size、crop resize 与 full image intrinsics 的一致性。
4. 监督风险：直接 xyz loss 可能在深度方向不稳定；GVHMR 通过监督 `pred_cam` 避免部分数值问题。
5. 数据风险：若训练集没有可靠 metric GT，MetricHMSR 式 `t_global` 可能学到相机/数据集先验，而不是真实 metric translation。
6. 集成风险：参考仓库中的 body model wrapper 对 `transl` 是否包含 pelvis/root offset 不一致，移植前必须写最小 forward/projection 检查。

## 推荐后续验证 checklist

若后续在 `vggt_human/` 中实现实验 translation head，建议至少验证：

- import 检查：新模块可独立导入。
- 最小 forward：输入 batch、bbox、intrinsics 后输出 `transl` shape 为 `(B, 3)` 或 `(B, T, 3)`。
- dtype/device：输出与主模型特征同 device、同合理 dtype。
- projection sanity：SMPL joints + `transl` 投影回图像后与 bbox/crop 位置方向一致。
- baseline switch：关闭新 head 时原 VGGT baseline 行为不变。
- 单位检查：1m/1000mm/focal scaling 在可视化中不会让人体飘到相机后方或远离 bbox。

