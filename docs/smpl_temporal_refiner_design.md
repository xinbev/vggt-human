# 独立 SMPL 时序优化器（v1）

## 目标与边界

`TemporalSMPLRefiner` 是一个离线、双向的 SMPL 残差去抖模型：输入同一 track 的连续单帧 SMPL 预测，输出受限的 `pose` 与 `translation` 修正。它不读取或修改 VGGT、HSI、TRSTR 的参数，也不修改 `betas`。

当前主链保持不变：

```text
单帧 NLF / HSI scale / TRSTR spatial
  -> 原始 pose_6d, transl_cam, betas, track_id
  -> 可选 TemporalSMPLRefiner adapter
  -> 时序 refined pose_6d, transl_cam；betas 原样保留
```

该模型使用未来帧，因此仅适用于离线视频或带延迟的 clip 推理。实时场景需要另训因果版本，不能直接复用本 checkpoint。

## 数据接口

实现文件为 [smpl_temporal_pickle.py](C:/Users/ROG/PycharmProjects/vggt-omega/vggt_omega/data/smpl_temporal_pickle.py)。它将 EMDB 和 3DPW 都转换为单人、连续窗口：

| 输出 | shape | 来源 |
| --- | --- | --- |
| `target_pose_6d` | `[S,144]` | EMDB `poses_root + poses_body`；3DPW `poses[person]` |
| `target_transl` | `[S,3]` | `smpl.trans` / `trans[person]` |
| `target_betas` | `[S,10]` | 序列级 beta 的重复；仅 passthrough |
| `valid_mask` | `[S]` | `good_frames_mask` / `campose_valid[person]` |
| `intrinsics` | `[S,3,3]` | 原始 K 重复 |
| `camera_extrinsics` | `[S,4,4]` | 原样保留，尚不假设变换方向 |

训练/验证按完整原始 pkl sequence 切分：同一人物的重叠窗口、以及同一 3DPW pkl 内的所有人都会在同一侧，避免共享动作和相机序列泄漏到验证集。

## 多人、身份与坐标的当前约束

3DPW 的一个 pkl 中可以有多人。加载器按 `poses[person_idx]`、`trans[person_idx]`、`betas[person_idx]` 和 `campose_valid[person_idx]` 拆成**独立 person track**；同一帧的两人不会拼接、不会互相当作邻帧，也不会混用 beta。一个训练样本始终只包含一个人的 `S` 帧 SMPL。

这符合 v1 的目标（消除每个检测 track 自身的闪动），但它不处理人与人相互遮挡、碰撞或 ID switch。推理 adapter 会以 `track_id` 聚合跨帧 slot；一帧同一 ID 出现多个检测时，该帧不应用时序修正，保留单帧结果。

另一个必须在正式接回前确认的风险是平移坐标系：训练 pkl 的 `trans` 和 `camera_extrinsics/cam_poses` 都按原样保留，v1 不会猜测外参方向或擅自求逆。当前训练名称使用通用 `target_transl`，不是保证为 `transl_cam`。因此，在把 adapter 接到 HSI/TRSTR 的 `transl_cam` 前，必须确认两者都在同一坐标系；若 pkl 的 `trans` 是 world coordinate，则应在推理上层先用已验证的外参将 `transl_cam` 转到 world，再送入 refiner、最后转回 camera。未确认前不能直接启用 translation refinement。

训练启动时会把扫描结果打印并保存到 `outputs/train/.../data_summary.json`：每个数据集的 pkl 文件数、人物轨迹数、总/有效/无效帧数，以及 train/val 的有效窗口数。数量不合理时应先停止，而不是直接训练。

## 训练构造

GT 不直接作为输入。每个窗口的全部帧都会加入：低频 translation drift、逐帧 translation jitter、少量 outlier、旋转 drift 与 rotation jitter。另有 15% clean clip，监督模型保持 no-op。

训练损失是 GT pose/translation 误差为主，GT velocity/acceleration 一致性为辅，并有 `no_worse` 项约束 refined 结果不能比扰动输入更差。它不是简单压小速度，避免把真实快速动作抹平。

## 启动与验收

先在服务器确认真实 pkl：

```bash
bash scripts/smoke/inspect_smpl_temporal_pickles.sh
bash scripts/smoke/check_smpl_temporal_refiner.sh
```

训练：

```bash
bash scripts/train/train_smpl_temporal_refiner_emdb_3dpw.sh
```

本地路径：

- `C:\Users\ROG\PycharmProjects\vggt-omega\scripts\train\train_smpl_temporal_refiner_emdb_3dpw.sh`
- `C:\Users\ROG\PycharmProjects\vggt-omega\outputs\train\smpl_temporal_refiner_emdb_3dpw_v1`

服务器路径：

- `/home/zhw/lab_users/xyb/home/projects/vggt-human/scripts/train/train_smpl_temporal_refiner_emdb_3dpw.sh`
- `/home/zhw/lab_users/xyb/home/projects/vggt-human/outputs/train/smpl_temporal_refiner_emdb_3dpw_v1`

训练会向 W&B 记录 `train/*`、`val/*`，包括 pose geodesic、translation L1、velocity、acceleration、输入到输出的 translation improvement 和 no-worse rate。应以 `checkpoint_best.pt` 为候选模型。

## 接回项目（训练验收后）

接入点是 [smpl_temporal_refiner.py](C:/Users/ROG/PycharmProjects/vggt-omega/vggt_omega/integrations/smpl_temporal_refiner.py) 的 `SMPLTemporalRefinementAdapter`。它要求离线 batch 内有稳定的 `track_ids`，可处理同一人物跨帧换 slot；缺失、重复匹配或无效位置保持原始单帧输出。

在服务器确认 checkpoint 对真实 NLF/TRSTR 输出不劣化后，再单独新增配置开关并挂到可视化/推理调用方；在此之前，主模型 forward 不会发生任何改变。

## 当前两阶段干扰课程

当前只启用两档，不预设第三档：

| 阶段 | epoch | translation drift / jitter / outlier | pose drift / jitter |
| --- | --- | --- | --- |
| small | 1–10 | 6 cm / 2.5 cm / 10 cm | 0.06 / 0.025 rad |
| medium | 11–结束 | 12 cm / 5 cm / 25 cm | 0.12 / 0.05 rad |

验证集固定使用 medium 干扰，保证不同 epoch 的验证曲线可直接比较。训练控制台、W&B 和 checkpoint 都会记录当前阶段与实际参数。medium 结果和真实序列可视化确认前，不增加第三档或继续扩大修正范围。
