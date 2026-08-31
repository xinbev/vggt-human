# 论文主表指标评测计划

## 1. 图片的含义与本项目任务边界

用户提供的图片是目标论文表格的**格式和指标参考**，不是要复制的实验结果或执行指令。表格中的核心指标是：

- 3DPW：PA-MPJPE、MPJPE、PVE。
- EMDB-1：PA-MPJPE、MPJPE、PVE。
- `Crop-free / Detection-free / Intrinsic-free`：方法输入能力说明，不是数值指标。

当前项目应先输出自己的 VGGT-Omega 数字，再决定如何与论文表中的其他方法对齐；不能直接把图片中的数字填入本项目表格。

## 2. 当前仓库已有能力

### 2.1 3DPW SMPL-base 评测

入口：`scripts/eval/evaluate_3dpw_smpl_base_metrics.sh`。

底层实现：`scripts/eval/evaluate_3dpw_smpl_base_metrics.py`。

它会生成：

```text
outputs/eval/3dpw_smpl_base/3dpw_smpl_base_metrics.json
outputs/eval/3dpw_smpl_base/3dpw_smpl_base_metric_rows.csv
```

其中 `pa_mpjpe_mm`、`mpjpe_mm`、`pve_mm` 已经是毫米，适合直接放进论文表格。

### 2.2 EMDB-1 / 3DPW 全系统评测

入口：`scripts/eval/evaluate_hmr4d_full_system.sh`。

底层实现：`scripts/eval/evaluate_hmr4d_smpl_metrics.py`。

该入口支持 `emdb1 emdb2 rich 3dpw`，会生成：

```text
outputs/eval/hmr4d_smpl_metrics/<dataset>/<dataset>_smpl_metrics.json
outputs/eval/hmr4d_smpl_metrics/<dataset>/<dataset>_smpl_metrics_rows.csv
```

JSON 同时保存米和毫米字段，例如 `pa_mpjpe_m` 与 `pa_mpjpe_m_mm`。论文表格统一使用毫米即可。

当前协议名为 `project_native_smpl24`：

- MPJPE：相机坐标下以根关节对齐后计算。
- PA-MPJPE：对 24 个关节做 Procrustes 对齐后计算。
- PVE：相机坐标下顶点误差；同时保留 camera-space MPJPE 和 acceleration error 作为诊断。

正式论文前，需在论文/代码对照后确认这套对齐定义与目标论文完全一致。特别是不同代码常把 PVE 也做 pelvis alignment，不能只看名称相同就认为口径相同。

## 3. 推荐的实际评测顺序

本节保留原有 HMR4D 全系统评测流程，便于回溯 baseline；你本次确定的“NLF detector + standalone SMPL temporal refiner、无 HSI/TRSTR/sidecar”请直接执行第 7 节专用入口。

### 阶段 A：准备评测输入

1. 在 `configs/path.yaml` 中确认服务器路径：
   - VGGT-Omega checkpoint；
   - `assets.smpl_model_dir`；
   - EMDB hmr4d support；
   - 3DPW hmr4d support；
   - RGB 抽帧目录（本实验不使用检测/跟踪 sidecar）。
2. 只抽取 EMDB-1 和 3DPW RGB 帧，不调用会额外生成检测/跟踪 sidecar 的全量准备脚本：

```bash
DATASETS="emdb1 3dpw" bash scripts/preprocess/prepare_hmr4d_eval_frames_only.sh
```

3. 先做接口检查，不加载模型：

```bash
DATASET=emdb1 bash scripts/diagnostics/check_hmr4d_eval_data_interface.sh
DATASET=3dpw bash scripts/diagnostics/check_hmr4d_eval_data_interface.sh
```

建议先用 `MAX_SEQUENCES=1 MAX_FRAMES=120` 做小样本检查，再做全量准备。

### 阶段 B：跑一个 checkpoint 的正式指标

全系统（包含 HSI 输出，评测器默认 `--prefer-hsi`）：

```bash
CHECKPOINT=/home/zhw/lab_users/xyb/home/projects/vggt-human/outputs/train/<run>/checkpoint_latest.pt \
DATASETS="emdb1 3dpw" \
OUT_ROOT=outputs/eval/paper_main/<run> \
bash scripts/eval/evaluate_hmr4d_full_system.sh
```

同一个评测器也可以输出 base 分支，便于做论文中的方法/消融对比：

```bash
PREFER_HSI=false \
CHECKPOINT=/home/zhw/lab_users/xyb/home/projects/vggt-human/outputs/train/<run>/checkpoint_latest.pt \
DATASETS="emdb1 3dpw" \
OUT_ROOT=outputs/eval/paper_main/<run>_base \
bash scripts/eval/evaluate_hmr4d_full_system.sh
```

脚本还支持通过 `TRAIN_CONFIG`、`PATH_CONFIG`、`BATCH_SIZE` 和 `NUM_WORKERS` 显式固定配置；默认值与原脚本相同，baseline 行为不变。

如果先只验证链路，可加 `MAX_WINDOWS=20`；论文最终数字必须去掉该限制并记录完整窗口数。

### 阶段 C：为论文表格建立可比的行

至少需要固定以下行：

1. 当前 VGGT-Omega base/SMPL 输出；
2. 当前 VGGT-Omega + HSI 输出；
3. 如果论文要展示组件贡献，再加去掉某个模块的 ablation，但每一行必须使用相同数据划分、帧、评测协议和 checkpoint 选择规则。

每一行记录：

```text
method, checkpoint, dataset, num_windows, num_metric_rows,
PA-MPJPE (mm), MPJPE (mm), PVE (mm), metric_protocol
```

不要把 `transl_l2_mm`、`cam_mpjpe_no_align_mm` 等诊断量替换进主表；它们用于解释全局平移是否是主要误差来源。

## 4. 图片中三个能力列如何映射到当前系统

| 列 | 当前系统判断 | 原因 |
|---|---|---|
| Crop-free | 可以主张“全帧输入/非人物裁剪”，但需在论文中明确 | HMR4D adapter 读取完整 RGB 帧，SMPL query 通过 patch/query pooling；不是把每个人先裁成独立图片 |
| Detection-free | 当前默认不能主张 | 正式评测会读取 YOLO/SAM2/BoostTrack sidecar；没有 sidecar 时只能 fallback 到标签框，不能算 detection-free |
| Intrinsic-free | 推理路径基本可以主张，但需做消融确认 | 模型从 camera head 预测相机参数；评测调用没有把 GT K 作为模型输入传入。但 HSI 几何会使用模型预测内参，不能把“使用预测内参”和“完全不需要相机建模”混为一谈 |

## 5. 当前必须核对的可比性风险

1. `vggt_omega/data/hmr4d_eval.py` 中的 `EMDB1_NAMES` 当前只有 17 个序列名，而图片标题写的是 `EMDB-1 (24)`。这说明当前代码覆盖数与目标表可能不一致，必须在服务器 support 文件中核对实际 24 个评测序列，不能直接宣称复现了图片协议。
   图片中的 `3DPW (14)` 也应理解为目标论文采用的评测子集数量；实际项目运行时要以 support label 中成功加载且 `eval_mask` 有效的序列/帧为准。
2. 当前 HMR4D 评测的 `PVE` 是相机坐标下的顶点误差；3DPW SMPL-base 评测器则先做 pelvis alignment。两者不能直接混成同一列，除非统一实现一个最终协议。
3. HMR4D 评测按窗口读取序列，并按有效帧聚合；主表应报告有效 metric row 数，而不是只报告窗口数。
4. 评测默认会选择 HSI refined prediction（`--prefer-hsi`）。比较 base 与 HSI 时，必须分别跑同一个 checkpoint/config，并在表注中说明输出分支。
5. RICH 暂不应放入这张表：当前代码明确拒绝在缺少 SMPL-X 到 SMPL 转换资产时生成 RICH 假指标。

## 6. 服务器端完成正式评测所需材料

- 一个明确的 checkpoint（建议固定 `checkpoint_top01.pt` 或 `checkpoint_latest.pt`，并记录选择理由）；
- `checkpoints/body_models/smpl` 中完整的 SMPL neutral 模型；
- EMDB hmr4d support 文件（至少 `emdb_vit_v4.pt`）；
- 3DPW hmr4d support 文件（至少 `test_3dpw_gt_labels.pt` 及其 bbox/kp2d 预处理文件）；
- 评测 RGB 帧目录；
- YOLO/SAM2/BoostTrack 所需权重和 sidecar；
- 服务器上的 Python 环境、CUDA 和项目依赖。

Windows 本地只能做静态检查；真正的 forward、SMPL 解码和全量指标必须在 Linux 服务器执行。

本次本地检查已完成 Python 语法编译；尝试运行 `--help` 时因本机未安装 PyTorch（`ModuleNotFoundError: torch`）而无法执行模型入口，这是预期的环境限制。

## 7. 按当前新目标的专用入口

当前新增的专用脚本是：

```text
scripts/eval/evaluate_nlf_temporal_metrics.sh
scripts/eval/evaluate_nlf_temporal_metrics.py
configs/eval_nlf_temporal.yaml
```

它固定执行：

```text
RGB -> VGGT -> NLF internal detector -> TemporalSMPLRefiner
```

并固定关闭 HSI、TRSTR、外部 sidecar。评测使用时序 checkpoint 的窗口长度，并按 `(视频, 原始帧号)` 去重，避免滑动窗口把同一帧重复计入。每个数据集输出 base 与 temporal-refined 两套指标，论文主表应使用 `temporal` 下的 `pa_mpjpe_mm`、`mpjpe_mm`、`pve_mm`。

服务器运行示例：

```bash
CHECKPOINT=/home/zhw/lab_users/xyb/home/projects/vggt-human/checkpoints/vggt_omega_1b_512.pt \
TEMPORAL_CHECKPOINT=/home/zhw/lab_users/xyb/home/projects/vggt-human/outputs/train/smpl_temporal_refiner_emdb_3dpw_v1/checkpoint_best.pt \
DATASETS="emdb1 3dpw" \
OUT_ROOT=outputs/eval/paper_nlf_temporal \
bash scripts/eval/evaluate_nlf_temporal_metrics.sh
```

输出位置：

```text
outputs/eval/paper_nlf_temporal/emdb1/emdb1_nlf_temporal_metrics.json
outputs/eval/paper_nlf_temporal/3dpw/3dpw_nlf_temporal_metrics.json
```

这里的 `CHECKPOINT` 必须替换成服务器上实际存在的 VGGT/实验 checkpoint，`TEMPORAL_CHECKPOINT` 必须是 `format=smpl_temporal_refiner_v1` 的独立时序模型 checkpoint。若只有 pose-only stabilizer v2 checkpoint，它与当前 adapter 接口不同，不能直接冒充 TemporalSMPLRefiner。
