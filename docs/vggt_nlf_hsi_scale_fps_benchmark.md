# VGGT + NLF / HSI Scale 稳态 FPS 评测

## 评测目标

比较以下两条真实推理路径：

1. `VGGT + NLF`：一次 VGGT forward，包括 camera/depth heads 与 NLF detector/SMPL。
2. `VGGT + NLF + HSI scale`：基础 VGGT+NLF forward、人体表面解析粗尺度、第二次
   VGGT+HSI residual scale/bias forward，以及最终尺度组合。

第二次 forward 复用第一次的 NLF SMPL 输出，不重复执行 NLF detector。当前模型接口
不能从第一次 forward 直接复用 VGGT 中间 scene tokens，因此 HSI residual 仍需要第二次
VGGT forward。该实现与已接受的 coarse–residual HSI scale 逻辑一致，同时去除了无意义
的第二次 NLF 计算。

本评测不包含 TRSTR、旧 translation alignment、contact refinement、grounding、时序 HSI、
可视化和精度指标计算。VGGT/NLF baseline 行为没有被修改。

## 输入与张量约定

- 输入：预先读取、resize 并搬到 GPU 的 RGB tensor。
- shape：`[1, S, 3, H, W]`，默认 `S=100`。
- dtype/device：输入为 `float32/CUDA`；模型内部沿用现有 autocast 行为。
- NLF：detector 模式，不使用 GT box 或预处理 sidecar。
- HSI 输入深度：`[1, S, Hd, Wd]` 的 Z-depth。
- 人体：NLF 输出的 SMPL pose、shape、camera-space root translation。
- 尺度：逐帧解析粗尺度；失败帧使用当前 clip 有效尺度的 log-median。整个 clip 都失败时
  数值回退为 1，并在 JSON 的 `validation` 中记录。

## 计时边界

FPS 使用：

```text
FPS = S / mean(synchronized wall-clock latency)
```

每次正式迭代前后均执行 `torch.cuda.synchronize()`。因此时间包括 Python 调度、NLF
detector、GPU kernel、SMPL surface decode、解析尺度和 HSI scale/bias，而不是只统计异步
kernel 提交时间。

明确排除：

- 配置读取、模型构造和 checkpoint 加载；
- 图片读取、解码、resize、stack 和 CPU→GPU 传输；
- NLF TorchScript 第一次惰性加载；
- warm-up；
- 输出校验及 JSON/CSV 写盘。

## 服务器运行

本地脚本：

```text
C:\Users\ROG\PycharmProjects\vggt-omega\scripts\eval\benchmark_vggt_nlf_hsi_scale_fps.sh
```

同步到服务器后：

```text
/home/zhw/lab_users/xyb/home/projects/vggt-human/scripts/eval/benchmark_vggt_nlf_hsi_scale_fps.sh
```

执行：

```bash
bash scripts/eval/benchmark_vggt_nlf_hsi_scale_fps.sh
```

默认使用 GPU 7、walking 连续 100 帧、512 输入分辨率、3 次 warm-up 和 10 次正式测量。
可覆盖：

```bash
CUDA_VISIBLE_DEVICES_VALUE=7 \
FRAMES_DIR=/path/to/contiguous/frames \
NUM_FRAMES=100 \
WARMUP=5 \
REPEATS=20 \
bash scripts/eval/benchmark_vggt_nlf_hsi_scale_fps.sh
```

若显存不足，可先用 `NUM_FRAMES=20` 检查，但论文图必须为所有方法固定相同的帧数、
分辨率、GPU、max humans、detector 模式和计时边界。

## 输出

结果写到：

```text
outputs/eval/vggt_nlf_hsi_scale_fps/fps_summary.json
outputs/eval/vggt_nlf_hsi_scale_fps/fps_summary.csv
```

JSON 记录完整环境、输入 shape、checkpoint 加载审计、逐次 latency、均值/中位数 FPS、
显存峰值和 HSI 相对开销。论文绘图建议采用 `fps_from_mean_latency`，并在图注中注明
这是不包含图像 I/O/resize 的 model-only steady-state FPS。
