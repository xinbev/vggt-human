# WA-MPJPE100–FPS 对比图

## 图中数据

本图统一使用 EMDB-2 的 WA-MPJPE，并保留当前方法的单个 Ours 点：

| Method | WA-MPJPE100 (mm) ↓ | FPS ↑ | 图中含义 |
|---|---:|---:|---|
| Human3R | 118.2 | 2.41 | EMDB-2 unified feed-forward baseline |
| UniSH | 118.5 | 2.00 | EMDB-2 unified feed-forward baseline |
| JOSH | 68.9 | 0.8 | EMDB-2 result; JOSH optimization variant |
| UniCon3R | 113.7 | 2.40 | EMDB-2 default feed-forward path |
| Ours | 58.468 | 5.285 | `VGGT + NLF + HSI scale` |

Ours 的 WA-MPJPE 按用户指定使用 58.468 mm，原始基础评测结果见
`outputs/eval/emdb2_global_chunk100/metrics/stage_metrics.csv`。FPS 是 100 帧输入上的
CUDA 同步稳态 wall-clock 测量；排除了模型/权重加载、I/O、resize、H2D、warm-up 和
结果写盘。

当前图只保留启用 HSI scale 的 Ours 点，采用用户指定的 WA=58.468 mm 和 FPS=5.285；
这不等同于此前评测文件中 HSI 变体的原始 WA=69.781 mm。

为改善点位可读性，图中的 WA-MPJPE 横轴对 75--110 mm 的空白区做了更强压缩，并将
110 mm 以上的密集点区域横向展开；刻度标签增加为 110、115、120，仍对应原始
WA-MPJPE 数值。

SHOW 论文（`.paper/base_pdf/SHOW.pdf`）没有报告模型推理 FPS。文中提到的 6 FPS
仅表示训练视频下采样和消融数据输入频率，不应作为 SHOW 的 inference FPS。SHOW 的
EMDB-2 表格报告其 Ours 为 WA=109.1 mm；当前图按照用户指定的本项目结果 58.5 mm，
不采用 SHOW 表格中的 109.1 mm。

## 重要口径限制

当前横坐标均改为 **EMDB-2**。baseline FPS 是既有论文报告值，Ours FPS 是本项目实测值；
不同方法的硬件和计时协议仍可能不同，因此本图不应直接表述为严格硬件归一化的速度排名。

正式论文版本应采用以下二者之一：

1. 在统一 GPU、输入分辨率、帧数和计时边界下复测所有 baseline；或
2. 在论文图注中明确 baseline FPS 为 reported values、Ours FPS 为 local measurement。

## 文件与运行

- 数据：`configs/vis/wa_mpjpe100_fps_comparison.csv`
- 绘图脚本：`scripts/vis/plot_emdb2_method_comparison.py`
- 服务器入口：`scripts/vis/plot_emdb2_method_comparison.sh`
- 输出：`outputs/vis/wa_mpjpe100_fps_comparison/`

服务器执行：

```bash
bash scripts/vis/plot_emdb2_method_comparison.sh \
  --y-metric fps \
  --y-scale log \
  --output-name wa_mpjpe100_vs_fps \
  --formats png,pdf,svg
```
