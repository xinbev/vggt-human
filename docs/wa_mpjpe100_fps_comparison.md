# WA-MPJPE100–FPS 对比图

## 图中数据

本图按用户提供的 UniCon3R 参考图重建 baseline，并绘制当前方法的两个版本：

| Method | WA-MPJPE100 (mm) ↓ | FPS ↑ | 图中含义 |
|---|---:|---:|---|
| Human3R | 97.5 | 2.41 | 参考图中的 unified feed-forward baseline；图中省略 † 上标 |
| UniSH | 118.1 | 2.00 | 参考图中的 unified feed-forward baseline |
| JOSH | 102.6 | 0.16 | 参考图中的 test-time optimization baseline |
| UniCon3R | 81.5 | 2.40 | 默认 feed-forward 路径 |
| Ours | 58.468 | 9.757 | `VGGT + NLF` |
| Ours + HSI | 69.781 | 5.285 | `VGGT + NLF + HSI scale` |

Ours 的 WA-MPJPE 数值来自
`outputs/eval/emdb2_global_chunk100/metrics/stage_metrics.csv`。FPS 是 100 帧输入上的
CUDA 同步稳态 wall-clock 测量；排除了模型/权重加载、I/O、resize、H2D、warm-up 和
结果写盘。

## 重要口径限制

参考图的 Human3R、UniSH、JOSH 和 UniCon3R 横坐标来自 **RICH**；当前
Ours 的 58.468/69.781 来自 **EMDB-2**。此外 baseline FPS 是论文报告值，Ours FPS
是本项目实测值。因此当前图只适合作为版式草图或非严格的 reported-number
visualization，不能直接用于声称同一 benchmark、同一硬件协议下的定量优势。

正式论文版本应采用以下二者之一：

1. 在 RICH 上补测两个 Ours 版本的 WA-MPJPE100；或
2. 将 baseline 横坐标全部替换为 EMDB-2 数值，并尽量在统一硬件上复测 FPS。

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
