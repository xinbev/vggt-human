# RICH WA-MPJPE100–FPS 预览图

## 使用数据

| Method | WA-MPJPE100 (mm) ↓ | FPS ↑ | Source |
|---|---:|---:|---|
| Human3R† | 97.5 | 2.41 | UniCon3R, Sec. 5.5 / Fig. 1(b) |
| UniSH | 118.1 | 2.00 | UniCon3R, Sec. 5.5 / Fig. 1(b) |
| JOSH | 102.6 | 0.16 | UniCon3R, Sec. 5.5 / Fig. 1(b) |
| UniCon3R | 81.5 | 2.40 | UniCon3R, Sec. 5.5 / Fig. 1(b) |
| UniCon3R* | 72.2 | 0.16 | UniCon3R, Sec. 5.5 / Fig. 1(b) |

UniCon3R* 是采用 JOSH-style camera refinement 的可选测试时优化版本。UniCon3R 与
UniCon3R* 均为 RICH 数据，因此图中使用虚线连接以呈现精度–速度权衡。

项目当前没有 Ours 的 RICH WA-MPJPE 评测结果；论文草稿的 RICH Ours 单元格也是
`--`。本预览图因此不绘制 Ours，避免把 EMDB-2 的 58.468 mm 混入 RICH 对比。

## 文件与服务器运行

- 数据：`configs/vis/rich_wa_mpjpe100_fps_comparison.csv`
- 绘图脚本：`scripts/vis/plot_emdb2_method_comparison.py`
- 服务器入口：`scripts/vis/plot_rich_method_comparison.sh`
- 输出：`outputs/vis/rich_wa_mpjpe100_fps_comparison/`

```bash
bash scripts/vis/plot_rich_method_comparison.sh
```
