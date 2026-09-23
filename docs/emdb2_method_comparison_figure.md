# EMDB-2 方法对比图准备说明

## 目标与当前状态

目标是制作类似参考图的二维 trade-off 散点图：横轴使用 EMDB-2 的
WA-MPJPE，纵轴使用 FPS。当前包含 Human3R、UniCon3R、UniSH、JOSH3R 和
Ours，并突出显示 Ours。Ours 对应完整的 `VGGT + NLF + HSI scale` 推理路径。

## 数据来源与口径

初始 CSV 数值来自本任务中用户提供的 EMDB-2 (24) 表格截图，而不是从
`.paper/` 代码直接移植：

| Method | WA-MPJPE (mm) | W-MPJPE (mm) | RTE (%) |
|---|---:|---:|---:|
| Human3R | 112.2 | 267.9 | 2.2 |
| UniCon3R | 113.7 | 285.6 | 2.3 |
| UniSH | 118.5 | 270.1 | 5.8 |
| JOSH3R | 220.0 | 661.7 | 13.1 |
| Ours | 58.468 | 149.959 | 1.079 |

这些指标均为越小越好。最终论文使用前仍需核对：数据集 split、序列/窗口定义、
关节集合、对齐方式和聚合方式是否完全一致。不同协议的数值不应放在同一张
定量图中直接比较。

FPS 使用值与来源如下：

| Method | FPS ↑ | 来源与口径 |
|---|---:|---|
| Human3R | 2.41 | UniCon3R 论文统一效率对比，ViT-L/896，单张 NVIDIA A5000 |
| UniCon3R | 2.40 | UniCon3R 论文统一效率对比，ViT-L/896，单张 NVIDIA A5000 |
| UniSH | 2.00 | UniCon3R 论文统一效率对比，单张 NVIDIA A5000 |
| JOSH3R | 15.4 | JOSH 论文 Table 3，RTX 4090，所有模块的 amortized FPS |
| Ours | 5.285 | 本项目实测，`VGGT + NLF + HSI scale`，100 帧输入 |

Ours 的实测输入张量为 `[1, 100, 3, 448, 592]`，采用 CUDA 同步后的稳态
wall-clock 时间。计时包含 VGGT、NLF detector、SMPL coarse scale 和 HSI residual
scale/bias；排除模型/配置/权重加载、图像 I/O、resize、H2D、NLF lazy load、
warm-up 和结果写盘。对照测试中 `VGGT + NLF` 为 9.757 FPS，加入 HSI scale 后为
5.285 FPS。

必须注意：baseline FPS 是既有论文在各自硬件/协议下的报告值，而 Ours 是本项目
实测；JOSH3R 使用 RTX 4090，其他三个 baseline 使用 A5000。因此本图用于展示
报告速度下的趋势，不能表述为严格硬件归一化的公平速度排名。论文图注应保留：
“Baseline FPS values are reported by prior works under their respective hardware and
protocols, whereas Ours is measured locally; FPS is therefore indicative rather than
strictly hardware-normalized.”

不要把参考图中 JOSH 的 0.16 FPS 填给 JOSH3R；当前 EMDB-2 数值对应的是 JOSH3R，
其匹配的论文报告速度是 15.4 FPS。

## 文件

- 数据：`configs/vis/emdb2_method_comparison.csv`
- 绘图：`scripts/vis/plot_emdb2_method_comparison.py`
- 服务器入口：`scripts/vis/plot_emdb2_method_comparison.sh`
- 输出：`outputs/vis/emdb2_method_comparison/`

CSV 已预留 `fps`、`latency_s`、`gpu_memory_gb` 和 `params_m`。`label_dx`、
`label_dy` 是文字相对点位的偏移（单位为 points），可用于解决最终图中的标签重叠。

## 运行方式

同步到服务器后，项目路径为：

```text
/home/zhw/lab_users/xyb/home/projects/vggt-human
```

生成当前 WA-MPJPE 单指标预览：

```bash
bash scripts/vis/plot_emdb2_method_comparison.sh
```

生成“精度—效率”二维图：

```bash
bash scripts/vis/plot_emdb2_method_comparison.sh \
  --y-metric fps \
  --y-scale log \
  --show-pareto \
  --output-name wa_mpjpe_vs_fps \
  --formats png,pdf,svg
```

若选择现有的 W-MPJPE 或 RTE 作为第二维，可分别使用
`--y-metric w_mpjpe_mm` 或 `--y-metric rte_percent`。但这类图表达的是两个误差指标间
的关系，不是精度—效率 trade-off，通常不如 FPS、延迟、显存或参数量直观。

默认同时导出 300-DPI PNG 和带可编辑文字的 PDF；也可以传
`--formats png,pdf,svg`。

## 解释边界

当前图已经可以用于内部对比或带明确限制说明的论文草稿。若要将“速度优势”作为
强结论，仍应在同一 GPU、相同输入分辨率、相同帧数、相同 batch size 和相同计时
边界下复测所有开源 baseline；否则只宜称为 reported-speed trade-off。
