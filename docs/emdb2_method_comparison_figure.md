# EMDB-2 方法对比图准备说明

## 目标与当前状态

目标是制作类似参考图的二维 trade-off 散点图：横轴使用 EMDB-2 的
WA-MPJPE，纵轴使用待定的第二指标。当前先支持 Human3R、UniCon3R、UniSH、
JOSH3R 和 Ours，并突出显示 Ours。

当前只有 WA-MPJPE 等精度指标，没有可靠的运行效率数据。因此脚本不会为缺失
指标生成占位坐标；未传 `--y-metric` 时生成 WA-MPJPE 单指标预览，传入纵轴指标
但 CSV 中仍有空值时会明确报错。

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

若补齐 CSV 的 `fps` 列，生成与参考图同类的“精度—效率”二维图：

```bash
bash scripts/vis/plot_emdb2_method_comparison.sh \
  --y-metric fps \
  --y-scale log \
  --show-pareto
```

若选择现有的 W-MPJPE 或 RTE 作为第二维，可分别使用
`--y-metric w_mpjpe_mm` 或 `--y-metric rte_percent`。但这类图表达的是两个误差指标间
的关系，不是精度—效率 trade-off，通常不如 FPS、延迟、显存或参数量直观。

默认同时导出 300-DPI PNG 和带可编辑文字的 PDF；也可以传
`--formats png,pdf,svg`。

## 推荐第二指标

若目标是复刻参考图的信息结构，优先选择统一硬件、统一输入长度和统一计时协议下的
FPS 或单 clip 延迟。若各方法无法在同一环境重测，参数量或峰值显存更容易保持可比，
但必须在图注中写清楚统计范围。不要混用论文自报 FPS 和本地实测 FPS。
