# EMDB 预测相机与 GT 相机轨迹对比

## 输入与约定

- 预测相机来自完整 Viewer cache。`raw_camera.position` 是原始 VGGT 世界中的相机中心，`hsi_camera.position` 是应用共享 metric scale 后的相机中心。
- GT 来自原生 EMDB `camera["extrinsics"] [T,4,4]`。项目已验证其为 `T_w2c`，脚本求逆得到 `T_c2w`，相机中心为 `T_c2w[:3,3]`。
- 帧关联优先读取新 cache manifest 的 `source_frame_index`；旧缓存回退到对应 `run_summary.json/selected_source_indices`，再回退到 frame ID 数字后缀。
- 默认只保留 `good_frames_mask=True` 的匹配帧。

## 为什么输出两种对齐

- SE(3) 对齐只校正旋转和平移，不修正尺度，适合检查 HSI metric scale 是否接近 GT。
- Sim(3) 对齐同时校正尺度，适合检查轨迹形状并计算常用 translation ATE。
- 当前 HSI 对 raw VGGT 相机平移主要施加共享尺度，因此 Sim(3) 后两条预测可能高度重合；这是预期现象，不代表 HSI 没有作用。尺度贡献应看 SE(3) 图、path-length ratio 和 Sim(3) alignment scale。

## 运行

```bash
cd /home/zhw/lab_users/xyb/home/projects/vggt-human
bash scripts/vis/plot_emdb_p8_68_handstand_camera_trajectory.sh
```

默认预测缓存：

`outputs/vis/stage2_emdb_p8_68_outdoor_handstand_300_1200_500f/full_viewer_cache/`

默认 GT：

`/home/zhw/xyb_space/emdb/P8/68_outdoor_handstand/P8_68_outdoor_handstand_data.pkl`

默认输出：

`outputs/vis/emdb_camera_trajectory/p8_68_outdoor_handstand_300_1200_500f/`

输出包括论文风格 Sim(3) PNG/PDF、SE(3)/Sim(3) 诊断图、逐帧 CSV 和指标 JSON。

如 EMDB 的主要运动平面不是 X-Z，可设置 `PLOT_AXES=xy`、`yz` 或 `auto`。如果图像文件编号与 GT 数组下标存在固定偏移，使用 `FRAME_INDEX_OFFSET`；默认 0。

服务器 smoke test：

```bash
bash scripts/smoke/check_emdb_camera_trajectory_plot.sh
```
