# EMDB 与 3DPW Pickle 数据格式

本文档记录以下两个真实数据文件的读取结果，供后续 Agent 实现统一数据加载类时使用。

- EMDB：`/home/zhw/xyb_space/emdb/P0/01_mvs_b/P0_01_mvs_b_data.pkl`
- 3DPW：`/home/zhw/xyb_space/3DPW/sequenceFiles/train/courtyard_arguing_00.pkl`

两个文件均可使用 Python `pickle.load` 读取。3DPW 文件建议显式传入 `encoding="latin1"`，以兼容其历史 Python pickle 格式。

## 1. EMDB 文件

### 1.1 顶层结构

顶层对象是 `dict`，包含以下字段：

```text
{
    "gender": str,
    "name": str,
    "emdb1": bool,
    "emdb2": bool,
    "n_frames": int,
    "good_frames_mask": np.ndarray,
    "camera": dict,
    "smpl": dict,
    "kp2d": np.ndarray,
    "bboxes": dict,
}
```

该文件的基本信息：

```text
gender: male
name: P0_01_mvs_b
emdb1: False
emdb2: False
n_frames: 554
camera.width: 1440
camera.height: 1920
```

### 1.2 `camera`

```text
camera["intrinsics"]
    type: np.ndarray
    shape: (3, 3)
    dtype: float64
    含义: 相机内参矩阵 K

camera["extrinsics"]
    type: np.ndarray
    shape: (554, 4, 4)
    dtype: float64
    含义: 每帧 4x4 外参矩阵，具体 world-to-camera 或 camera-to-world
          应以该数据集的原始约定为准，不能仅凭字段名假设。

camera["width"]
    type: int
    value: 1440

camera["height"]
    type: int
    value: 1920
```

实际内参：

```python
K = np.array([
    [1436.4219970703125, 0.0, 723.2894287109375],
    [0.0, 1436.4219970703125, 963.9358520507812],
    [0.0, 0.0, 1.0],
], dtype=np.float64)
```

### 1.3 `smpl`

```text
smpl["poses_root"]
    shape: (554, 3), dtype: float64
    每帧 SMPL 根节点 axis-angle 旋转

smpl["poses_body"]
    shape: (554, 69), dtype: float64
    每帧 SMPL body pose，23 个 body joints x 3

smpl["trans"]
    shape: (554, 3), dtype: float32
    每帧平移

smpl["betas"]
    shape: (10,), dtype: float32
    序列级 SMPL 身体形状参数
```

其中，`poses_root` 与 `poses_body` 可以拼接为每帧 72 维 SMPL axis-angle pose：

```python
pose72 = np.concatenate(
    [data["smpl"]["poses_root"], data["smpl"]["poses_body"]],
    axis=-1,
)
# pose72.shape == (554, 72)
```

### 1.4 关键点与包围框

```text
kp2d
    type: np.ndarray
    shape: (554, 24, 2)
    dtype: float64
    含义: 每帧 24 个二维关键点，最后一维为 (x, y)，单位为像素

bboxes["bboxes"]
    type: np.ndarray
    shape: (554, 4)
    dtype: float64
    含义: 每帧一个 bbox，格式为 [x1, y1, x2, y2]，单位为像素

bboxes["invalid_idxs"]
    type: np.ndarray
    shape: (0,)
    dtype: int64
    含义: 无效帧索引；该样本为空数组

good_frames_mask
    type: np.ndarray
    shape: (554,)
    dtype: bool
    该样本中 554 个值全部为 True
```

### 1.5 EMDB 首帧真实数据

首帧索引为 `frame_idx = 0`：

```python
frame0 = {
    "good_frame": True,
    "smpl_root_pose": [
        0.0301839253, -2.6950244169, 0.1059646635,
    ],
    "smpl_trans": [
        1.1270416975, -0.1287951767, -0.9106571078,
    ],
    "smpl_betas": [
        0.6417693496, 1.3236318827, -0.0795477331,
        0.1708967984, -0.2126822769, -0.7399393916,
        0.2144684494, -0.1137121618, 0.0000883793,
        0.3325954676,
    ],
    "bbox_xyxy": [
        492.1082031250, 581.0966523438,
        1440.0, 1599.0604827881,
    ],
    "kp2d_first_4": [
        [954.0434875488, 1048.3990402222],
        [978.2705078125, 1083.5051345825],
        [927.6865997314, 1085.0170288086],
        [958.2739410400, 990.9023418427],
    ],
}
```

首帧外参矩阵为：

```python
T0 = np.array([
    [-0.8044775487,  0.0119724517,  0.5938623864,  1.9185989272],
    [-0.0701724767, -0.9947109990, -0.0750056810, -0.1648031171],
    [ 0.5898234457, -0.1020131808,  0.8010628027,  3.0030819287],
    [ 0.0,           0.0,           0.0,           1.0],
], dtype=np.float64)
```

## 2. 3DPW 文件

### 2.1 顶层结构

顶层对象也是 `dict`，包含以下字段：

```text
{
    "trans_60Hz": list[np.ndarray],
    "cam_intrinsics": np.ndarray,
    "poses": list[np.ndarray],
    "img_frame_ids": np.ndarray,
    "betas_clothed": list[np.ndarray],
    "sequence": np.ndarray or scalar-like np.ndarray,
    "v_template_clothed": list[np.ndarray],
    "jointPositions": list[np.ndarray],
    "poses_60Hz": list[np.ndarray],
    "betas": list[np.ndarray],
    "cam_poses": np.ndarray,
    "campose_valid": list[np.ndarray],
    "genders": list[np.ndarray or str],
    "trans": list[np.ndarray],
    "poses2d": list[np.ndarray],
    "texture_maps": list,
}
```

该序列的基本信息：

```text
sequence: courtyard_arguing_00
人数: 2
图像帧数: 765
原始运动帧率: 60 Hz 相关字段存在
保存的图像帧 ID: [0, 2, 4, ..., 1528]
```

### 2.2 相机字段

```text
cam_intrinsics
    shape: (3, 3), dtype: float64
    含义: 序列共享的相机内参矩阵 K

cam_poses
    shape: (765, 4, 4), dtype: float64
    含义: 每个保存图像帧对应的相机位姿矩阵

img_frame_ids
    shape: (765,), dtype: uint16
    示例: [0, 2, 4, 6, 8, 10, ...]

campose_valid
    list 长度: 2
    每项 shape: (765,), dtype: uint8
    含义: 每个人在每一帧的相机姿态有效标记
```

实际内参：

```python
K = np.array([
    [1961.8528610354, 0.0, 540.0],
    [0.0, 1969.2307692308, 960.0],
    [0.0, 0.0, 1.0],
], dtype=np.float64)
```

注意：`cam_intrinsics` 的主点约为 `(540, 960)`，说明原始图像很可能是宽 1080、高 1920 的竖幅图像。读取图片后仍应以实际图片尺寸校验，不要硬编码尺寸。

### 2.3 每个人的 SMPL 字段

以下字段都是长度为 2 的 list，list 下标是 `person_idx`：

```text
poses[person_idx]
    shape: (765, 72), dtype: float64
    每帧 SMPL 24 joints 的 axis-angle pose

trans[person_idx]
    shape: (765, 3), dtype: float64
    每帧平移

betas[person_idx]
    shape: (10,), dtype: float64
    序列级身体形状参数

poses_60Hz[person_idx]
    shape: (1530, 72), dtype: float64
    60 Hz 采样的 pose

trans_60Hz[person_idx]
    shape: (1530, 3), dtype: float64
    60 Hz 采样的平移

betas_clothed[person_idx]
    shape: (10,), dtype: float64

v_template_clothed[person_idx]
    shape: (6890, 3), dtype: float64
    clothed SMPL 模板顶点

genders[person_idx]
    标量字符串，样本中 person 0 为 "f"，person 1 为 "m"
```

`poses[person_idx][frame_idx]` 的 72 维顺序通常是 SMPL 的 24 个关节，每个关节 3 个 axis-angle 参数。对于本文件，不能把它当成 `(24, 3)` 以外的关键点坐标。

### 2.4 3DPW 关键点字段

```text
poses2d[person_idx]
    shape: (765, 3, 18), dtype: float64
    原始布局为 (frame, channel, joint)
    channel 0: x 像素坐标
    channel 1: y 像素坐标
    channel 2: 置信度
```

如需常用的 `(frame, joint, channel)` 布局，应转换为：

```python
keypoints_2d = np.asarray(data["poses2d"][person_idx]).transpose(0, 2, 1)
# keypoints_2d.shape == (765, 18, 3)
# keypoints_2d[frame, joint] == [x, y, confidence]
```

`jointPositions[person_idx]`：

```text
shape: (765, 72), dtype: float64
```

它不是 72 个二维参数，而是每帧 24 个 3D 关节坐标展平后的结果。使用时应 reshape：

```python
joints_3d = np.asarray(data["jointPositions"][person_idx]).reshape(-1, 24, 3)
# joints_3d.shape == (765, 24, 3)
```

### 2.5 3DPW 其他字段

```text
sequence
    标量字符串数组，值为 "courtyard_arguing_00"

texture_maps
    空 list，本样本没有纹理图数据
```

### 2.6 3DPW 首帧首人真实数据

使用：

```python
frame_idx = 0
person_idx = 0
```

首帧图像 ID、人物和相机信息：

```python
frame0_person0 = {
    "sequence": "courtyard_arguing_00",
    "img_frame_id": 0,
    "person_idx": 0,
    "gender": "f",
    "campose_valid": 1,
    "trans": [
        0.4148810071, -0.8961028617, 1.4217196010,
    ],
    "betas": [
        -0.7517868581, 1.4048918136, -0.2639936725,
        -0.1345450694, 0.0916591714, -0.2704026368,
        0.0647704660, -0.0095674864, 0.0229634709,
        0.0074978687,
    ],
}
```

首帧首人 72 维 pose：

```python
pose72 = [
    -0.0120759295, -0.1709354568,  0.0503373624,
     0.0289408695, -0.2607221664, -0.0645466533,
     0.0534365651,  0.0881130281,  0.0344089484,
    -0.0758388161,  0.0113605071, -0.0142092194,
    # 后续共 72 个值，完整值直接从 data["poses"][0][0] 读取
]
```

首帧首人 18 个二维关键点，已经转换为 `(joint, x/y/confidence)`：

```python
keypoints_2d = [
    [244.656, 615.699, 0.848425],
    [280.926, 730.631, 0.910117],
    [218.459, 730.426, 0.950004],
    [202.890, 855.935, 0.834249],
    [187.559, 970.430, 0.816147],
    [353.804, 735.716, 0.923294],
    [364.360, 881.759, 0.819395],
    [353.785, 986.086, 0.960976],
    [228.928, 991.292, 0.821661],
    [265.123, 1173.800, 0.875412],
    [265.608, 1335.560, 0.899632],
    [312.370, 996.615, 0.814667],
    [322.553, 1173.870, 0.806455],
    [317.545, 1351.260, 0.910580],
    [229.225, 605.124, 0.1915666],
    [265.547, 605.047, 0.197949],
    [0.0, 0.0, 0.0],
    [312.134, 610.630, 0.1791256],
]
```

首帧相机位姿矩阵：

```python
cam_pose0 = np.array([
    [ 0.9583380685,  0.0366945973, -0.2832695764, -0.4610863065],
    [ 0.0877543815, -0.9815754126,  0.1697317820, -1.3847345526],
    [-0.2718222120, -0.1875185746, -0.9439011968,  5.0435929779],
    [ 0.0,           0.0,           0.0,           1.0],
], dtype=np.float64)
```

## 3. 建议的统一样本接口

两个数据集的原始组织方式不同：EMDB 是单人序列级字典，3DPW 是多人序列级字典。因此，建议加载类内部统一为“按帧、按人”的结构，而不是直接暴露原始 pickle：

```python
sample = {
    "dataset": "emdb" or "3dpw",
    "sequence": str,
    "frame_index": int,
    "image_frame_id": int or None,
    "image_hw": (height, width),
    "K": np.ndarray,                 # (3, 3), float32
    "camera_pose": np.ndarray,       # (4, 4), float32
    "persons": [
        {
            "person_id": int,
            "gender": str,
            "pose": np.ndarray,      # (72,), float32
            "trans": np.ndarray,     # (3,), float32
            "betas": np.ndarray,     # (10,), float32
            "keypoints_2d": np.ndarray or None,
            "joints_3d": np.ndarray or None,
            "bbox_xyxy": np.ndarray or None,
            "valid": bool,
        },
    ],
}
```

建议统一规则：

1. 所有数值数组在进入模型前转换为 `np.float32` 或对应的 PyTorch `torch.float32`。
2. 所有 bbox 使用像素坐标 `[x1, y1, x2, y2]`，不要混用 `[x, y, w, h]`。
3. EMDB 的 `kp2d` 是 `(24, 2)`，没有置信度通道；3DPW 的 `poses2d` 转置后是 `(18, 3)`，包含置信度。
4. 3DPW 的 `poses2d` 必须从 `(T, 3, 18)` 转换为 `(T, 18, 3)`。
5. 3DPW 的 `poses` 已经是 `(T, 72)`；EMDB 需要拼接 `poses_root` `(T, 3)` 和 `poses_body` `(T, 69)`。
6. 3DPW 一帧可能有多个人，必须保留 `person_idx`，不能只取第一个人。
7. EMDB 本文件只有一个人，`gender` 和 `betas` 是序列级字段。
8. 3DPW 的 `img_frame_ids` 与数组索引不同：数组索引 `0` 对应原始图像帧 ID `0`，数组索引 `1` 对应图像帧 ID `2`。
9. `cam_poses`、EMDB 的 `camera["extrinsics"]` 的变换方向需要结合项目使用方式确认；实现时应保留原始矩阵，不要未经验证直接求逆。
10. 图像文件不在这两个 pickle 中，需要根据序列名、帧号和对应数据集根目录另行拼接。

## 4. 最小读取代码

```python
import pickle
import numpy as np

with open("/home/zhw/xyb_space/emdb/P0/01_mvs_b/P0_01_mvs_b_data.pkl", "rb") as f:
    emdb = pickle.load(f)

with open("/home/zhw/xyb_space/3DPW/sequenceFiles/train/courtyard_arguing_00.pkl", "rb") as f:
    dpw = pickle.load(f, encoding="latin1")

# EMDB 第 0 帧
emdb_pose0 = np.concatenate([
    emdb["smpl"]["poses_root"][0],
    emdb["smpl"]["poses_body"][0],
]).astype(np.float32)
emdb_kp2d_0 = np.asarray(emdb["kp2d"][0], dtype=np.float32)
emdb_bbox_0 = np.asarray(emdb["bboxes"]["bboxes"][0], dtype=np.float32)

# 3DPW 第 0 帧、第 0 人
dpw_pose0 = np.asarray(dpw["poses"][0][0], dtype=np.float32)
dpw_kp2d_0 = np.asarray(dpw["poses2d"][0][0], dtype=np.float32).T
dpw_joints3d_0 = np.asarray(dpw["jointPositions"][0][0], dtype=np.float32).reshape(24, 3)
dpw_frame_id_0 = int(dpw["img_frame_ids"][0])
```

## 5. 已完成的检查

- 两个 pickle 文件均存在且可读取。
- 已确认两个文件的顶层类型均为 `dict`。
- 已检查所有主要字段的类型、shape、dtype 和部分数值范围。
- 已抽取 EMDB 第 0 帧真实数据。
- 已抽取 3DPW `courtyard_arguing_00` 第 0 帧第 0 人真实数据。
- 未修改原始数据文件。
