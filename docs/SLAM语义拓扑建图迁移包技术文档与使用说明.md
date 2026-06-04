# SLAM 语义拓扑建图迁移包技术文档与使用说明

本文档面向 `/home/zyf/Public/slam` 迁移包，说明这套工程从数据集出发，如何完成 SVO2 前视数据解析、cuVSLAM 位姿与点云生成、多传感器时间同步、三维物体节点构建、语义拓扑建图，以及 Rerun 在线式回放导出。文档同时包含技术分析、数据集要求、安装方式、运行命令、输出说明和常见问题。

## 1. 工程目标

这套工程的目标是把一次多传感器采集结果转换成可分析、可回放、可用于导航和语义理解的多层语义拓扑地图。

整体流程遵循第一性原理：先从原始传感器数据中恢复每一帧的时间、图像、深度、位姿和点云，再把二维物体检测结果结合真实深度投影到三维空间，随后通过稳定性过滤、重复过滤、运动物体过滤和消失回访过滤得到可靠物体节点，最后构建途径点、拓扑节点、物体节点和高层语义区域，并统一导出可视化结果。

## 2. 迁移包目录结构

```text
/home/zyf/Public/slam/
  configs/                 # 运行配置
  docs/                    # 技术文档
  models/                  # YOLO、SAM、OPR、深度模型权重
  runtime/                 # 可迁移运行时，包含 ZED SDK、cuVSLAM 绑定和本地 wheel
  scripts/                 # 安装、检查、冒烟测试和一键运行脚本
  semantic_topomap/        # 主流程代码
  third_party/             # OPR、Object3D、导航图工具等第三方模块
  outputs/                 # 默认输出目录
  requirements.txt         # Python 依赖
  pyproject.toml           # Python 包入口配置
```

核心入口是 `semantic_topomap.cli`。常用脚本包括：

- `scripts/setup_env.sh`：安装 Python 依赖、本地 ZED Python wheel，并以可编辑方式安装本工程。
- `scripts/env.sh`：设置 `PYTHONPATH`、`LD_LIBRARY_PATH`、ZED SDK 路径和 cuVSLAM 运行库路径。
- `scripts/run_pipeline.sh`：一键运行完整流程。
- `scripts/smoke_test_dataset.sh`：运行 3 帧冒烟测试。
- `scripts/run_dataset_test3.sh`：针对当前测试集的快捷脚本。

## 3. 数据集要求

最小必需数据结构如下：

```text
dataset/
  zed/
    zed.svo2
    zed_svo_index.csv
  rear_camera/
    rgb/
  lidar/
    scans/
```

其中 `zed.svo2` 是前视主数据源。它负责提供前视彩色图像、真实深度、相机内参，并作为 cuVSLAM 运行输入生成过程位姿和稀疏点云。

`zed_svo_index.csv` 是时间同步核心，至少需要包含：

```text
frame_id
jetson_timestamp_ns
zed_image_timestamp_ns
recorded_frame_index
```

字段含义：

- `frame_id`：采集系统中的帧编号。
- `jetson_timestamp_ns`：外部采集主机时间戳，用于匹配后视图像和 LiDAR。
- `zed_image_timestamp_ns`：ZED 图像时间戳，用于匹配 SVO2 导出的前视图像、深度和 cuVSLAM 轨迹。
- `recorded_frame_index`：SVO2 中的录制帧索引。

后视图像和 LiDAR 属于外部辅助输入，需要单独提供。系统默认要求它们存在，因为当前 OPR 位置描述符使用前视、后视和 LiDAR 多模态信息。

可选数据包括：

- `rear_camera/frames.csv`：后视图像显式索引。
- `rear_camera/calibration.json`：后视相机标定。
- `front_camera/rgb`：已有前视图像缓存。
- `depth`：已有深度缓存。
- `sync/alignment.csv`：额外同步文件。

当前工程优先把 SVO2 作为前视数据主源，即使数据集中已经存在前视图像和深度目录，也会优先从 SVO2 重新导出，保证 RGB、Depth、内参和 cuVSLAM 时间链路一致。

## 4. 完整技术流程

### 4.1 数据检查

系统首先检查数据集是否包含 SVO2、SVO2 时间索引、后视图像目录和 LiDAR 扫描目录。如果缺失必需项，默认会停止运行。

检查命令：

```bash
cd /home/zyf/Public/slam
source scripts/env.sh
python -m semantic_topomap.cli check \
  --dataset /home/zyf/Desktop/dataset_test3 \
  --output outputs/dataset_test3/reports/dataset_check.json
```

### 4.2 SVO2 前视数据导出

系统使用 ZED SDK 打开 `zed.svo2`，逐帧导出：

- 前视 RGB 图像
- 前视真实深度 PNG
- 相机内参

这些内容会进入 `outputs/<run>/extracted/`。

### 4.3 cuVSLAM 位姿与点云生成

系统再次读取同一个 SVO2，调用 cuVSLAM 生成：

- 相机过程轨迹
- 稀疏点云
- 按时间抽样的点云快照
- cuVSLAM 点云 HTML 回放

输出目录为 `outputs/<run>/cuvslam/`。

注意：cuVSLAM 从 SVO2 中恢复的是前视视觉惯性里程计相关信息，后视图像和 LiDAR 不从 SVO2 中获得，需要数据集额外提供。

### 4.4 时间同步与数据适配

同步阶段使用 SVO2 时间索引作为桥梁：

- 使用 `zed_image_timestamp_ns` 对齐前视 RGB、真实深度和 cuVSLAM 轨迹。
- 使用 `jetson_timestamp_ns` 对齐后视图像和 LiDAR。
- 按 stride 抽取建图帧，默认 stride 为 15。
- 生成系统可读的建图帧序列。

输出目录为 `outputs/<run>/prepared/`。

### 4.5 坐标系统一

ZED/cuVSLAM 坐标与系统世界坐标不完全一致。工程中使用统一修正：

```text
ZED/cuVSLAM 坐标 [x, y, z] -> 系统世界坐标 [x, z, -y]
```

这个转换非常关键。之前物体左右位置不合理、物体集中到轨迹中线附近，本质上就是坐标轴映射没有统一导致。当前配置中 `coordinate_axis_map: zed_to_code` 表示启用这套坐标修正。

### 4.6 二维目标检测

系统使用微调后的 YOLO 权重检测前视 RGB 图像中的物体，输出：

- 类别
- 置信度
- 二维检测框

权重路径默认为：

```text
models/yolo/best.pt
```

### 4.7 YOLO + SAM + 真实深度构建 Object3D

每个 YOLO 检测框会交给 SAM 生成更精确的物体掩膜。随后系统在掩膜区域读取真实深度，把像素点回投为相机坐标系下的三维点云，再结合当前相机位姿转换到世界坐标系。

Object3D 构建包含：

- 前景点云过滤，减少背景深度污染。
- DBSCAN 密度聚类去噪，去除离群点。
- 三米深度门限，只保留近距离可信物体观测。
- 单帧三维物体观测生成。
- 跨帧跟踪与合并。
- 三维包围框生成。

这部分是当前语义物体节点质量的核心。

### 4.8 物体过滤后处理

Object3D 结果不是直接进入导航图，而是先经过多级过滤：

- 连续帧稳定过滤：物体至少需要连续出现达到配置阈值。
- 同类三维框重叠去重：同类别、空间高度重叠的框只保留更稳定的对象。
- 运动物体过滤：只针对配置中的可运动类别，默认包括 `person`。
- 消失后回访过滤：只有当物体位置处于三米内、视场内、回访位置附近，且连续可见缺失达到阈值时，才判定为消失。
- 导航图二次裁剪：进入图结构前会继续去掉不可靠对象。

当前策略的原则是：不让远距离不可确认信息误删近距离稳定物体，也不让临时误检或动态目标长期保留在图中。

### 4.9 拓扑图构建

拓扑构建阶段使用 OPR 位置识别模型提取位置描述符。系统结合视觉相似度和空间距离判断当前位置属于已有拓扑节点还是新拓扑节点。

同时，系统不会把每一帧都作为途径点，而是通过采样减少冗余：

- 行进距离超过阈值保留。
- 朝向变化超过阈值保留。
- 首帧和尾帧保留。
- 拓扑节点变化时保留。

最终构建：

- 途径点节点
- 拓扑节点
- 物体节点
- 途径点时间边
- 途径点到物体边
- 拓扑层级边

### 4.10 高层语义区域构建

系统会基于拓扑节点及其相连物体类别，对拓扑节点进一步聚合，形成多层语义区域：

- 一级：原始拓扑节点
- 二级：局部语义区域
- 三级：更高层区域汇总

这样既能表达精细路径，也能表达区域级空间语义。

### 4.11 Rerun 在线式回放

Rerun 导出用于把建图过程放到统一时间轴中查看。当前回放包含：

- RGB 图像随时间播放。
- YOLO 检测框随时间出现。
- cuVSLAM 点云随时间增长。
- 相机轨迹随时间增长。
- 途径点随时间出现。
- 三维物体框和物体节点随时间出现或消失。
- 拓扑节点、二级语义区域和三级语义区域在更高高度显示。
- 黑色背景、小半径点云、高对比度标签。

输出文件为：

```text
outputs/<run>/rerun/semantic_topomap_replay.rrd
```

## 5. 安装说明

推荐在 Python 3.10 环境中安装。

```bash
cd /home/zyf/Public/slam
bash scripts/setup_env.sh
```

安装脚本会执行：

```bash
python -m pip install -U pip
python -m pip install -r requirements.txt
python -m pip install runtime/wheels/*.whl
python -m pip install -e .
```

如果不使用安装脚本，至少需要在运行前执行：

```bash
cd /home/zyf/Public/slam
source scripts/env.sh
```

否则 Python 可能找不到本地 cuVSLAM 绑定、ZED SDK 动态库或第三方模块。

## 6. 环境检查

运行：

```bash
cd /home/zyf/Public/slam
source scripts/env.sh
python -m semantic_topomap.cli doctor \
  --dataset /home/zyf/Desktop/dataset_test3 \
  --output outputs/dataset_test3/reports/doctor.json
```

检查重点：

- Python 依赖是否可导入。
- `pyzed.sl` 是否可用。
- cuVSLAM Python 绑定是否可用。
- Rerun 是否可用。
- 模型权重是否存在。
- 数据集结构是否满足要求。

## 7. 冒烟测试

冒烟测试只跑 3 帧，适合迁移到新设备后快速确认工程能闭环。

```bash
cd /home/zyf/Public/slam
bash scripts/smoke_test_dataset.sh \
  /home/zyf/Desktop/dataset_test3 \
  /home/zyf/Public/slam/outputs/dataset_test3_smoke
```

冒烟测试会执行：

- doctor 环境检查
- SVO2 前视导出
- cuVSLAM 运行
- 数据同步
- Object3D 与拓扑建图
- Rerun 回放导出

## 8. 全量运行

### 8.1 默认 CPU 配置

```bash
cd /home/zyf/Public/slam
bash scripts/run_pipeline.sh \
  /home/zyf/Desktop/dataset_test3 \
  /home/zyf/Public/slam/outputs/dataset_test3_default
```

默认配置是：

```text
configs/default.yaml
```

其中 `runtime.device: cpu`，适合没有 GPU 或需要最大兼容性的设备，但速度较慢。

### 8.2 推荐混合 GPU 配置

当前环境推荐使用：

```text
configs/gpu_yolo_sam.yaml
```

运行：

```bash
cd /home/zyf/Public/slam
bash scripts/run_pipeline.sh \
  /home/zyf/Desktop/dataset_test3 \
  /home/zyf/Public/slam/outputs/dataset_test3_gpu_yolo_sam \
  configs/gpu_yolo_sam.yaml
```

该配置中：

```yaml
runtime:
  device: cpu
  yolo_device: cuda:0
  sam_device: cuda:0
```

原因是当前环境中的 MinkowskiEngine 是 CPU_ONLY 编译，如果把 OPR 整体放到 CUDA，会触发报错。因此推荐让 OPR 走 CPU，让 YOLO 和 SAM 走 GPU。这样既能避开 MinkowskiEngine 限制，又能加速最重的检测和分割环节。

### 8.3 纯 GPU 配置

`configs/gpu.yaml` 中将 `device`、`yolo_device`、`sam_device` 全部设置为 CUDA。只有当目标设备上的 MinkowskiEngine 支持 CUDA 时才建议使用。

如果当前环境直接使用纯 GPU 配置，可能出现：

```text
AssertionError: The MinkowskiEngine was compiled with CPU_ONLY flag.
```

解决方式有两个：

- 使用 `configs/gpu_yolo_sam.yaml`。
- 重新安装支持 CUDA 的 MinkowskiEngine。

## 9. 分步运行

如果需要定位问题，可以分步执行。

导出 SVO2：

```bash
python -m semantic_topomap.cli export-svo2 \
  --dataset /home/zyf/Desktop/dataset_test3 \
  --output outputs/debug_run
```

运行 cuVSLAM：

```bash
python -m semantic_topomap.cli run-cuvslam \
  --dataset /home/zyf/Desktop/dataset_test3 \
  --output outputs/debug_run
```

同步准备：

```bash
python -m semantic_topomap.cli prepare \
  --dataset /home/zyf/Desktop/dataset_test3 \
  --output outputs/debug_run \
  --stride 15
```

构建语义拓扑图：

```bash
python -m semantic_topomap.cli --config configs/gpu_yolo_sam.yaml build-map \
  --output outputs/debug_run
```

导出 Rerun：

```bash
python -m semantic_topomap.cli --config configs/gpu_yolo_sam.yaml export-rerun \
  --output outputs/debug_run
```

## 10. 输出说明

完整运行后，输出结构如下：

```text
outputs/<run>/
  reports/
    dataset_check.json
    doctor.json
  extracted/
    front_rgb/
    front_depth/
    camera_intrinsics.json
  cuvslam/
    trajectory.csv
    final_landmarks.ply
    replay_snapshots.json
    cuvslam_pointcloud_replay.html
    summary.json
  prepared/
    front_cam/
    back_cam/
    lidar/
    depth/
    track.csv
  semantic_map/
    nav_graph.pkl
    nav_graph_contents.json
    nav_graph_stats.json
    object3d_tracking_summary.json
    object3d_global_map.html
    nav_graph_visualization.html
    nav_graph_visualization_3d.html
  rerun/
    semantic_topomap_replay.rrd
```

重点文件：

- `trajectory.csv`：cuVSLAM 过程轨迹。
- `final_landmarks.ply`：cuVSLAM 最终稀疏点云。
- `track.csv`：同步后的建图帧索引。
- `nav_graph.pkl`：可程序读取的导航图。
- `nav_graph_contents.json`：导航图内容导出。
- `nav_graph_stats.json`：节点数量、过滤统计和采样统计。
- `object3d_tracking_summary.json`：Object3D 跟踪与过滤过程摘要。
- `object3d_global_map.html`：三维物体全局地图。
- `nav_graph_visualization_3d.html`：三维导航图可视化。
- `semantic_topomap_replay.rrd`：Rerun 在线式回放文件。

## 11. 打开 Rerun 回放

全量测试输出中会打印类似命令：

```bash
PYTHONPATH=/tmp/rerun_sdk:/tmp/rerun_sdk/rerun_sdk \
  /tmp/rerun_sdk/bin/rerun \
  /home/zyf/Public/slam/outputs/dataset_test3_gpu_yolo_sam/rerun/semantic_topomap_replay.rrd
```

如果本机已经正常安装 `rerun` 命令，也可以直接：

```bash
rerun /home/zyf/Public/slam/outputs/dataset_test3_gpu_yolo_sam/rerun/semantic_topomap_replay.rrd
```

查看重点：

- 图像窗口中检测框是否随时间变化。
- 点云是否随时间逐渐增长，而不是一开始全量出现。
- 相机轨迹是否随时间增长。
- 物体三维框是否在合理位置出现。
- 途径点、拓扑节点和高层语义区域是否按时间逐步出现。

## 12. 当前 dataset_test3 全量验证结果

使用数据集：

```text
/home/zyf/Desktop/dataset_test3
```

使用配置：

```text
configs/gpu_yolo_sam.yaml
```

输出目录：

```text
/home/zyf/Public/slam/outputs/dataset_test3_full_test_gpu_yolo_sam
```

关键结果：

- SVO2 总帧数：3600
- cuVSLAM tracked 帧数：3600
- cuVSLAM failed 帧数：0
- 同步后建图帧：241
- 跳过同步帧：0
- 导入途径点：84
- 原始观测：98
- 拓扑节点：7
- 物体节点：17
- 语义区域节点：5
- Object3D 原始对象：45
- Object3D 稳定对象：24
- 重叠过滤删除对象：15
- 运动过滤删除对象：0
- 消失回访过滤删除对象：3
- Rerun 回放帧：241
- cuVSLAM 轨迹点：3600
- Rerun 使用点云点数：44098

速度现象：

- CPU 默认配置下，语义帧循环约 5.8 秒每帧。
- 混合 GPU 配置下，YOLO 推理约 6 到 9 毫秒每帧。
- 本次 241 帧语义帧循环约 2 分 50 秒完成。

## 13. 配置文件说明

### 13.1 `configs/default.yaml`

默认保守配置，全部核心运行设备为 CPU，适合迁移后先保证可跑通。

### 13.2 `configs/smoke.yaml`

冒烟测试配置，限制帧数，适合快速验证环境和流程闭环。

### 13.3 `configs/gpu_yolo_sam.yaml`

推荐配置。OPR 走 CPU，YOLO 和 SAM 走 CUDA。

适合当前机器状态，也适合大多数没有 CUDA 版 MinkowskiEngine 的环境。

### 13.4 `configs/gpu.yaml`

纯 GPU 配置。仅当 MinkowskiEngine 支持 CUDA 时使用。

## 14. 关键参数说明

Object3D 相关：

- `min_consecutive_frames`：物体至少连续出现多少帧才认为稳定。
- `overlap_filter_enabled`：是否启用同类三维框重叠去重。
- `motion_filter_enabled`：是否启用运动物体过滤。
- `motion_filter_classes`：被认为可能运动的类别，当前默认是 `person`。
- `disappearance_filter_enabled`：是否启用消失回访过滤。
- `disappearance_max_observation_distance_m`：消失判断只在近距离内生效，默认三米。
- `disappearance_min_visible_misses`：视场内连续未看到多少次后认为消失。

途径点采样相关：

- `min_distance_m`：距离变化超过该阈值保留途径点。
- `min_yaw_deg`：朝向变化超过该阈值保留途径点。
- `keep_first_last`：是否保留首尾帧。
- `keep_topology_change`：拓扑变化时是否强制保留。

Rerun 相关：

- `cloud_points: 0`：表示导出全量点云，而不是固定采样数量。
- `grow_pointcloud: true`：点云随时间逐步增长。
- `black_background: true`：黑色背景。
- `cloud_radius`：主点云半径。
- `topology_z`：拓扑层显示高度。
- `semantic_level_z_step`：高层语义区域之间的高度差。

## 15. 常见问题

### 15.1 为什么默认不直接全 GPU？

因为当前环境中的 MinkowskiEngine 是 CPU_ONLY 编译。OPR 位置描述符模型会使用 MinkowskiEngine，如果把总设备设为 CUDA，会在构建稀疏张量时报错。

推荐使用 `configs/gpu_yolo_sam.yaml`，让 OPR 保持 CPU，YOLO/SAM 使用 GPU。

### 15.2 为什么 YOLO/SAM 已经上 GPU，但还有 CPU 占用？

因为完整流程不只有深度学习推理，还包括：

- SVO2 文件读取
- cuVSLAM 轨迹处理
- 时间同步
- 点云过滤
- DBSCAN 聚类
- 三维框计算
- 跨帧跟踪合并
- 图结构构建
- JSON/HTML/Rerun 导出

这些环节多数仍在 CPU 上执行。

### 15.3 为什么有时帧循环速度变化很大？

每帧检测到的物体数量不同。检测框越多，SAM 分割次数、点云回投、聚类和 Object3D 后处理越多，因此同一段数据中不同帧速度会明显变化。

### 15.4 Rerun 中点云看起来不够密怎么办？

检查配置：

```yaml
rerun:
  cloud_points: 0
  grow_pointcloud: true
```

`cloud_points: 0` 表示不按固定数量采样，而是尽量使用全量点云。`grow_pointcloud: true` 表示按时间逐步增长。

### 15.5 为什么回放里最开始不应该出现完整轨迹？

在线式回放要求所有信息随时间逐步出现。当前 Rerun 导出已经把轨迹和拓扑层级改成按时间增长，而不是一开始静态显示全局结果。

### 15.6 物体位置左右颠倒或集中到轨迹中线怎么办？

优先检查坐标映射。当前正确映射是：

```text
[x, y, z] -> [x, z, -y]
```

如果迁移时重新接入位姿来源，必须确认位姿、点云、Object3D 和可视化使用同一世界坐标定义。

## 16. 迁移到新设备时的建议步骤

1. 复制整个 `/home/zyf/Public/slam` 文件夹到目标设备。
2. 确认目标设备 Python 版本为 3.10 或兼容版本。
3. 进入工程目录并执行 `bash scripts/setup_env.sh`。
4. 准备满足要求的数据集目录，至少包含 SVO2、SVO2 时间索引、后视图像和 LiDAR。
5. 运行 `doctor` 检查环境和数据。
6. 先运行 `smoke_test_dataset.sh` 验证 3 帧闭环。
7. 再运行全量流程。
8. 有 GPU 时优先使用 `configs/gpu_yolo_sam.yaml`。
9. 只有确认 MinkowskiEngine 支持 CUDA 后，才尝试 `configs/gpu.yaml`。
10. 打开 Rerun 文件检查时间回放、点云、轨迹、物体框和拓扑层级是否合理。

## 17. 推荐运行命令汇总

安装：

```bash
cd /home/zyf/Public/slam
bash scripts/setup_env.sh
```

环境检查：

```bash
source scripts/env.sh
python -m semantic_topomap.cli doctor \
  --dataset /home/zyf/Desktop/dataset_test3 \
  --output outputs/dataset_test3/reports/doctor.json
```

冒烟测试：

```bash
bash scripts/smoke_test_dataset.sh \
  /home/zyf/Desktop/dataset_test3 \
  outputs/dataset_test3_smoke
```

推荐全量运行：

```bash
bash scripts/run_pipeline.sh \
  /home/zyf/Desktop/dataset_test3 \
  outputs/dataset_test3_gpu_yolo_sam \
  configs/gpu_yolo_sam.yaml
```

打开 Rerun：

```bash
rerun outputs/dataset_test3_gpu_yolo_sam/rerun/semantic_topomap_replay.rrd
```

## 18. 当前工程边界

这套迁移包已经能完成从数据集到语义拓扑地图和 Rerun 回放的完整闭环，但仍有一些工程边界需要明确：

- 前视信息来自 SVO2，后视图像和 LiDAR 必须由数据集额外提供。
- 当前 Object3D 构建依赖 YOLO、SAM 和真实深度质量。
- 当前 OPR 的 MinkowskiEngine 在本环境中是 CPU_ONLY，不能直接全链路 CUDA。
- 三维物体框质量依赖深度对齐、坐标统一、mask 精度和前景点过滤。
- Rerun 是结果回放和调试工具，不改变建图结果本身。

## 19. 一句话总结

`/home/zyf/Public/slam` 是一个可迁移的多传感器语义拓扑建图工程包：它从 SVO2 和外部后视/LiDAR 数据出发，通过 cuVSLAM 恢复位姿和点云，利用 YOLO、SAM 和真实深度构建三维物体节点，再经过过滤、拓扑采样和语义层级聚合，最终输出导航图、三维物体地图和 Rerun 在线式回放。
