# 可迁移语义拓扑建图包方案

本文档说明 `/home/zyf/Public/slam` 这个工程包的迁移使用方案。目标是：把整个文件夹复制到另一台设备后，只要安装好运行环境，并提供符合要求的数据集，就可以从 ZED 录制文件开始，完成 cuVSLAM 位姿/点云提取、传感器同步、语义物体构建、过滤、拓扑建图和 Rerun 在线式回放导出。

## 1. 第一性原理输入

本系统只把真正必要的数据作为输入，不依赖历史中间结果。

必须提供：

- `zed/zed.svo2`：ZED 录制文件，是前视相机信息的主来源。
- `zed/zed_svo_index.csv`：ZED 图像时间戳与外部采集时间戳之间的同步桥。
- `rear_camera/rgb/`：后视图像，用于位置识别辅助。
- `lidar/scans/`：激光雷达原始扫描，用于位置识别辅助。
- `models/`：YOLO、SAM、OPR 等模型权重，已经随当前工程包复制。
- `runtime/`：ZED SDK 核心运行库、PyCuVSLAM Python 绑定、本地 wheel 和必要动态库，已经随当前工程包复制。

可选提供：

- `front_camera/rgb/`：如果目标设备暂时无法从 SVO2 导出前视图像，可临时作为调试回退。
- `depth/`：如果目标设备暂时无法从 SVO2 导出真实深度，可临时作为调试回退。
- `sync/alignment.csv`：如果存在，会优先使用其中的对齐关键帧，减少无意义帧处理。

其中最关键的是 `zed_svo_index.csv`。它需要至少包含：

- `frame_id`
- `jetson_timestamp_ns`
- `zed_image_timestamp_ns`
- `recorded_frame_index`

系统使用 `zed_image_timestamp_ns` 匹配 cuVSLAM 的位姿和点云，使用 `jetson_timestamp_ns` 匹配后视图像与激光雷达。

## 2. 系统主流程

完整流程按下面顺序执行：

1. 检查数据集结构，确认 SVO2、同步索引、后视图像、激光雷达是否存在。
2. 从 SVO2 导出前视图像、真实深度和相机内参。
3. 对同一个 SVO2 运行 cuVSLAM，得到过程位姿、稀疏点云和点云快照。
4. 根据同步索引，把 cuVSLAM 位姿、前视图像、深度、后视图像和激光雷达对齐为统一建图帧序列。
5. 将 cuVSLAM 坐标转换为系统世界坐标，转换规则为 `[x,y,z] -> [x,z,-y]`。
6. 使用 YOLO 检测前视图像中的物体，再用 SAM 生成物体掩膜。
7. 结合真实深度，把掩膜内像素回投成三维点云，并生成单帧 Object3D 观测。
8. 执行前景点过滤、DBSCAN 去噪、三米深度门限、跨帧跟踪合并和三维包围框生成。
9. 执行连续帧稳定过滤、同类三维框去重、运动物体过滤和消失回访过滤，得到稳定物体节点。
10. 使用 OPR 提取位置描述符，结合视觉相似度和空间距离构建拓扑节点。
11. 通过距离、朝向、首尾帧和拓扑变化采样途径点，减少冗余节点。
12. 聚合拓扑节点，生成更高层语义区域。
13. 导出语义拓扑图、三维可视化和 Rerun 在线式回放文件。

## 3. 目录结构

```text
slam/
  configs/                 # 默认参数
  docs/                    # 说明文档
  models/                  # 模型权重
  scripts/                 # 安装、检查、一键运行脚本
  semantic_topomap/        # 主流程代码
  third_party/             # Object3D、导航图、OPR 相关代码
  outputs/                 # 运行输出
```

## 4. 安装与检查

进入工程目录：

```bash
cd /home/zyf/Public/slam
```

安装 Python 依赖和本地 wheel：

```bash
bash scripts/setup_env.sh
```

直接运行脚本时会自动加载 `scripts/env.sh`。如果要手动进入环境，可以执行：

```bash
source scripts/env.sh
```

检查环境、模型、第三方代码和数据集：

```bash
python -m semantic_topomap.cli doctor \
  --dataset /home/zyf/Desktop/dataset_test3 \
  --output outputs/dataset_test3/reports/doctor.json
```

当前工程包已经内置并验证了 `pyzed.sl`、`cuvslam` 和 `rerun` 的运行入口。如果 `doctor` 报告这些模块缺失，优先检查是否已经运行 `bash scripts/setup_env.sh`，以及是否加载了 `scripts/env.sh`。

## 5. 一键运行

通用运行方式：

```bash
bash scripts/run_pipeline.sh <数据集目录> <输出目录> [配置文件]
```

例如：

```bash
bash scripts/run_pipeline.sh \
  /home/zyf/Desktop/dataset_test3 \
  /home/zyf/Public/slam/outputs/dataset_test3
```

也可以逐步运行，便于定位问题：

```bash
python -m semantic_topomap.cli check --dataset /home/zyf/Desktop/dataset_test3
python -m semantic_topomap.cli export-svo2 --dataset /home/zyf/Desktop/dataset_test3 --output outputs/dataset_test3
python -m semantic_topomap.cli run-cuvslam --dataset /home/zyf/Desktop/dataset_test3 --output outputs/dataset_test3
python -m semantic_topomap.cli prepare --dataset /home/zyf/Desktop/dataset_test3 --output outputs/dataset_test3
python -m semantic_topomap.cli build-map --output outputs/dataset_test3
python -m semantic_topomap.cli export-rerun --output outputs/dataset_test3
```

## 6. 当前测试状态

在当前设备上，已经用 `/home/zyf/Desktop/dataset_test3` 验证：

- 数据集检查通过。
- `pyzed.sl`、`cuvslam`、`rerun` 检查通过。
- SVO2 文件存在。
- 已从 SVO2 成功导出前视 RGB、真实深度和相机内参。
- 已从同一个 SVO2 成功运行 cuVSLAM，输出轨迹、稀疏点云和点云快照。
- 同步索引共有 3600 行。
- 通过同步表选出 241 个建图关键帧。
- 后视图像共有 241 帧。
- 激光雷达扫描共有 241 个同步输入。
- 已经生成统一建图帧序列，共 241 行，没有跳帧。
- 激光雷达原始包已经转换为系统可读的四维点云。
- 已完成 3 帧从零完整冒烟闭环，覆盖 SVO2 导出、cuVSLAM、同步准备、语义建图、Object3D 过滤、导航图构建和 Rerun 回放导出。

当前完整冒烟输出目录：

```text
/home/zyf/Public/slam/outputs/full_smoke_from_scratch
```

## 7. 迁移到新设备时的最小操作

1. 复制整个 `slam/` 文件夹到新设备。
2. 准备符合要求的数据集，至少包含 SVO2、同步索引、后视图像、激光雷达。
3. 运行 `bash scripts/setup_env.sh` 安装 Python 依赖和本地 wheel。
4. 运行 `doctor` 检查环境和数据。
5. 运行 `run_pipeline.sh`。
6. 在输出目录查看语义拓扑图和 Rerun 回放文件。

这套封装的核心原则是：前视数据来自 SVO2，位姿和稀疏点云来自 cuVSLAM，后视和 LiDAR 通过外部时间戳并入，同步后再进入统一语义拓扑建图流程。
