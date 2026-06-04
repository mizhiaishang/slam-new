# Semantic Topomap Portable

这是一个可迁移的多传感器语义拓扑建图工程包。设计原则是从数据集本身出发：SVO2 提供前视相机主信息，cuVSLAM 从同一个 SVO2 中生成过程位姿和稀疏点云，后视图像与 LiDAR 通过时间同步并入，随后统一进入 Object3D、过滤、拓扑建图和 Rerun 在线式回放流程。

## 数据集中必须提供什么

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

`zed_svo_index.csv` 是同步核心，需要包含：

```text
frame_id
jetson_timestamp_ns
zed_image_timestamp_ns
recorded_frame_index
```

其中 `zed_image_timestamp_ns` 用来匹配 SVO2/cuVSLAM 时间，`jetson_timestamp_ns` 用来匹配后视图像和 LiDAR。

## 安装

```bash
cd /home/zyf/Public/slam
bash scripts/setup_env.sh
```

工程包已内置当前验证可用的运行时资源：

- `runtime/zed_sdk/`：ZED SDK 核心运行库和标定目录。
- `runtime/python/cuvslam/`：PyCuVSLAM Python 绑定和运行库。
- `runtime/wheels/`：ZED Python API wheel。

安装脚本会安装 Python 依赖和本地 wheel；运行脚本会自动加载 `scripts/env.sh` 中的运行库路径。

## 被 Git 忽略的大文件资产

为了避免 GitHub 单文件大小限制和仓库体积过大，`.gitignore` 会忽略模型权重、动态库、点云、回放文件、视频和压缩包等大文件。下面这些文件是当前流程运行需要的本地大文件，但不建议直接提交到普通 Git 仓库：

| 路径 | 约大小 | 用途 |
|---|---:|---|
| `models/sam/sam_vit_b_01ec64.pth` | 357.7 MB | SAM 物体掩膜模型权重 |
| `models/depth/depth_anything_v2_metric_vkitti_vits.pth` | 94.6 MB | Depth Anything 深度模型权重 |
| `third_party/OpenPlaceRecognition/third_party/Depth-Anything-V2/weights/depth_anything_v2_metric_vkitti_vits.pth` | 94.6 MB | OPR/Depth-Anything 兼容路径下的深度权重 |
| `runtime/zed_sdk/lib/libsl_zed.so` | 60.6 MB | ZED SDK 核心动态库 |
| `models/opr/multi-image_lidar_late-fusion_itlp-finetune.pth` | 21.1 MB | OPR 位置识别模型权重 |
| `third_party/OpenPlaceRecognition/third_party/AdelaiDepth/examples/2.gif` | 20.0 MB | 第三方示例资源，运行主流程不依赖 |
| `third_party/OpenPlaceRecognition/third_party/AdelaiDepth/examples/3.gif` | 17.9 MB | 第三方示例资源，运行主流程不依赖 |
| `third_party/OpenPlaceRecognition/third_party/Depth-Anything-V2/metric_depth/dataset/splits/hypersim/train.txt` | 13.1 MB | 第三方训练 split 文件，运行主流程不依赖 |
| `runtime/zed_sdk/lib/libsl_ai.so` | 12.7 MB | ZED SDK AI 相关动态库 |
| `third_party/OpenPlaceRecognition/third_party/Depth-Anything-V2/assets/teaser.png` | 12.2 MB | 第三方说明图片，运行主流程不依赖 |

如果从 GitHub clone 后缺少这些文件，需要从本机备份、内部网盘、模型仓库或 Git LFS 重新放回相同路径。否则可能出现 SAM 初始化失败、深度模型加载失败，或 `pyzed.sl` / ZED SDK 动态库加载失败。

当前 `.gitignore` 还会忽略以下大文件类型：

```text
*.pth, *.pt, *.ckpt, *.safetensors, *.onnx, *.engine,
*.bin, *.weights, *.so, *.so.*, *.a, *.o,
*.tar, *.tar.gz, *.tgz, *.zip, *.7z, *.rar,
*.rrd, *.ply, *.pcd, *.las, *.laz,
*.svo, *.svo2, *.mp4, *.avi, *.mov
```

如果希望把这些大文件也纳入版本管理，建议使用 Git LFS，而不是直接提交到普通 Git 历史中。

## 环境和数据检查

```bash
python -m semantic_topomap.cli doctor \
  --dataset /home/zyf/Desktop/dataset_test3 \
  --output outputs/dataset_test3/reports/doctor.json
```

只检查数据集结构：

```bash
python -m semantic_topomap.cli check \
  --dataset /home/zyf/Desktop/dataset_test3 \
  --output outputs/dataset_test3/reports/dataset_check.json
```

## 冒烟测试

如果想确认迁移包能完整跑通小规模闭环，可以运行：

```bash
bash scripts/smoke_test_dataset.sh \
  /home/zyf/Desktop/dataset_test3 \
  /home/zyf/Public/slam/outputs/dataset_test3_smoke
```

它会执行 3 帧完整冒烟流程：环境检查、SVO2 前视图像/深度导出、cuVSLAM 位姿/点云、同步准备、语义建图和 Rerun 回放导出。

## 一键运行

```bash
bash scripts/run_pipeline.sh \
  /home/zyf/Desktop/dataset_test3 \
  /home/zyf/Public/slam/outputs/dataset_test3
```

也可以使用默认测试脚本：

```bash
bash scripts/run_dataset_test3.sh
```

## 分步运行

```bash
python -m semantic_topomap.cli export-svo2 \
  --dataset /home/zyf/Desktop/dataset_test3 \
  --output outputs/dataset_test3

python -m semantic_topomap.cli run-cuvslam \
  --dataset /home/zyf/Desktop/dataset_test3 \
  --output outputs/dataset_test3

python -m semantic_topomap.cli prepare \
  --dataset /home/zyf/Desktop/dataset_test3 \
  --output outputs/dataset_test3

python -m semantic_topomap.cli build-map \
  --output outputs/dataset_test3

python -m semantic_topomap.cli export-rerun \
  --output outputs/dataset_test3
```

## 输出结构

```text
outputs/<run_name>/
  extracted/      # 从 SVO2 导出的前视 RGB、深度、相机内参
  cuvslam/        # cuVSLAM 轨迹、稀疏点云、点云快照
  prepared/       # 统一时间同步后的建图帧序列
  semantic_map/   # Object3D、拓扑图和语义层级结果
  rerun/          # Rerun 在线式回放文件
  reports/        # 环境检查、数据检查和运行报告
```

## 当前验证状态

已经用 `/home/zyf/Desktop/dataset_test3` 验证：

- `doctor` 检查通过，`pyzed.sl`、`cuvslam`、`rerun` 均可导入。
- SVO2 小规模导出通过，已从 SVO2 导出前视 RGB、真实深度和相机内参。
- cuVSLAM 小规模运行通过，已输出轨迹、稀疏点云、点云快照和 HTML 回放。
- 同步准备通过，SVO2 前视图像/深度按 ZED 图像时间戳匹配，后视图像和 LiDAR 按外部采集时间戳匹配。
- 3 帧完整冒烟闭环通过，已生成 Object3D、导航图和 Rerun `.rrd` 回放。
- 全量同步准备链路通过：同步索引 3600 行，最终选出 241 个建图关键帧，准备阶段没有跳帧，LiDAR 原始包已转换为系统可读点云。

更完整的迁移说明见 [docs/portable_pipeline_plan.md](docs/portable_pipeline_plan.md)。
