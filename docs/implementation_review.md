# 从 dataset_test3 到 Rerun 全量点云语义拓扑回放的完整实现复盘

生成时间：2026-05-28  
项目目录：`/home/zyf/code_made`  
原始数据集目录：`/home/zyf/Desktop/dataset_test3`  
最终语义拓扑结果目录：`/home/zyf/code_made/result/semantic_graphs_dataset_test3_foreground_filter_full`  
最终 Rerun 回放文件：`/home/zyf/code_made/result/semantic_graphs_dataset_test3_foreground_filter_full/online_semantic_build_full_cloud_small_replay.rrd`

## 0. 文档目的

这份文档用于完整梳理我们为了实现当前这版功能，从你给出的原始数据集 `/home/zyf/Desktop/dataset_test3` 开始，到最终得到“RGB 检测结果 + 真实深度 3DBox + Object3D 过滤 + waypoint 采样 + 拓扑图 + 语义层级 + cuVSLAM 点云 + Rerun 在线式回放”的全过程。

这里的“当前这版功能”指的是最后你比较满意的这一版 Rerun 回放：

```bash
/home/zyf/code_made/result/semantic_graphs_dataset_test3_foreground_filter_full/online_semantic_build_full_cloud_small_replay.rrd
```

它具备以下主要能力：

- 使用 dataset_test3 的 RGB、真实深度、轨迹和 cuVSLAM 点云结果进行离线回放。
- 在 Rerun 中显示 RGB 图像和检测框。
- 在 3D 空间中显示随时间增长的相机轨迹、waypoint、Object3D 物体框、物体中心、拓扑节点、L2/L3 语义区域。
- 使用黑色背景，提高点云和拓扑可视化对比度。
- 点云半径经过调小，避免点云过粗遮挡结构。
- 不再在第 0 帧提前显示完整绿色 cuVSLAM 全轨迹，只显示随时间增长的当前轨迹。
- 使用全量 `final_landmarks.ply` 读取模式；在线增长点云来自 `replay_snapshots.json`，最终增长层包含约 1.5 万个唯一 landmark。
- 保存了最终语义拓扑图、Object3D 全局 3DBox 图、导航图 HTML/SVG、Rerun 回放文件和对应 summary。

这份文档不是单纯的命令记录，而是把我们一路遇到的问题、为什么改、最终代码里怎么做、结果怎么看、还有当前版本的边界都系统整理下来。后续如果需要复现、交接、扩展或写论文/汇报材料，可以直接以本文档作为技术底稿。

## 1. 原始数据集是什么

你给出的数据集根目录是：

```bash
/home/zyf/Desktop/dataset_test3
```

这个目录里有多种传感器数据。当前我们实际用到的核心内容包括：

```text
/home/zyf/Desktop/dataset_test3/front_camera/rgb
/home/zyf/Desktop/dataset_test3/rear_camera/rgb
/home/zyf/Desktop/dataset_test3/lidar/scans
/home/zyf/Desktop/dataset_test3/depth
/home/zyf/Desktop/dataset_test3/zed
/home/zyf/Desktop/dataset_test3/zed/depth
/home/zyf/Desktop/dataset_test3/sync/alignment.csv
/home/zyf/Desktop/dataset_test3/sync/timeline.json
/home/zyf/Desktop/dataset_test3/summary.json
```

数据集 summary 中记录的规模为：

```json
{
  "zed_frames": 3600,
  "rear_frames": 241,
  "lidar_packets": 90623,
  "exported_front_frames": 241,
  "aligned_rows": 241,
  "errors": []
}
```

这说明原始 ZED 序列有 3600 帧，但为了和 front camera、rear camera、LiDAR、同步表对齐，最终用于语义拓扑主流程的是 241 行对齐数据。也就是说，我们的语义拓扑构建不是处理全部 3600 帧 RGB，而是处理同步后的 241 个 front camera 关键帧。

在 `/home/zyf/Desktop/dataset_test3/depth` 中有 241 张深度 PNG，和这 241 个 front camera RGB 对齐。这个点非常关键，因为后续 2D 检测转 3DBox 的效果强依赖 RGB、bbox、depth 三者的对齐关系。前期我们怀疑过“物体位置不对是不是 RGB/depth/bbox 没对齐”，后来检查后确认，当前语义主流程实际使用的是这组 241 张同步深度，而不是随便拿的 3600 帧 ZED depth。

另外，`/home/zyf/Desktop/dataset_test3/zed/depth` 中有 3600 张深度图，它们对应 ZED 原始帧。这个目录更多服务于 ZED/cuVSLAM 侧的完整序列；语义拓扑主链路里使用的是已经对齐到 241 个 front frame 的 `/home/zyf/Desktop/dataset_test3/depth`。

## 2. 为什么要准备 code_made 可读的数据布局

原始数据集的组织形式和 `/home/zyf/code_made/run_semantic_topomap.py` 期望的数据布局不完全一致。`run_semantic_topomap.py` 依赖 OpenPlaceRecognition 的 ITLPCampus 数据读取逻辑，默认希望看到类似：

```text
dataset_root/
  track.csv
  front_cam/
  back_cam/
  lidar/
  depth/
```

因此我们写了数据准备脚本：

```bash
/home/zyf/code_made/prepare_dataset_test3_for_code_made.py
```

这个脚本的职责是把原始 `/home/zyf/Desktop/dataset_test3` 转换成 code_made 主流程可直接读取的布局：

```bash
/home/zyf/code_made/core_content/dataset_test3_cuvslam
```

最终这个目录中包括：

```text
front_cam/*.png
back_cam/*.png
lidar/*.bin
depth/*.png
track.csv
```

其中 RGB 和 depth 主要通过软链接或同步复制关系接入，LiDAR 则从原始 packet 轻量解码为 `.bin`。这一步的目的是让主流程不用理解原始数据集复杂的 sync 目录结构，而是像处理一个标准 OPR/ITLP 风格数据集一样处理它。

## 3. track.csv 的作用

主流程几乎所有逐帧处理都围绕 `track.csv` 展开。它提供了每个语义处理帧对应的：

- front camera 时间戳。
- rear camera 时间戳。
- LiDAR 时间戳。
- depth 时间戳。
- 位姿 `tx, ty, tz, qx, qy, qz, qw`。
- 对应的 cuVSLAM frame id。

在 `run_semantic_topomap.py` 中，主流程读取：

```python
self.track_csv = pd.read_csv(os.path.join(self.cfg.data_base_path, "track.csv"))
self.track_csv["front_cam_ts"] = self.track_csv["front_cam_ts"].astype(str)
self.track_csv["lidar_ts"] = self.track_csv["lidar_ts"].astype(str)
if "depth_ts" in self.track_csv.columns:
    self.track_csv["depth_ts"] = self.track_csv["depth_ts"].fillna("").astype(str)
```

随后每一帧都根据 `front_cam_ts` 读取 RGB，根据 `depth_ts` 或 `front_cam_ts` 找到 depth PNG，根据位姿字段构造当前相机姿态。

因此可以把 `track.csv` 理解为整条离线回放链路的“时间轴索引”。没有它，RGB、深度、LiDAR、轨迹、拓扑节点和 Object3D 观测之间就无法统一到同一帧。

## 4. cuVSLAM 轨迹如何接入 code_made 世界坐标

我们曾经遇到一个很明显的问题：RGB 图像中门一直在左侧，但 3D 地图中物体却被投到了右侧或轨迹中间。这不是 YOLO 框的问题，也不是 SAM mask 的问题，而是坐标系转换问题。

cuVSLAM/ZED 常见坐标约定可以理解为：

```text
x: right
y: down
z: forward
```

而 code_made 语义地图和导航图更适合使用：

```text
x: right
y: forward
z: up
```

因此我们需要固定轴映射：

```python
[x, y, z] -> [x, z, -y]
```

在 `prepare_dataset_test3_for_code_made.py` 中，函数 `remap_slam_pose` 做了这件事：

```python
basis = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, -1.0, 0.0],
    ],
    dtype=np.float64,
)
mapped_position = basis @ position
mapped_rotation = basis @ Rotation.from_quat(quat).as_matrix() @ basis.T
mapped_quat = Rotation.from_matrix(mapped_rotation).as_quat()
```

这个转换不仅改变 position，也改变 rotation。只改 position 不改 rotation 会导致轨迹点看起来位置大致对，但相机朝向、物体回投方向、视场判断仍然错。

后续 Rerun 点云导出脚本 `/home/zyf/code_made/export_rerun_semantic_replay.py` 中也复用了同样的轴映射函数：

```python
def zed_to_code(point) -> list[float]:
    x, y, z = [float(v) for v in point]
    return [x, z, -y]
```

这保证了 cuVSLAM 点云、cuVSLAM 轨迹、code_made 语义地图、Object3D 物体框在同一个世界坐标里显示。

## 5. YOLO 为什么要重新训练

最开始我们使用的是通用 YOLOv8 权重。通用权重能识别一些常见类别，但对于当前数据集中的桌面/室内/门/椅子等具体场景，检测结果不够理想。你希望提升 YOLO 的识别能力，因此我们引入了标注和训练流程。

我们讨论并尝试过 LabelImg、CVAT、Label Studio 等标注工具。最后你的标注数据位于：

```bash
/home/zyf/Desktop/train
```

原始 RGB 图片位于：

```bash
/home/zyf/Desktop/dataset_test3/front_camera/rgb
```

我们把标注整理为 YOLO 数据集格式，生成数据集目录：

```bash
/home/zyf/Desktop/yolo_dataset_test3_front
```

训练命令大致如下：

```bash
cd /home/zyf/code_made
CUDA_VISIBLE_DEVICES=2 yolo detect train \
  model=/home/zyf/code_made/core_content/yolov8n.pt \
  data=/home/zyf/Desktop/yolo_dataset_test3_front/data.yaml \
  epochs=100 \
  imgsz=640 \
  batch=8 \
  device=0 \
  project=/home/zyf/Desktop/yolo_train_runs \
  name=test3_front \
  exist_ok=True \
  workers=4
```

训练过程中我们遇到过两个问题。

第一个问题是 CUDA out of memory。原因不是模型特别大，而是当时指定的 GPU 或 PyTorch 实际可见设备上已经有占用，或者 `CUDA_VISIBLE_DEVICES` 与 `device` 参数理解不一致。`CUDA_VISIBLE_DEVICES=2 device=0` 的含义是“只暴露物理 GPU 2，然后在进程内部把它看作 cuda:0”。如果写成 `CUDA_VISIBLE_DEVICES=2 device=2`，就可能让 Ultralytics 在内部寻找 cuda:2，造成混乱。

第二个问题是 YOLO 在 AMP 检查时尝试加载 `yolo26n.pt`，并报 checkpoint zip 损坏。这不是我们要训练 YOLOv26，而是当前 Ultralytics 版本内部做 AMP check 时使用了默认检查权重。解决方式是避免让它自动下载/读取损坏权重，或者关闭 AMP 检查，或者清理损坏缓存。最终训练成功后结果保存在：

```bash
/home/zyf/Desktop/yolo_train_runs/test3_front/weights/best.pt
```

当前 `config.yaml` 已经指向训练好的 YOLO：

```yaml
yolo_weights: /home/zyf/Desktop/yolo_train_runs/test3_front/weights/best.pt
```

这也是后续语义拓扑主流程使用的检测模型。

## 6. 主流程配置

当前主流程配置文件是：

```bash
/home/zyf/code_made/config.yaml
```

关键配置包括：

```yaml
database_track_dir: /home/zyf/code_made/core_content/dataset_test3_cuvslam
data_base_path: /home/zyf/code_made/core_content/dataset_test3_cuvslam
yolo_weights: /home/zyf/Desktop/yolo_train_runs/test3_front/weights/best.pt
depth_model_path: /home/zyf/code_made/core_content/depth_anything_v2_metric_vkitti_vits.pth
output_dir: /home/zyf/code_made/semantic_graphs
device: cpu
max_frames: -1
hfov: 91.0
vfov: 65.0
object3d_engine_root: /home/zyf/Desktop/3d
sam_checkpoint: /home/zyf/code_made/core_content/sam_vit_b_01ec64.pth
sam_model_type: vit_b
sam_device: null
```

这里有几个容易误解的点。

第一，`device: cpu` 主要是 OPR/DepthAnything 等主流程默认设备。YOLO 可以通过 `--yolo-device` 单独指定，SAM 也可以通过 `--sam-device` 指定。代码里 SAM device 的选择逻辑是：

```python
sam_device=cfg.get("sam_device") or cfg.get("yolo_device") or cfg.device
```

所以如果命令行给了 `--yolo-device cuda:0` 且 `sam_device` 为空，SAM 会跟着跑到同一个 CUDA 设备上。

第二，当前真实 depth 已经可用，所以主流程优先使用 recorded depth，不再依赖 DepthAnything+LiDAR 的估计深度。DepthAnything 模型路径仍保留，是 fallback 用的。

第三，`max_frames: -1` 表示全量跑完 241 帧，不再只跑 smoke test。

## 7. 每一帧如何处理

在 `run_semantic_topomap.py` 中，每一帧大致走如下流程：

1. 读取 RGB 图像。
2. 读取 LiDAR 点云。
3. 读取当前帧位姿。
4. 使用 YOLO 对 RGB 做 2D 检测。
5. 优先读取 recorded depth PNG。
6. 如果没有 recorded depth，则使用 DepthAnything+LiDAR fallback。
7. 构造当前帧语义 payload。
8. 使用 OPR descriptor 判断当前帧属于已有拓扑节点还是新拓扑节点。
9. 把 YOLO + SAM + depth 产生的 3D 物体观测交给 Object3D tracking engine。
10. 最后把帧级 payload 放入后处理队列。

这一步输出的不是最终导航图，而是逐帧语义观测。真正的 Object3D 过滤、waypoint 采样、nav graph 构建发生在全帧处理完成之后。

## 8. 深度信息从哪里来

当前主流程使用的深度目录是：

```bash
/home/zyf/code_made/core_content/dataset_test3_cuvslam/depth
```

这个目录对应的是原始数据集中的：

```bash
/home/zyf/Desktop/dataset_test3/depth
```

数量为 241 张，和主流程的 241 个 front camera 处理帧对应。

在代码中，深度加载逻辑为：

```python
if np.issubdtype(depth_raw.dtype, np.integer):
    depth_m = depth_raw.astype(np.float32) / 1000.0
else:
    depth_m = depth_raw.astype(np.float32)
```

也就是说，如果 PNG 是整数深度，就按毫米转米；如果本身是 float，就直接当米使用。最终所有 Object3D 逻辑中的深度单位都是米。

我们曾经讨论过“是不是所有深度都是 3m”。答案是否定的。3m 是我们人为设定的最大观测距离过滤阈值，不是深度图本身都是 3m。真实 depth PNG 中每个像素有自己的深度值，只是在构建物体节点时，超过 3m 的物体观测不进入 Object3D。

## 9. 2D 检测如何转成 3DBox

当前 2D 到 3D 的核心链路是：

```text
RGB 图像
  -> YOLO 2D bbox
  -> SAM 根据 bbox 生成 mask
  -> mask 内像素读取真实 depth
  -> 根据相机内参把像素回投成局部 3D 点云
  -> 前景点云清洗
  -> 3m 深度门限
  -> Object3D observation
  -> 跨帧 tracking
  -> 3D bbox
```

这条链路的第一性原理是：2D bbox 本身没有深度，也没有真实 3D 尺寸。要得到 3DBox，必须利用深度图把 bbox 或 mask 内的像素变成点云。SAM 的作用不是识别类别，而是把 YOLO 给出的矩形框进一步收缩成更像物体轮廓的 mask。这样比直接用 bbox 内所有像素回投更合理，因为 bbox 内通常包含大量背景、墙面、地板或桌面。

回投公式可以理解为：

```text
z = depth(u, v)
x = (u - cx) * z / fx
y = (v - cy) * z / fy
```

这里的 `(x, y, z)` 是相机 optical 坐标：

```text
x: 图像右方
y: 图像下方
z: 相机前方
```

得到局部点云之后，需要结合当前相机位姿变换到世界坐标。世界坐标下的点云再用于估计物体中心、尺寸和 3DBox。

## 10. 为什么 Object3D 不能简单理解为“跑 GPU”

我们之前讨论过 Object3D 能不能上 GPU。结论是：YOLO 和 SAM 可以使用 GPU，DepthAnything 也可以使用 GPU，但 Object3D 的很多部分是几何后处理和 Python/NumPy/聚类/匹配逻辑，不是一个端到端神经网络推理模块。

Object3D 的主要耗时包括：

- SAM mask 生成。
- mask 内点云回投。
- 点云清洗。
- DBSCAN 去噪。
- 3D bbox 估计。
- 跨帧观测匹配。
- 后处理过滤。

其中 SAM 可以上 GPU，而且我们已经让它具备跟随 `sam_device` / `yolo_device` 的能力。YOLO 也可以上 GPU。但点云清洗、DBSCAN、3DBox 构造、track 合并等逻辑仍主要在 CPU 上执行。它们不是不能永远上 GPU，而是如果要上 GPU，需要重写为 CuPy/Open3D CUDA/torch tensor 版本，工程量会明显变大。

所以当前加速方向不是“把 Object3D 整体 `.to(cuda)`”，而是：

- YOLO 用 GPU。
- SAM 用 GPU。
- 减少每帧进入 Object3D 的候选数量。
- 使用 3m 深度过滤减少远处噪声目标。
- 使用 waypoint 采样减少导航图节点。
- 使用前景点云过滤减少 3DBox 噪声。
- 必要时再把点云聚类和 bbox 估计替换为 GPU 版本。

## 11. SAM 为什么被加入链路

最初的 Object3D 3DBox 效果不好，一个重要原因是 bbox 内点云过于发散。bbox 是矩形，真实物体通常不是矩形。比如椅子 bbox 里可能包含背景墙、地面、桌腿、其他椅子边缘；门 bbox 里可能包含门外区域、门框、墙面；这些深度点一旦全部用于 3DBox，就会导致物体框过大、偏移或不符合现实。

SAM 的引入是为了把“矩形框内所有点”变成“更接近物体前景区域的点”。当前使用的是：

```yaml
sam_checkpoint: /home/zyf/code_made/core_content/sam_vit_b_01ec64.pth
sam_model_type: vit_b
```

在 `run_semantic_topomap.py` 中，SAM 初始化逻辑如下：

```python
sam = sam_model_registry[self.sam_model_type](checkpoint=str(checkpoint))
sam.to(device=self.sam_device)
sam.eval()
self.object3d_mask_provider = SamMaskProvider(SamPredictor(sam))
```

YOLO 提供 bbox，SAM 使用 bbox 作为 prompt，输出 mask。然后 Object3D 用 mask 内深度点生成 3D 观测。

## 12. 前景点云过滤为什么需要

即使使用 SAM，mask 仍然可能包含背景点。尤其在深度边缘、反光、半遮挡、门框、桌面边缘等场景中，mask 内点云可能不集中。你提出一个很重要的判断：当 3DBox 内的点过于分散时，去掉和中心太偏离的点是合理的。

我们因此加入了前景点云过滤相关设置：

```python
settings = EngineSettings(
    voxel_size=0.02,
    dbscan_remove_noise=True,
    dbscan_eps=0.10,
    dbscan_min_points=20,
    min_points_threshold=50,
    mask_area_threshold=18,
    observation_max_depth_m=3.0,
    foreground_depth_filter_enabled=True,
    foreground_depth_window_m=0.30,
    foreground_center_filter_enabled=True,
    foreground_center_distance_percentile=90.0,
    foreground_min_points_threshold=50,
    foreground_rerun_clean=True,
    use_oriented_bbox=False,
)
```

这组参数背后的思路是：

- `dbscan_remove_noise=True`：先用聚类去掉离群噪声点。
- `dbscan_eps=0.10`：聚类半径控制在 10cm。
- `dbscan_min_points=20`：少于 20 个点不形成可靠 cluster。
- `min_points_threshold=50`：最终至少要有 50 个有效点才能生成稳定观测。
- `observation_max_depth_m=3.0`：只保留 3m 内物体观测。
- `foreground_depth_window_m=0.30`：优先保留深度上接近主体前景的一层点，避免背景墙把框拉长。
- `foreground_center_filter_enabled=True`：对距离中心过远的点做剔除。
- `foreground_center_distance_percentile=90.0`：保留中心距离分布中较主体的 90% 点，去掉最离散的尾部点。
- `foreground_min_points_threshold=50`：前景过滤后仍需要足够点数，否则这次观测不可靠。

这套过滤的目标不是让物体框“更好看”而已，而是让 3DBox 更符合物体主体结构，减少背景点对中心和尺寸的污染。

## 13. 3m 深度过滤的定位

3m 深度过滤不是最后的导航图过滤，而是 Object3D 单帧观测生成阶段的早期过滤。它发生在：

```text
YOLO bbox -> SAM mask -> depth 回投 -> 点云清洗 -> 计算 mask 主体深度 -> 判断是否 <= 3m
```

如果一个检测结果的 mask 主体深度超过 3m，它不会进入 Object3D tracking engine。这样做的原因有两个。

第一，远距离物体的深度噪声更大，3DBox 更容易漂移。尤其室内小物体，远距离时 mask 边缘和 depth 对齐误差会被放大，导致 3DBox 偏离现实。

第二，我们当前系统目标更偏向可导航、可交互、可确认的近距离语义物体。超过 3m 的物体即使 YOLO 识别到了，也不一定适合作为稳定导航物体节点保存。

3m 过滤使用的是 mask 内点云的前向深度，而不是世界坐标 z，也不是 bbox 中心像素。判断单位是米。

## 14. 3m 过滤和消失回访过滤的关系

我们特别讨论过一个风险：如果只允许 3m 内的物体进入观测，那么物体先被看到，后面离相机超过 3m 时没有被检测到，会不会被消失过滤误删？

最终我们采用的闭环逻辑是：

```text
只在 3m 内做消失回访判断。
```

也就是说，一个物体如果在后续帧几何上超过 3m，就不把这一帧当成“应该看见它”的机会。只有当满足以下条件时，才认为这是一次有效回访检查：

- 物体中心在当前相机前方。
- 物体中心到当前相机的前向深度不超过 3m。
- 物体中心到相机的欧式距离也不超过 3m。
- 物体中心在安全 FOV 内。
- 当前相机位置距离历史观测位置足够近，说明是在回访同一区域。

代码中对应逻辑在 `_evaluate_object_visibility_opportunity`：

```python
forward_z = float(local_point[2])
if forward_z <= 0.15:
    return {"eligible": False, "reason": "behind_camera"}
if forward_z > max_observation_distance_m:
    return {"eligible": False, "reason": "outside_object3d_depth_gate"}

camera_distance_m = float(np.linalg.norm(local_point[:3]))
if camera_distance_m > max_observation_distance_m:
    return {"eligible": False, "reason": "too_far"}
```

这里的 `max_observation_distance_m` 默认也是 3.0。这样就保证了：3m 深度过滤不会要求后续系统在 3m 外证明“物体还在”。只有当相机真的回到足够近、视场内、按几何应该能看到物体时，才判断它是否消失。

## 15. 消失回访过滤如何删除物体

Object3D 先产生稳定物体轨迹，然后消失过滤检查这些轨迹在后续回访中是否还存在。

对于每个稳定物体，系统会从它最后一次被观察到的帧之后开始扫描后续 payload。每一帧先判断是不是有效回访机会。如果不是，就跳过，不增加 miss。如果是有效回访机会，再尝试在当前帧找同类、空间距离足够近的检测结果。

如果找到了匹配检测：

- `matched_revisit_count += 1`
- 连续 miss 清零。
- 更新最后确认帧。

如果找不到匹配检测：

- `visible_miss_count += 1`
- `consecutive_visible_miss_count += 1`
- 如果连续 miss 达到阈值，就删除。

当前阈值为：

```python
object3d_disappearance_min_visible_misses = 2
object3d_disappearance_match_distance_m = 1.0
object3d_disappearance_position_tolerance_m = 1.0
object3d_disappearance_fov_margin_deg = 8.0
```

这意味着只有连续两次“3m 内、视场内、回访位置附近、理论上应该看到，但没有找到匹配检测”时，才把物体从最终图中删除。这样比“一帧没看到就删”稳健很多。

## 16. 运动物体过滤如何处理

我们讨论过运动物体的逻辑闭环。最终原则是：

```text
运动物体不需要保留在最终语义拓扑图内。
```

但不是所有类别都默认可运动。你明确提出：

```text
chair / door 默认不能移动。
```

当前默认配置中：

```python
object3d_motion_filter_classes = "person"
```

这意味着只有 `person` 被当作需要运动判断的类别。对于不在这个列表里的类别，比如 `chair`、`door`，系统默认它们是静态物体，不因为中心点跨帧有微小变化就走运动删除逻辑。

运动判断基于 3m 内连续观测的 3DBox 中心历史，主要检查：

- 连续观测数量是否足够。
- center span 是否过大。
- median step 是否过大。
- single step 是否过大。
- net displacement 是否和 span 同时过大。

核心阈值包括：

```python
object3d_motion_min_consecutive_observations = 2
object3d_motion_static_max_center_span_m = 1.0
object3d_motion_static_max_median_step_m = 0.35
object3d_motion_static_max_single_step_m = 1.2
```

如果类别不在 `motion_filter_classes` 中，直接返回：

```python
motion_state = "static"
motion_reason = "class_not_in_motion_filter_classes_static_assumed"
```

这样就避免了 chair/door 因为 3DBox 抖动、深度噪声或位姿误差被误判为运动物体。

## 17. Object3D 稳定过滤

Object3D tracking engine 会从单帧观测中合并出一批 track。但不是所有 track 都应该进入导航图。短暂出现一帧的检测可能是误检，也可能是 mask/depth 偶然成功。因此我们保留了最基础的稳定过滤：

```python
object3d_min_consecutive_frames = 2
```

也就是说，一个物体至少要在连续两帧里被跟踪到，才有资格成为稳定 Object3D。

这也是为什么我们曾经做一帧测试时，结果里可能有 YOLO 检测、有 Object3D 单帧观测，但最终 `object_nodes = 0`。一帧测试天然无法满足连续两帧稳定条件。

最终 foreground_filter_full 结果中：

```text
object3d_raw_object_count = 45
object3d_stable_object_count = 24
object3d_min_consecutive_frames = 2
```

说明 Object3D 原始 track 有 45 个，通过连续帧稳定过滤后剩下 24 个稳定 track。

## 18. Object3D 重叠去重过滤

同一个真实物体可能被 YOLO/SAM/Object3D 分裂成多个 track。例如一把椅子可能因为遮挡或 mask 差异被分成多个相邻框。为了避免最终导航图里同一位置出现多个重复物体，我们保留了同类 3D 重叠去重。

关键参数：

```python
object3d_overlap_filter_enabled = True
object3d_overlap_iou_threshold = 0.05
object3d_overlap_min_ratio_threshold = 0.35
```

它的含义是：对于同类物体，如果两个 3DBox 的 IoU 或交集占较小体积比例达到阈值，则认为它们可能是重复物体。系统倾向保留观测次数更多、支持点更多、更稳定的那个。

最终结果中：

```text
object3d_overlap_removed_object_count = 15
```

这说明重叠去重过滤删除了 15 个重复或高度重叠的 Object3D track。

## 19. Object3D 最终统计

最终语义拓扑结果目录：

```bash
/home/zyf/code_made/result/semantic_graphs_dataset_test3_foreground_filter_full
```

其中 `nav_graph_stats.json` 记录了最终主结果：

```text
imported_waypoints = 86
imported_observations = 100
topology_nodes = 4
waypoint_nodes = 86
object_nodes = 18
semantic_region_nodes = 3
raw_observations = 100
object3d_raw_object_count = 45
object3d_stable_object_count = 24
object3d_overlap_removed_object_count = 15
object3d_motion_removed_object_count = 0
object3d_disappearance_removed_object_count = 3
```

这里可以理解为：

- YOLO/SAM/depth/Object3D 先产生了 45 个原始物体 track。
- 连续帧稳定后剩下 24 个。
- 同类重叠去重删除 15 个。
- 运动过滤没有删除物体，因为当前运动过滤主要针对 `person`，最终没有符合删除条件的运动物体。
- 消失回访过滤删除 3 个。
- 进入 nav graph 的最终 object 节点为 18 个。

注意 `24 - 15 - 3 = 6` 看起来和 18 不一致，是因为这些统计来自不同阶段和不同集合口径：稳定 track、overlap removed、disappearance removed、nav graph object prune 并不是简单线性相减。nav graph 中的 object 节点来自过滤后的稳定 Object3D 结果和 waypoint observations 的合并导入，还包括后续图构建阶段的对象节点合并逻辑。

## 20. waypoint 为什么要采样

最开始每一帧都保存 waypoint，会导致拓扑图过密、可视化混乱、导航图节点太多，而且很多相邻帧位置差异很小，没有必要都成为导航节点。

我们因此加入 waypoint 采样。当前默认逻辑：

```python
waypoint_sampling_enabled = True
waypoint_min_distance_m = 0.8
waypoint_min_yaw_deg = 25.0
waypoint_keep_first_last = True
waypoint_keep_topology_change = True
```

规则可以朴素理解为：

- 第一帧一定保留。
- 最后一帧一定保留。
- 距离上一个保留 waypoint 超过 0.8m，保留。
- 朝向变化超过 25 度，保留。
- 拓扑节点发生变化，保留。
- 其他相邻帧不保留。

最终统计：

```text
input_frames = 241
kept_waypoints = 86
removed_waypoints = 155
```

保留原因统计：

```text
first = 1
topology_change = 33
distance = 48
yaw = 9
last = 1
```

这说明 waypoint 采样确实减少了节点数量，同时保留了空间移动、朝向变化和拓扑切换这些关键时刻。

## 21. 拓扑节点是如何形成的

拓扑节点不是预先手工指定数量，而是由算法根据帧的视觉位置描述符和空间位置关系动态形成。

主流程中使用 OpenPlaceRecognition 生成每帧 descriptor，再和历史节点比较。如果当前帧和已有节点足够相似，并且空间坐标也足够近，就归入已有拓扑节点；如果视觉距离和坐标距离都超过阈值，就创建新节点。

相关阈值：

```yaml
feature_distance_threshold: 1.8
coord_distance_threshold: 2.5
new_node_distance_threshold: 2.7
```

最终 nav graph 中形成：

```text
topology_nodes = 4
```

这些 topology node 代表更高层的空间区域或地点簇。waypoint 是轨迹上的关键帧节点，topology 是 waypoint 的聚类/归属上层。

## 22. 语义层级 L2/L3 是如何形成的

除了底层 waypoint 和 object 节点，我们还生成了语义区域层级。

最终统计：

```text
semantic_region_nodes = 3
level_counts:
  level_1 = 4
  level_2 = 2
  level_3 = 1
```

这里的 level_1 可以理解为 topology 层，level_2 是由若干 topology 聚合出的语义区域，level_3 是更高层区域。最终 Rerun 中你看到的 `L2:chair_battle_region`、`L3:chair_door_region` 等，就是从最终语义层级结构中提取出来的。

需要注意：当前 Rerun 回放不是主流程真实运行时逐帧记录的 event log，而是从最终 `nav_graph_contents.json` 反推生成的在线式可视化。因此 L2/L3 的出现时机是根据“当前出现的 topology 是否属于最终语义区域”推断出来的，不是完全真实的在线聚类历史。

## 23. 最终输出文件

最终语义结果目录为：

```bash
/home/zyf/code_made/result/semantic_graphs_dataset_test3_foreground_filter_full
```

主要文件包括：

```text
nav_graph.pkl
nav_graph_contents.json
nav_graph_stats.json
nav_graph_visualization.html
nav_graph_visualization_3d.html
nav_graph_visualization.svg
object3d_global_map.html
object3d_tracking_summary.json
cuvslam_semantic_3dbox_overlay.html
cuvslam_semantic_3dbox_overlay.summary.json
online_semantic_build_full_cloud_small_replay.rrd
online_semantic_build_full_cloud_small_replay.summary.json
```

其中：

- `nav_graph.pkl` 是 NetworkX 图对象。
- `nav_graph_contents.json` 是可读的图内容导出。
- `nav_graph_stats.json` 是统计信息和过滤摘要。
- `nav_graph_visualization.html` 是 2D 导航图可视化。
- `nav_graph_visualization_3d.html` 是 3D 导航图可视化。
- `object3d_global_map.html` 是 Object3D 物体框全局图。
- `object3d_tracking_summary.json` 是 Object3D track 和过滤结果详情。
- `cuvslam_semantic_3dbox_overlay.html` 是 cuVSLAM 点云与语义 3DBox overlay。
- `online_semantic_build_full_cloud_small_replay.rrd` 是最终 Rerun 回放。

## 24. cuVSLAM 点云回放如何生成

你后来提出需要 cuVSLAM 点云回放，并希望把点云、3DBox、waypoint、拓扑节点对应起来。我们使用 cuVSLAM/ZED SVO2 数据生成了点云回放目录：

```bash
/home/zyf/imu/cuvslam/results/dataset_test3_zed_svo2_pointcloud_replay
```

其中主要文件为：

```text
trajectory.csv
final_landmarks.ply
replay_snapshots.json
cuvslam_pointcloud_replay.html
summary.json
```

`summary.json` 中记录：

```json
{
  "processed_frames": 3600,
  "tracked_frames": 3600,
  "failed_frames": 0,
  "snapshot_stride": 30,
  "snapshot_count": 121,
  "final_landmarks_count": 45421
}
```

这说明 cuVSLAM 侧完整处理了 3600 帧，并生成了 45421 个最终 landmarks。`replay_snapshots.json` 每 30 帧保存一个点云快照，共 121 个 snapshot，用于做“点云随时间增长”的回放。

## 25. 为什么最终增长点云不是 45421 个

最后我们生成全量点云版 Rerun 时，你问“点云数目是所有的点云都记录了吗，还是固定数量，我需要全量点云”。我们检查后确认：

```text
final_landmarks.ply = 45421 个点
replay_snapshots.json = 121 个 snapshot
snapshot 中唯一 landmark id 约 15079 个
```

最终导出命令使用了：

```bash
--cloud-points 0
--max-growing-cloud-points 100000
```

这里 `--cloud-points 0` 表示读取 `final_landmarks.ply` 时不采样，读取全量 45421 个点。`--max-growing-cloud-points 100000` 表示增长点云最多保留 10 万点，大于当前 snapshot 唯一点数量，因此不会截断 snapshot 点。

但 Rerun 里“随时间增长”的点云来自 `replay_snapshots.json`，不是直接把最终 `final_landmarks.ply` 每个点都安排到某一帧出现。由于 snapshot 文件中最终只覆盖约 15079 个唯一 landmark，所以在线增长层最终显示的是这约 1.5 万个点。

这不是导出脚本截断造成的，而是 snapshot 数据本身没有包含全部 45421 个 final landmarks 的逐时刻出现记录。最终 PLY 是全局最终点云；snapshot 是回放点云。两者口径不同。

如果未来必须让 45421 个点也“随时间增长”，需要在生成 cuVSLAM replay snapshot 时保存完整 landmark 生命周期，或者把 final landmarks 按最近观测帧/首次出现帧重新分配到时间轴上。这需要 cuVSLAM 导出阶段提供更完整的 per-landmark 时间信息。

## 26. Rerun 回放为什么要做

最开始我们用 HTML/Plotly 做过 `object3d_global_map.html`、`nav_graph_visualization_3d.html` 和一些拓扑回放 HTML。它们适合静态查看最终地图，但不太适合模拟“在线构建”的过程。

你希望看到：

- RGB 图像随时间播放。
- YOLO 检测框随时间显示。
- 点云随着时间逐渐出现。
- 相机轨迹随时间增长。
- waypoint 随时间出现。
- 物体节点在坐标位置出现。
- 删除时物体能消失或标记。
- 拓扑节点和 L2/L3 高层节点也随时间出现。
- 3D 视图背景更清楚，最好黑色。

因此我们写了 Rerun 导出脚本：

```bash
/home/zyf/code_made/export_rerun_semantic_replay.py
```

它读取三个输入：

```text
result_dir: /home/zyf/code_made/result/semantic_graphs_dataset_test3_foreground_filter_full
dataset_root: /home/zyf/code_made/core_content/dataset_test3_cuvslam
cuvslam_dir: /home/zyf/imu/cuvslam/results/dataset_test3_zed_svo2_pointcloud_replay
```

然后输出 `.rrd` 文件，供 Rerun Viewer 打开。

## 27. Rerun 导出脚本做了什么

`export_rerun_semantic_replay.py` 的核心工作是把最终语义结果、原始 RGB、cuVSLAM 点云快照整合到一个时间轴中。

每一帧执行：

- 设置 Rerun timeline：`frame` 和 `timestamp`。
- 读取当前 RGB 图像。
- 如果当前 waypoint 有 observations，就把 bbox 画到 RGB 上。
- 记录 `camera/rgb`。
- 记录 `camera/detections`。
- 记录当前相机位置 `world/current_camera`。
- 记录当前累计轨迹 `world/trajectory_so_far`。
- 从 cuVSLAM snapshot 中累计 landmark 点，记录 `world/cuvslam_landmarks_growing`。
- 记录当前已出现 waypoint 和 waypoint path。
- 根据 Object3D first_seen 和 active_object_ids 记录物体 3DBox。
- 如果开启 include_removed，记录被删除物体。
- 根据当前已出现 topology 推断并显示 topology/L2/L3 层级。
- 记录 hierarchy edges。

最终生成 summary：

```json
{
  "frames": 241,
  "cloud_points": 45421,
  "grow_pointcloud": true,
  "growing_cloud_points_final": 14986,
  "topology_nodes": 4,
  "semantic_region_nodes": 3,
  "waypoints": 86,
  "object_nodes": 18,
  "removed_objects": 3
}
```

## 28. Rerun 不是完整真实事件日志

我们需要明确一个边界：当前 Rerun 回放是“最终结果倒推版”，不是主流程运行时每一步事件的原始日志。

原因是 `nav_graph_contents.json` 保存的是最终图结构，而不是每一帧内部发生的全部事件。它知道最终有哪些 waypoint、object、topology、semantic region，也知道一些 first_seen/last_seen 信息，但不知道所有原始 YOLO 检测、每次过滤的中间状态、每个 object track 的逐帧生命周期事件。

所以当前 Rerun 能做到“在线式展示”，但不能严格还原：

- 每一帧原始 YOLO 全部检测。
- 每个检测为什么被 3m 过滤。
- 每个检测为什么被前景过滤。
- Object3D track 每一帧的创建、合并、拆分。
- 重叠过滤在具体哪一帧删除。
- 消失过滤在逻辑上每一次 miss 的完整可视化。
- L2/L3 语义区域真实在线聚类历史。

如果未来要做真正完整的在线回放，需要在主流程中加入 `event_log.jsonl`，每帧记录：

- raw_yolo_detections
- sam_masks_summary
- depth_gate_decisions
- object3d_observation_created
- object3d_track_created / updated / merged / removed
- waypoint_created
- topology_created / matched
- semantic_region_updated
- nav_graph_edge_added

然后 Rerun 直接读取 event log，而不是从最终结果反推。

## 29. 为什么一开始会显示高层节点

你曾经问：为什么最开始 Rerun 就显示 `L2/L3` 和 `node_0 wp=34 obj=2`？

原因是当前 Rerun 使用最终结果里的 topology 和 semantic hierarchy 反推显示。第 0 帧已经属于 `topology:node_0`，而 `node_0` 在最终图中属于某个 L2/L3 区域，所以脚本会把它的父层级一起显示出来。

另外标签中的 `wp=34 obj=2` 是最终统计，不是第 0 帧已经有 34 个 waypoint。也就是说：

```text
node_0 在第 0 帧出现，但 node_0 的标签使用最终统计。
```

这属于最终结果倒推回放的固有限制。如果要完全在线一致，标签应该显示当前累计统计，而不是最终统计；这可以继续优化，但当前你已经接受它作为最终结果回放版的一部分。

## 30. 为什么取消一开始显示完整绿色轨迹

Rerun 初版中，我们把完整 cuVSLAM 轨迹作为静态参考画到了 3D 视图里：

```python
rr.log("world/cuvslam_trajectory", ..., static=True)
```

这导致第 0 帧一打开就能看到完整绿色轨迹。你指出这不合理，因为在线回放不应该一开始就知道完整未来轨迹。

我们随后修改了 `export_rerun_semantic_replay.py`，把完整静态轨迹改成默认不显示，只在显式传参时显示：

```bash
--show-full-cuvslam-trajectory
```

现在默认只显示：

```python
world/trajectory_so_far
```

也就是随帧增长的当前轨迹。这一点是最终满意版本的重要改动。

## 31. 为什么点云半径调小

在 Rerun 中，点云显示太粗会遮挡物体框和拓扑节点。你指出点云点需要小一点，半径小一半。我们于是把点云半径从：

```text
主点: 0.055
辅助白点: 0.018
```

改为：

```text
主点: 0.0275
辅助白点: 0.009
```

最终全量点云版使用：

```bash
--cloud-radius 0.0275
--cloud-secondary-radius 0.009
```

这样点云仍然清楚，但不会像之前那样糊成一片。

## 32. 为什么背景改成黑色

默认 Rerun 3D 视图背景下，青色点云、白色文字、黄色拓扑标签有时对比度不足。你希望背景更清楚，最好黑色。

我们在 Rerun blueprint 中设置：

```python
rrb.Spatial3DView(
    origin="world",
    background=rrb.Background(color=[0, 0, 0, 255], kind=rrb.BackgroundKind.SolidColor),
    line_grid=False,
)
```

并配合：

- 青色点云：`40,230,255`
- 白色辅助点：`255,255,255`
- topology/L2/L3 标签亮色显示。
- object label 默认隐藏，避免文字过多。
- topology 层级整体抬高，避免和地面点云混在一起。

最终视觉上比普通 HTML/Plotly 更适合检查“点云 + 物体 + 拓扑”的空间关系。

## 33. 最终 Rerun 版本命令

最终满意版本的生成命令为：

```bash
PYTHONPATH=/tmp/rerun_sdk:/tmp/rerun_sdk/rerun_sdk python /home/zyf/code_made/export_rerun_semantic_replay.py \
  --max-frames -1 \
  --cloud-points 0 \
  --image-width 640 \
  --include-removed \
  --grow-pointcloud \
  --max-growing-cloud-points 100000 \
  --show-hierarchy \
  --black-background \
  --object-label-mode none \
  --hierarchy-label-color-mode bright \
  --cloud-color 40,230,255 \
  --cloud-radius 0.0275 \
  --cloud-secondary-color 255,255,255 \
  --cloud-secondary-radius 0.009 \
  --topology-z 5.5 \
  --semantic-level-z-step 2.6 \
  --output /home/zyf/code_made/result/semantic_graphs_dataset_test3_foreground_filter_full/online_semantic_build_full_cloud_small_replay.rrd
```

打开命令为：

```bash
PYTHONPATH=/tmp/rerun_sdk:/tmp/rerun_sdk/rerun_sdk /tmp/rerun_sdk/bin/rerun \
  /home/zyf/code_made/result/semantic_graphs_dataset_test3_foreground_filter_full/online_semantic_build_full_cloud_small_replay.rrd
```

验证命令为：

```bash
PYTHONPATH=/tmp/rerun_sdk:/tmp/rerun_sdk/rerun_sdk /tmp/rerun_sdk/bin/rerun rrd verify \
  /home/zyf/code_made/result/semantic_graphs_dataset_test3_foreground_filter_full/online_semantic_build_full_cloud_small_replay.rrd
```

验证结果：

```text
1 file verified without error.
```

## 34. 最终 Rerun summary

最终 Rerun summary 文件：

```bash
/home/zyf/code_made/result/semantic_graphs_dataset_test3_foreground_filter_full/online_semantic_build_full_cloud_small_replay.summary.json
```

内容摘要：

```json
{
  "frames": 241,
  "cloud_points": 45421,
  "grow_pointcloud": true,
  "growing_cloud_points_final": 14986,
  "cloud_color": [40, 230, 255],
  "cloud_radius": 0.0275,
  "show_hierarchy": true,
  "object_label_mode": "none",
  "show_removed_labels": false,
  "hierarchy_label_color_mode": "bright",
  "topology_nodes": 4,
  "semantic_region_nodes": 3,
  "cuvslam_trajectory_points": 3600,
  "waypoints": 86,
  "object_nodes": 18,
  "removed_objects": 3
}
```

这就是当前最终可视化版本的参数和数据规模记录。

## 35. 结果应该如何理解

最终结果不是一个单一模型的输出，而是多层处理后的产物：

```text
原始数据集
  -> code_made 数据适配
  -> cuVSLAM 轨迹/点云
  -> YOLO 微调检测
  -> SAM mask
  -> 真实 depth 回投
  -> 前景点云过滤
  -> 3m 深度过滤
  -> Object3D tracking
  -> 稳定过滤
  -> 重叠过滤
  -> 运动过滤
  -> 消失回访过滤
  -> waypoint 采样
  -> nav graph 构建
  -> semantic hierarchy
  -> HTML / Rerun 可视化
```

因此如果最终 3DBox 不符合现实，需要判断问题发生在哪一层：

- RGB 检测框是否正确。
- SAM mask 是否覆盖主体。
- depth 是否与 RGB 对齐。
- depth 数值是否正确。
- 点云前景过滤是否过松或过严。
- 坐标轴是否正确。
- 相机位姿是否正确。
- Object3D track 是否错误合并。
- overlap filter 是否误删。
- disappearance filter 是否误删。
- Rerun 可视化是否只是最终结果倒推导致误解。

这也是我们之前一直逐层排查的原因。3DBox 位置异常不能只看最终框，要沿着链路往前查。

## 36. 当前版本最重要的技术修正

如果只列最关键的修正，可以概括为以下几项。

第一，使用训练后的 YOLO：

```bash
/home/zyf/Desktop/yolo_train_runs/test3_front/weights/best.pt
```

它替代通用 `yolov8n.pt`，提高了当前数据集场景内物体检测能力。

第二，使用 recorded depth：

```bash
/home/zyf/code_made/core_content/dataset_test3_cuvslam/depth
```

避免仅靠 DepthAnything+LiDAR fallback 造成深度估计误差。

第三，引入 YOLO -> SAM -> depth point cloud：

用 SAM mask 把 bbox 内背景点排除掉，减少 3DBox 被背景拉偏。

第四，修复坐标轴：

```python
[x, y, z] -> [x, z, -y]
```

避免左侧物体被投到右侧，或物体全部挤到轨迹中间。

第五，加入 3m 深度门限：

只让 3m 内物体进入 Object3D，减少远距离噪声。

第六，加入前景点云过滤：

对 mask 内过于发散的点做清洗，让 3DBox 更贴近主体。

第七，恢复完整 Object3D 后处理：

包括稳定过滤、重叠去重、运动过滤、消失回访过滤。

第八，保留 waypoint 距离/角度/拓扑变化采样：

把 241 帧压缩成 86 个 waypoint，使拓扑图更可读。

第九，做 Rerun 回放：

把 RGB、检测框、点云、轨迹、3DBox、waypoint、拓扑、语义层级放到同一个时间轴中查看。

第十，优化 Rerun 显示：

黑背景、点云半径减半、不提前显示完整未来轨迹、层级节点抬高、文字亮色。

## 37. 当前已知限制

当前系统已经能满足你这版回放和结果检查需求，但仍有几个边界需要明确。

第一，Rerun 回放不是完整真实事件日志。它是从最终结果反推的在线式展示，所以检测、过滤、删除的中间过程不完整。

第二，RGB 检测框来自 waypoint observations，而不是每帧 raw YOLO 全量检测。因此不是所有 241 帧都有 bbox overlay。

第三，增长点云来自 `replay_snapshots.json`，不是全部 45421 个 final landmarks 的生命周期。最终增长层约 14986 个点。

第四，Object3D 3DBox 仍然依赖深度和 mask 质量。如果 depth 边缘不准或 mask 包含背景，框仍可能偏大或偏移。

第五，当前运动过滤默认只针对 `person`。如果以后希望 `chair` 在某些场景中可移动，需要扩展类别策略，但当前你明确要求 chair/door 默认不能移动。

第六，语义 L2/L3 的出现时机是从最终结构推断，不是真实在线聚类事件。

第七，当前 `/home` 磁盘空间非常紧张，后续继续生成 Rerun 文件前最好清理旧版本。

## 38. 如果后续要进一步完善

如果下一阶段继续做，我建议优先做三件事。

第一，加入主流程 event log：

```bash
semantic_build_events.jsonl
```

每帧记录 raw YOLO、SAM mask、depth gate、Object3D track event、filter event、waypoint event、topology event。这样 Rerun 就能做真正完整在线回放。

第二，改进 3DBox 估计：

当前用清洗后的点云做 axis-aligned bbox。后续可以针对不同类别加入尺寸先验，例如 door 高宽比例、chair 高度范围、桌面高度范围；也可以引入 oriented bbox 或平面约束。

第三，改进点云时间信息：

如果需要全量 45421 个 cuVSLAM landmarks 都随时间出现，需要在 cuVSLAM 导出时记录每个 landmark 的首次观测帧、最后观测帧和观测次数，而不仅仅是 final PLY。

## 39. 一键查看当前满意版本

打开最终 Rerun：

```bash
PYTHONPATH=/tmp/rerun_sdk:/tmp/rerun_sdk/rerun_sdk /tmp/rerun_sdk/bin/rerun \
  /home/zyf/code_made/result/semantic_graphs_dataset_test3_foreground_filter_full/online_semantic_build_full_cloud_small_replay.rrd
```

查看 Object3D 全局框：

```bash
/home/zyf/code_made/result/semantic_graphs_dataset_test3_foreground_filter_full/object3d_global_map.html
```

查看 3D 导航图：

```bash
/home/zyf/code_made/result/semantic_graphs_dataset_test3_foreground_filter_full/nav_graph_visualization_3d.html
```

查看统计：

```bash
/home/zyf/code_made/result/semantic_graphs_dataset_test3_foreground_filter_full/nav_graph_stats.json
```

查看 Object3D 轨迹和过滤详情：

```bash
/home/zyf/code_made/result/semantic_graphs_dataset_test3_foreground_filter_full/object3d_tracking_summary.json
```

## 40. 总结

这次工作从一个原始多传感器数据集开始，最终形成了一条可以落地查看的语义拓扑建图链路。中间的关键不是某一个模型，而是把每层信息按正确坐标、正确时间、正确过滤逻辑连接起来。

我们先解决数据格式问题，让 dataset_test3 能被 code_made 主流程读取；再引入 cuVSLAM 位姿和点云，让 3D 世界有可靠坐标基础；再训练 YOLO，提高当前场景识别能力；再用 SAM 和真实 depth 把 2D 检测转成 3D 点云；再通过坐标轴修复解决左右和前后错位；再通过 3m 深度门限、前景点云过滤、稳定过滤、重叠过滤、运动过滤、消失回访过滤，让 Object3D 物体节点更可信；再通过 waypoint 采样和 semantic hierarchy 构建可导航的拓扑图；最后用 Rerun 把 RGB、检测、点云、轨迹、物体框、waypoint、拓扑和语义层级放在同一个时间轴里回放。

最终满意版本的核心产物是：

```bash
/home/zyf/code_made/result/semantic_graphs_dataset_test3_foreground_filter_full/online_semantic_build_full_cloud_small_replay.rrd
```

这版已经具备了当前需要的完整展示能力：黑色背景、全量 PLY 读取、小半径点云、增长式点云、增长式轨迹、RGB 检测、Object3D 物体框、waypoint、拓扑节点和 L2/L3 高层节点。它不是严格的在线事件日志，但已经是基于最终结果和原始数据构造出的、目前最适合检查空间语义建图效果的回放版本。

