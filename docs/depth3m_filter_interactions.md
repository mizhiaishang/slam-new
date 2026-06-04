# 3m深度过滤与Object3D全过滤交互说明

生成时间：2026-05-28

适用项目：`/home/zyf/code_made`

适用数据集：`/home/zyf/Desktop/dataset_test3`

当前主要结果目录参考：

- 一帧严格检查：`/home/zyf/code_made/result/semantic_graphs_dataset_test3_axisfix_strict_oneframe`
- 五帧严格检查：`/home/zyf/code_made/result/semantic_graphs_dataset_test3_axisfix_strict_5frames_check`
- 坐标轴修正版完整结果：`/home/zyf/code_made/result/semantic_graphs_dataset_test3_axisfix_full`

## 0. 先给结论

现在新增的“深度 3m 以内才作为物体节点候选”的过滤，发生在 Object3D 单帧观测生成阶段。它不是最后的导航图过滤，也不是 Object3D 轨迹稳定性过滤，而是在“YOLO 检测框 + SAM mask + 深度图回投为 3D 点云”之后，根据该物体 mask 内有效点云的中位深度做一次早期筛选。

因此它的作用可以理解为：

> 只允许离相机不超过 3m 的检测结果进入 Object3D 跟踪系统。

更准确一点说：

> 只允许 SAM mask 内清洗后的局部点云 `local_points[:, 2]` 的中位深度不超过 3m 的检测结果进入 Object3D 跟踪系统。

这里的 `local_points[:, 2]` 是相机局部坐标系下的前向深度，也就是物体沿相机视线方向的距离，不是世界坐标里的 z 高度。

这个过滤不会直接删除已经进入导航图的物体节点。它的影响是间接的：如果某一帧物体因为深度超过 3m 被丢掉，那么这一帧对 Object3D 来说就像“没看到这个物体”。后面的连续帧稳定过滤、运动过滤、消失过滤都会基于“Object3D 到底看到了哪些帧”继续判断。

我们已经做过一个一帧和五帧的小验证。

一帧严格检查结果：

- YOLO 原始检测看到 8 个目标。
- 经过 SAM + 深度点云 + 3m 深度约束后，单帧 payload 中保留 3 个检测。
- 3 个保留检测都是 chair，深度分别约为 `2.444m`、`1.966m`、`2.981m`。
- 最终 `object_nodes = 0`。
- 这个 `0` 不是 3m 过滤把物体全删了，而是因为严格模式下 `object3d_min_consecutive_frames = 2`，只跑一帧永远无法满足“连续两帧出现”。

五帧严格检查结果：

- `object3d_raw_object_count = 6`
- `object3d_stable_object_count = 4`
- `object3d_overlap_removed_object_count = 1`
- `object3d_motion_removed_object_count = 0`
- `object3d_disappearance_removed_object_count = 0`
- 导航图最终 `object_nodes = 2`

这说明 3m 过滤没有天然导致后续严格过滤完全清空。它确实会减少进入 Object3D 的候选，但在当前 dataset_test3 的前 5 帧上，仍然有稳定物体能通过后续过滤并进入导航图。

真正需要警惕的不是“3m 一定会把所有东西删光”，而是下面这个更细的交互风险：

> 如果一个真实静态物体先在 3m 内被确认，后面相机回访时从几何上应该还能看到它，但它此时因为深度超过 3m、深度噪声、mask 偏移、深度和 RGB 未对齐，导致没有进入 payload，那么消失过滤可能把它当成“应该看到但没看到”，从而删除。

也就是说，3m 过滤对“物体消失又出现”逻辑的影响主要体现在：它会改变“后续帧是否存在可匹配检测”。如果 3m 过滤挡掉了后续匹配，那么消失过滤可能误以为物体消失。

这份文档后面会把完整链路、每层过滤、相互作用和保护方案讲清楚。

## 1. 当前完整管线

当前项目里，从数据集回放到拓扑图构建，大体是这一条链路：

1. 读取数据集中的 RGB、点云、轨迹、深度。
2. 使用 YOLO 对 RGB 做 2D 目标检测。
3. 使用 SAM 根据 YOLO 框生成物体 mask。
4. 使用真实 depth 图，把 mask 内像素回投成相机局部 3D 点云。
5. 对点云做清洗、最少点数检查、mask 面积检查。
6. 执行 3m 深度过滤。
7. 将通过过滤的单帧 3D 观测交给 Object3D tracking engine。
8. Object3D 根据空间距离和类别，把跨帧观测合并成物体轨迹。
9. 导出 Object3D tracking summary。
10. 对 Object3D 轨迹做连续帧稳定过滤。
11. 对稳定轨迹做同类 3D 重叠去重过滤。
12. 对稳定轨迹做运动物体过滤。
13. 对稳定轨迹做消失后回访过滤。
14. 根据过滤后的稳定 Object3D 轨迹，反向过滤每帧 payload 中的 detections。
15. 对 waypoint 做距离、角度、首尾、拓扑变化采样。
16. 将保留下来的 waypoint 和 object observations 导入 nav graph。
17. 输出 `nav_graph_contents.json`、`nav_graph_stats.json`、`object3d_global_map.html`、`nav_graph_visualization_3d.html`、回放 HTML 等。

这条链路里，3m 深度过滤的位置很靠前。它不是图构建阶段的过滤，而是“单帧观测是否有资格进入 Object3D”的过滤。

用更朴素的话说：

- YOLO 说“这里可能有个 chair”。
- SAM 说“chair 大概是这块 mask”。
- depth 说“这块 mask 里的点在三维空间这些位置”。
- 3m 过滤说“如果这块 mask 对应的主体距离超过 3m，就先不要把它当作 Object3D 观测”。
- Object3D 说“我只跟踪那些已经通过前面检查的观测”。
- 后续过滤说“我只处理 Object3D 已经跟踪到的轨迹”。

因此，3m 过滤的上游是 YOLO、SAM、depth、点云清洗；下游是 Object3D tracking、稳定过滤、重叠过滤、运动过滤、消失过滤和导航图入图。

## 2. 3m 深度过滤在哪里实现

当前 3m 阈值是在 `/home/zyf/code_made/run_semantic_topomap.py` 初始化 Object3D engine 时写入的：

```python
settings = EngineSettings(
    voxel_size=0.02,
    dbscan_remove_noise=True,
    dbscan_eps=0.15,
    dbscan_min_points=10,
    min_points_threshold=18,
    mask_area_threshold=18,
    observation_max_depth_m=3.0,
    use_oriented_bbox=False,
)
```

关键字段是：

```python
observation_max_depth_m=3.0
```

真正执行过滤的位置在：

`/home/zyf/Desktop/3d/object3d_engine/core/observation_estimator.py`

核心逻辑是：

```python
if self.settings.observation_max_depth_m is not None:
    median_depth_m = float(np.median(local_points[:, 2]))
    if not np.isfinite(median_depth_m) or median_depth_m > float(self.settings.observation_max_depth_m):
        return None
```

这段代码的含义非常直接：

- 如果设置了 `observation_max_depth_m`，就启用最大观测深度过滤。
- 从 mask 对应的局部点云 `local_points` 里取第 3 列，也就是前向深度。
- 计算这些点的中位数深度。
- 如果中位数不是有效数，删除这次观测。
- 如果中位数大于 3.0m，删除这次观测。
- 如果中位数小于等于 3.0m，允许继续生成 ObjectObservation3D。

这里用的是中位数，不是平均值。中位数的好处是对少量深度噪声更稳。例如一个椅子的 mask 内大多数点在 2.5m，少数点因为边缘漏到了 8m，那么中位数通常仍然接近 2.5m，不会被少量远点带偏。

但中位数也有副作用。如果 mask 里包含了较大面积背景，比如门框、墙面、玻璃反射、桌子后面的区域，那么中位数可能被背景主导。此时即使目标前景局部在 2m，也可能因为 mask 内大部分有效点在 4m，被 3m 过滤掉。

## 3. 3m 深度过滤使用的深度来源

当前 dataset_test3 的 depth 来源是导出的真实深度 PNG，而不是 DepthAnything 的估计深度。

项目里使用的是：

`/home/zyf/code_made/core_content/dataset_test3_cuvslam/depth`

这个路径当前指向：

`/home/zyf/Desktop/dataset_test3/depth`

在 `run_semantic_topomap.py` 中，加载 recorded depth 的逻辑大体是：

- 如果配置 `prefer_recorded_depth=True`
- 并且数据集根目录下存在 `depth/*.png`
- 则优先读取这些深度 PNG
- 如果深度 PNG 是整数类型，则除以 1000 转成米
- 如果深度 PNG 已经是 float，则直接作为米使用

相关逻辑在：

```python
if np.issubdtype(depth_raw.dtype, np.integer):
    depth_m = depth_raw.astype(np.float32) / 1000.0
else:
    depth_m = depth_raw.astype(np.float32)
```

因此，3m 阈值是在“米”这个单位上判断的，不是毫米，也不是像素值。

一个很重要的点是：3m 过滤不是直接看 bbox 中心点的深度，而是先由 SAM 得到 mask，再把 mask 内有效深度点回投为点云，然后看这些点云的中位前向深度。

所以它和 YOLO bbox 的关系是间接的：

- YOLO bbox 决定 SAM 的提示范围。
- SAM mask 决定哪些像素参与深度回投。
- depth 图决定这些像素对应的三维点。
- 点云清洗决定最终哪些点参与中位深度统计。
- 中位深度决定是否超过 3m。

如果 RGB、depth、bbox、mask 没有对齐，那么 3m 过滤会直接受影响。比如 bbox 在门上，但对应深度图位置偏到了墙、玻璃、地面或远处背景，那么生成的点云就会错，3m 判断也会错。

## 4. 3m 深度过滤前还有哪些隐性过滤

在 3m 过滤之前，其实已经有几层过滤了。它们不一定被用户感知为“过滤”，但它们都会影响后续是否有物体。

### 4.1 YOLO 检测过滤

YOLO 结果进入 Object3D 前，会通过 adapter 做基础筛选：

```python
raw_detections = adapter.build_detections(
    result,
    prefer_masks=False,
    min_confidence=0.25,
    min_bbox_area=48.0,
)
```

含义是：

- 置信度低于 0.25 的检测不进入 Object3D。
- bbox 面积太小，小于 48 像素面积的检测不进入 Object3D。

这一层的目标是排除非常弱、非常小、极可能是误检的 2D 检测。

对后续的影响是：

- 如果 YOLO 没检出来，后面所有 Object3D 逻辑都不会知道这个物体存在。
- 如果 YOLO 检出来但置信度低于 0.25，也不会进入 Object3D。
- 如果物体很远，bbox 很小，可能会被 `min_bbox_area` 删掉。

所以当我们说“3m 过滤删掉了物体”时，必须先确认它是否已经通过 YOLO adapter。否则可能根本不是 3m 的问题。

### 4.2 SAM mask 生成过滤

YOLO bbox 会交给 SAM 生成 mask。如果 SAM 没能生成有效 mask，或者 mask 质量太差，后续就没有点云可以回投。

当前使用的是：

```python
self.object3d_mask_provider = SamMaskProvider(SamPredictor(sam))
```

也就是根据 YOLO 2D 框调用 SAM 做 mask。

这里的核心风险是：

- YOLO 框不准，SAM mask 就可能不准。
- YOLO 框覆盖多个物体，SAM mask 可能包含错误区域。
- 物体遮挡严重，SAM mask 可能只抠到一小块。
- 门、玻璃、椅背、桌面这类结构复杂物体，mask 边缘容易混入背景。

这些问题都会改变 mask 内点云的深度分布，从而影响 3m 中位深度。

### 4.3 mask 面积过滤

在 Object3D 的 `ObservationEstimator` 里，首先会检查：

```python
if detection.mask_area < self.settings.mask_area_threshold:
    return None
```

当前设置：

```python
mask_area_threshold=18
```

也就是说，mask 面积太小的检测会被删除。

这个阈值比较低，主要是防止极小碎片进入点云计算。一般来说它不是最容易导致大面积物体消失的原因，但对远距离小物体会有影响。

### 4.4 mask 置信度过滤

还有一层：

```python
if detection.confidence < self.settings.mask_conf_threshold:
    return None
```

这个置信度通常来自检测结果。它和 YOLO adapter 的 `min_confidence=0.25` 有重叠，但属于 Object3D 内部再次检查。

### 4.5 最少点数过滤

回投点云后，会检查点数：

```python
local_points, colors = self.pointcloud_service.project_mask_to_local_points(frame, detection)
if len(local_points) < max(self.settings.min_points_threshold, 1):
    return None
```

当前：

```python
min_points_threshold=18
```

清洗之后还会再检查一次：

```python
local_points, colors = self.pointcloud_service.clean(local_points, colors)
if len(local_points) < max(self.settings.min_points_threshold, 1):
    return None
```

这意味着：

- mask 内有效深度点太少，会删除。
- 清洗后剩余点太少，会删除。
- 如果 depth 图某个区域大量无效值、0 值、NaN、空洞，会导致点数不足。

这个过滤和 3m 过滤的关系很强。只有通过点数过滤的点云，才会计算中位深度。也就是说，3m 判断不是对所有 mask 像素做的，而是对“有效且清洗后的点云”做的。

### 4.6 DBSCAN 噪声清洗

当前设置里：

```python
dbscan_remove_noise=True
dbscan_eps=0.15
dbscan_min_points=10
```

这说明点云会做 DBSCAN 噪声移除。

DBSCAN 的直觉是：保留局部密集的点群，删除离群点。

它的好处是：

- 删除背景散点。
- 删除深度飞点。
- 让 3D bbox 更稳定。

它的风险是：

- 物体点云本身稀疏时，可能被当成噪声。
- 细长物体、边缘物体、部分遮挡物体，可能点群被切碎。
- 如果 mask 混入背景，DBSCAN 可能保留最大的背景簇，而不是前景物体簇。

这会间接影响 3m 过滤，因为 3m 用的是清洗后的 `local_points`。

## 5. 3m 深度过滤本身的第一性原理解释

从第一性原理看，3D 物体节点要成立，至少需要三个东西：

- RGB 里有这个物体的像素证据。
- 深度图里这些像素能对应到合理三维点。
- 相机位姿能把局部三维点放到全局坐标系。

YOLO 提供的是“物体类别和 2D 大概位置”。

SAM 提供的是“这个物体大概占哪些像素”。

Depth 提供的是“这些像素离相机多远”。

Camera pose 提供的是“相机在世界里在哪里、朝哪边”。

Object3D 节点本质上是：

> 把一个 2D 语义检测，通过深度和位姿，变成一个世界坐标系里的 3D 语义物体。

这里最容易出错的是深度和 mask。因为只要 mask 里混入背景，或者深度和 RGB 不对齐，回投出来的点云就不是那个物体，而是别的东西。

新增 3m 过滤的第一性目的不是“让图更漂亮”，而是：

> 限制只把近距离、深度更可信、mask 回投误差更小的物体作为 Object3D 观测。

近距离物体通常有几个优点：

- bbox 更大。
- SAM mask 更容易贴合。
- depth 相对更可靠。
- 点云密度更高。
- 位姿误差带来的全局坐标偏移相对更小。

远距离物体的问题是：

- bbox 小，类别容易误识别。
- mask 容易覆盖背景。
- depth 噪声和空洞更多。
- 一点点像素偏移会导致较大三维误差。
- 远处多个物体容易在 3D 上被糊在一起。

所以 3m 过滤是一个“提高 Object3D 节点可信度”的早期准入门槛。

但它的代价是：

- 远处物体不会进入 Object3D。
- 物体从远到近过程中，只有进入 3m 后才会被记录。
- 物体离开 3m 后，后续帧可能不再提供匹配观测。
- 如果消失过滤认为后续帧“应该能看到”，而 3m 层又挡掉了检测，就有误删风险。

## 6. Object3D 跟踪合并的作用

通过 3m 过滤后，每个检测会成为一个 `ObjectObservation3D`。这个 observation 不是最终物体节点，而是“某一帧看到的某个物体候选”。

Object3D tracking engine 会把多个帧里的 observation 合并成一个 object track。

当前跟踪相关设置包括：

```python
settings.spatial_similarity_type = SpatialSimilarityType.CENTER_DISTANCE
settings.max_assignment_distance = 1.3
settings.match_threshold = 0.35
settings.postprocess_interval = -1
```

直觉上：

- 主要根据 3D 中心距离做跨帧关联。
- 如果两个观测中心足够近，并且类别/相似性满足要求，就可能归到同一个 object track。
- 最大分配距离是 1.3m。

跟踪合并的目标是：

> 把“每帧看到的 chair”合并成“同一个 chair 物体轨迹”。

它依赖前面生成的 3D bbox、centroid、class_name 和 frame_id。

3m 过滤对 tracking 的影响很直接：

- 如果某帧物体超过 3m，该帧不会产生 observation。
- Object3D track 的 `frame_ids` 会少一帧。
- 如果中间缺帧太多，连续帧稳定过滤可能失败。
- 如果后续重新进入 3m，可能被合并回原 track，也可能成为新 track，取决于空间距离、类别和 tracking 状态。

这里有一个关键点：3m 过滤发生在 tracking 之前，所以 Object3D tracking engine 不知道“这个物体其实被 YOLO 看到了，但因为超过 3m 被拒绝”。对 tracking 来说，这一帧就是没有这个物体。

## 7. 连续帧稳定过滤

连续帧稳定过滤在导出 tracking summary 时执行。

当前默认：

```python
object3d_min_consecutive_frames = 2
```

实现位置：

`FrameSemanticPayloadBuilder._filter_tracking_objects_by_consecutive_frames`

核心逻辑：

```python
longest_run = cls._longest_consecutive_frame_run(list(obj.frame_ids))
if longest_run >= min_consecutive_frames:
    kept.append(obj)
else:
    removed.append(removal_payload)
```

朴素解释：

- 如果一个 object track 至少连续 2 帧出现，就认为它有基本稳定性。
- 如果只出现 1 帧，或者出现多帧但不连续，比如第 0 帧和第 3 帧出现，中间断了，也可能不满足。

这个过滤的目标是：

> 防止单帧误检直接变成地图物体。

它和 3m 过滤的交互非常重要。

假设有一个门：

- 第 0 帧深度 2.9m，通过 3m。
- 第 1 帧深度 3.1m，被 3m 删除。
- 第 2 帧深度 2.8m，通过 3m。

对于人来说，这个门一直在。但对于 Object3D 来说，它可能出现在第 0 和第 2 帧，中间第 1 帧断了。如果 tracking 的 frame_ids 是 `[0, 2]`，最长连续 run 只有 1，就无法通过 `min_consecutive_frames=2`。

所以 3m 过滤会通过“制造缺帧”影响连续帧稳定过滤。

当前一帧严格检查里最终没有物体，就是因为这个逻辑。单帧最多只能得到最长连续 run = 1，而阈值是 2，因此所有 raw objects 都会在稳定过滤阶段被删除。这是预期行为，不是异常。

五帧严格检查中，仍然有 4 个稳定物体，说明在这 5 帧里至少有 4 个 object track 满足连续 2 帧出现。

## 8. 同类 3D 重叠去重过滤

连续帧稳定过滤之后，会执行同类 3D 重叠过滤。

当前默认：

```python
object3d_overlap_filter_enabled = True
object3d_overlap_iou_threshold = 0.05
object3d_overlap_min_ratio_threshold = 0.35
```

实现位置：

`FrameSemanticPayloadBuilder._filter_tracking_objects_by_class_volume_overlap`

核心逻辑：

- 先按对象质量排序。
- 遍历候选 object track。
- 如果和已经保留的同类别物体 3D bbox 重叠过高，就删除当前对象。
- 重叠判断条件是：
  - 3D IoU 大于等于 0.05，或
  - intersection / min_volume 大于等于 0.35。

这个过滤的目标是：

> 防止同一个真实物体因为 tracking 分裂，生成多个重复节点。

例如同一把椅子可能被分成两个 track。它们类别都是 chair，3D bbox 高度重叠，那么质量较低的一个会被删掉。

它和 3m 过滤的交互主要体现在点云质量上。

如果 3m 过滤留下的都是近距离点，bbox 通常更稳定，重叠去重更容易正确。

如果 3m 阈值附近有大量断续观测，可能出现：

- 同一物体被拆成两个 track。
- 两个 track 的 bbox 位置有偏移。
- 如果偏移小，重叠过滤能合并掉重复。
- 如果偏移大，重叠过滤可能认为它们是两个不同物体。

另外，如果 mask 混入背景导致 3D bbox 很大，即使是不同物体，也可能因为 bbox 重叠而误删。3m 过滤对这种情况既可能有帮助，也可能无能为力。帮助在于它删掉远处更不可靠的观测；无能为力在于近处 mask 如果本身错了，bbox 仍然会错。

五帧严格检查里：

```text
object3d_overlap_removed_object_count = 1
```

说明确实有一个同类重叠轨迹被去重删掉。

## 9. 运动物体过滤

重叠过滤之后，会执行运动过滤。

当前默认：

```python
object3d_motion_filter_enabled = True
object3d_motion_static_max_center_span_m = 1.0
object3d_motion_static_max_median_step_m = 0.35
object3d_motion_static_max_single_step_m = 1.2
```

实现位置：

- `_analyze_tracking_object_motion`
- `_build_object_motion_metrics`
- `_filter_tracking_objects_by_motion_state`

朴素解释：

系统会看同一个 object track 在连续帧里的 3D bbox center 是否移动太大。如果移动范围超过阈值，并且移动模式不像静态物体的测量抖动，就把它判为 moving，然后过滤掉。

当前判断 moving 的几个条件大致是：

- 最大中心跨度 `max_center_span_m` 超过 1.0m，并且净位移也足够大。
- 或者最大中心跨度超过 1.0m，并且连续帧中心移动的中位步长超过 0.35m。
- 或者最大中心跨度超过 1.0m，并且单步最大移动超过 1.2m。

这个过滤的目标是：

> 地图里的物体节点应当主要表示静态环境物体，而不是移动中的人、车、临时物体、误跟踪漂移目标。

它和 3m 过滤的交互也比较明显。

3m 过滤可能减少远距离噪声，使得物体中心更稳定，从而降低误判 moving 的概率。

但 3m 过滤也可能造成观测不连续。运动过滤只分析连续 run。如果一个物体因为深度阈值断断续续出现，运动分析可能只看到短 run。短 run 不足时，代码会倾向于认为它是 static：

```python
if len(valid_entries) < 2:
    return {
        "motion_state": "static",
        "motion_reason": "insufficient_motion_history_static_assumed",
    }
```

这意味着：

- 3m 过滤不会直接让运动过滤更严格。
- 反而在观测太少时，运动过滤可能没有足够证据判断移动，从而默认 static。
- 但是这些观测太少的对象可能已经在连续帧稳定过滤阶段被删掉。

所以，运动过滤主要处理“已经有足够稳定观测，但中心移动异常”的物体。

五帧严格检查中：

```text
object3d_motion_removed_object_count = 0
```

说明前 5 帧里没有物体因为运动过滤被删。

### 9.1 针对运动物体判别，3m 过滤会不会导致误判

这是一个更精确的问题：

> 3m 过滤是否会导致运动物体过滤把运动物体误判成静态，或者把静态物体误判成运动？

答案是：会有影响，但主要不是“把静态误判成运动”，而是更容易造成“运动证据不足”，从而让运动物体没有被运动过滤识别出来。

原因在于当前运动过滤依赖的是同一个 Object3D track 在连续帧里的 `bbox_3d_center` 历史。它不是看 YOLO 原始 bbox 在图像里动没动，也不是看 RGB 光流，而是看已经通过 Object3D 观测链路后的 3D 中心点序列。

因此，3m 过滤对运动判别的影响路径是：

1. 物体在某帧被 YOLO/SAM 检测到。
2. 如果该帧 mask 点云中位深度超过 3m，这次观测被删除。
3. 被删除的观测不会进入 Object3D track history。
4. 运动过滤只能看到剩下那些 3m 内的观测。
5. 如果剩下观测太少、太短、太不连续，运动过滤没有足够证据判断它在移动。
6. 当前代码在运动历史不足时默认 `static`。

所以最典型的误判不是：

> 3m 把一个静态物体变成 moving。

而是：

> 一个真实移动物体，因为只有少数近距离帧通过 3m，运动历史被截断，最后没有被判成 moving。

举个直观例子：

假设一个人或椅子被推动，从 2.5m 移动到 4m。

- 第 0 帧：2.5m，通过 3m。
- 第 1 帧：2.9m，通过 3m。
- 第 2 帧：3.2m，被 3m 删除。
- 第 3 帧：3.8m，被 3m 删除。
- 第 4 帧：4.0m，被 3m 删除。

运动过滤实际看到的只有第 0、1 帧。它最多知道这个物体在短短两帧里中心移动了一点，但看不到后面真正远离的过程。如果这两帧的中心跨度没有超过运动阈值，就会被认为 static，或者在稳定过滤阶段被保留为静态候选。

再举一个反方向例子：

一个物体从 4m 向相机移动到 2m。

- 4m、3.5m 时都被 3m 删除。
- 2.9m 开始进入 Object3D。
- 2.5m 继续进入 Object3D。

运动过滤只看到“进入 3m 后”的后半段运动。如果进入 3m 后它已经基本停下，那么系统会认为它是静态物体。这在地图构建里未必完全错误，因为系统只对近距离可信范围内的状态负责；但如果业务目标是识别“曾经移动过的物体”，那就会漏判。

3m 过滤也可能造成另一种偏差：

> 只保留近距离局部轨迹，会低估物体实际运动跨度。

当前 moving 判断有一个关键门槛：`max_center_span_m > 1.0m`。如果完整轨迹从 2m 到 5m 跨了 3m，但 3m 过滤只留下 2m 到 2.8m 的部分，那么中心跨度可能小于 1m，最终不会触发 moving。

还有一种比较少见但存在的情况是静态物体被误判为 moving。它通常不是 3m 过滤单独造成，而是 3m 边界附近的深度噪声、mask 抖动、点云错位造成：

- 同一个静态物体在 2.9m 附近。
- 某些帧 mask 抠到前景，中心在正确位置。
- 某些帧 mask 混入背景或深度错位，中心跳到另一处。
- 通过 3m 的观测中心出现大幅跳变。
- 运动过滤看到 `bbox_3d_center` 大幅变化，可能判为 moving。

这种情况下，真正的问题不是“3m 使静态变运动”，而是 3m 阈值附近的观测质量不稳定，让 Object3D track history 的中心点不可信。

所以针对运动过滤，结论可以写得很明确：

- 3m 过滤会减少运动过滤可用的时间证据。
- 3m 过滤可能让运动物体被误认为静态，尤其是运动发生在 3m 外或跨越 3m 边界时。
- 3m 过滤一般不会直接让静态物体变成运动，但如果深度/RGB/mask 在 3m 边界附近不稳定，可能通过中心点跳变间接造成误判。
- 如果业务要可靠过滤运动物体，仅靠 3m 内的 Object3D center history 不够，应该额外保留“被 3m 拒绝但 YOLO/SAM 看见”的诊断证据，或者运动过滤使用 2D bbox/track 辅助。

### 9.2 针对运动过滤的保护建议

如果我们坚持“只有 3m 内的物体可以入图”，但又不想让运动过滤失效，可以这样保护：

1. 运动过滤不要只看通过 3m 的 3D observations，也记录被 3m 拒绝的同类 YOLO/SAM 观测作为辅助证据。
2. 对每个 Object3D track 记录 `depth_gate_miss_count`，也就是本来有同类/近邻 2D 检测但因 3m 没进来的次数。
3. 如果一个 track 在 3m 内短暂出现，随后多次在 3m 外继续被 YOLO 看到，不能简单当作稳定静态物体，应该标记为 `motion_unknown` 或 `insufficient_motion_evidence`。
4. 当前 `insufficient_motion_history_static_assumed` 更适合改成三态：`static`、`moving`、`unknown`。证据不足时不要强行当 static。
5. 如果最终地图只保留静态物体，可以让 `unknown` 默认不入图，或者入图但标注低置信度。

工程上最小改动是：

> 保留现有 3m 过滤，但把运动历史不足的 object 标注为 `motion_unknown`，不要直接当 `static`。

这样不会让 3m 外运动物体悄悄混入“静态地图”。

## 10. 消失后回访过滤

这是用户特别关心的部分：物体消失又出现时，过滤怎么判断。

当前默认：

```python
object3d_disappearance_filter_enabled = True
object3d_disappearance_max_observation_distance_m = 5.0
object3d_disappearance_position_tolerance_m = 1.0
object3d_disappearance_match_distance_m = 1.0
object3d_disappearance_min_visible_misses = 1
object3d_disappearance_fov_margin_deg = 8.0
```

实现入口：

`SemanticTopomapRunner._apply_object3d_disappearance_filter`

核心评估函数：

`SemanticTopomapRunner._evaluate_object3d_disappearance`

### 10.1 消失过滤解决什么问题

假设系统前面认为有一个物体，比如 chair。后来机器人绕了一圈，又回到一个应该能看到这把 chair 的位置。如果这时 chair 没有被检测到，那么可能有几种情况：

- chair 本来就是误检。
- chair 是临时移动物体，后来被移走了。
- chair 被遮挡了。
- YOLO/SAM/depth 失败了。
- 位姿或坐标投影错了。
- 3m 过滤把它挡掉了。

消失过滤想做的是：

> 如果后续确实有机会重新看到一个稳定物体，但系统没有看到，那就把这个物体从最终地图里删除或标记为消失。

这可以减少假阳性物体节点。

### 10.2 什么叫“后续有机会看到”

代码不会对所有后续帧都做消失检查，而是先判断这一帧是不是一个有效的 visibility opportunity。

实现位置：

`_evaluate_object_visibility_opportunity`

它会检查：

1. 是否有相机位置和旋转。
2. 物体在当前相机坐标系中是否在前方。
3. 相机到物体距离是否小于 `max_observation_distance_m`，当前是 5m。
4. 物体是否在安全视场角内。
5. 当前相机位置是否接近过去观察过该物体的位置，距离阈值是 `position_tolerance_m=1.0m`。

只有这些条件都满足，才认为：

> 这一帧从几何上应该有机会看到这个物体。

这一步很重要，因为如果机器人走到了完全不同的位置、背对物体、或者离物体太远，没看到不能说明物体消失。

### 10.3 什么叫“重新匹配到”

如果一帧是有效 visibility opportunity，系统会调用：

`_find_matching_detection`

匹配逻辑大致是：

- 优先看当前 payload detection 的 `object3d_track_id` 是否和 stable object 的 object_id 一致。
- 如果 track_id 对上，认为匹配成功。
- 如果 track_id 没对上，就看类别是否一致。
- 类别一致后，再看 detection center 到 stable object center 的 3D 距离。
- 距离小于等于 `object3d_disappearance_match_distance_m=1.0m`，认为空间匹配成功。

如果匹配成功：

- `matched_revisit_count += 1`
- `consecutive_visible_miss_count = 0`
- 更新 last confirmed frame。

如果没有匹配成功：

- `total_visible_miss_count += 1`
- `consecutive_visible_miss_count += 1`
- 如果连续 miss 数达到 `min_visible_misses`，当前默认是 1，就删除该 stable object。

### 10.4 “物体消失又出现”时会发生什么

假设一个物体在前面已经成为 stable object。后面它短暂没被检测到，然后又被检测到。

如果没检测到的那几帧不是 visibility opportunity，比如太远、视角不对、不在安全视场、相机没有回到类似观察位姿，那么不会删。

如果没检测到的一帧是 visibility opportunity，而 `min_visible_misses=1`，那系统会立即删除。后面即使又出现，也可能已经被标记为 `removed_after_disappearance_filter`。

如果后续帧在删除前重新匹配到了：

- miss 计数会清零。
- 物体保持 active。

所以当前消失过滤是比较严格的，因为 `min_visible_misses=1`。只要有一次“应该能看到但没看到”，就可能删除。

### 10.5 3m 过滤如何影响消失过滤

这是这份文档最重要的部分。

消失过滤判断“有没有重新匹配到”，看的不是 YOLO 原始输出，而是 payload 中已经经过 Object3D 观测生成的 detections。

而 3m 过滤发生在 payload detection 生成之前。

因此，如果某个真实物体在后续回访帧里：

- YOLO 检到了。
- SAM 也出了 mask。
- 但 mask 内中位深度超过 3m。
- 或深度噪声导致中位深度超过 3m。
- 或 depth/RGB 未对齐导致中位深度超过 3m。

那么这个检测不会进入 payload。消失过滤在匹配时就看不到它。

如果同一帧又满足 visibility opportunity，系统会认为：

> 这个位置应该能看到旧物体，但当前没有匹配检测。

于是它可能把该物体删除。

这就是 3m 过滤和消失过滤之间最大的误删风险。

### 10.6 3m 和 5m 的阈值冲突

这里存在一个特别值得注意的阈值不一致：

- 3m 过滤：只允许 3m 内的观测进入 Object3D。
- 消失过滤：只要相机到物体距离不超过 5m，就认为有机会看到。

这意味着，在 3m 到 5m 的距离范围内，可能出现逻辑不一致：

> 消失过滤认为“应该能看到”，但 3m 过滤认为“这个检测太远，不允许作为观测”。

如果物体距离相机 4m：

- 对 3m 过滤：太远，删掉检测。
- 对消失过滤：还在 5m 范围内，可以作为可见机会。

这会制造潜在误删。

所以如果我们保留 3m 观测上限，建议把消失过滤的 `max_observation_distance_m` 也与 3m 对齐，或者在消失过滤里增加一个条件：只有当预测物体深度也在 3m 内时，才把这一帧算作 visible miss。

这是目前最重要的保护建议之一。

### 10.7 针对消失后回访判别，3m 过滤会不会导致失效

这个问题也可以更精确地表述为：

> 物体之前出现过，后来回访时本应再次确认它还在，但 3m 过滤会不会把这次确认挡掉，从而让消失过滤误判？

答案是：会，这是 3m 过滤和消失回访过滤之间最需要保护的地方。

当前消失过滤的核心判断链路是：

1. 先有一个 stable object。
2. 找到这个物体最后一次被看到的帧。
3. 从最后一次看到之后的后续帧开始检查。
4. 如果某一帧从几何上应该能看到它，记为 visibility opportunity。
5. 在该帧 payload detections 里找同 track 或同类近距离匹配。
6. 找到匹配，说明回访确认成功。
7. 找不到匹配，记为 visible miss。
8. 当前 `min_visible_misses=1`，一次有效 miss 就可能删除。

这里的关键是第 5 步：消失过滤只在 payload detections 中找匹配。

而 payload detections 已经经过 3m 过滤。也就是说，如果回访帧里真实物体存在，但由于中位深度大于 3m，它不会出现在 payload detections 里。对消失过滤来说，这和“没有检测到物体”是一样的。

因此，3m 过滤确实可能导致“回访确认失效”。

典型误判路径如下：

1. 第 10 帧，门在 2.5m，进入 Object3D，并最终成为 stable object。
2. 第 50 帧，机器人回到附近，门仍然存在。
3. 从相机几何看，门在视场内，距离 4m，小于消失过滤的 5m 门槛。
4. 消失过滤认为第 50 帧是有效 visibility opportunity。
5. YOLO 也许检测到了门，SAM 也许出了 mask。
6. 但 3m 深度过滤认为中位深度 4m，删除这次 observation。
7. payload detections 中没有这个门。
8. 消失过滤匹配失败。
9. 因为 `min_visible_misses=1`，门被删除。

这个删除在人的直觉里是错的，因为门并没有消失，只是系统的 3m 观测门槛不允许它作为当前帧确认。

所以更准确的结论是：

> 只要消失过滤的“可见机会范围”大于 3m 观测范围，3m 过滤就可能让回访确认失效。

当前配置正好存在这个问题：

- Object3D 观测范围：3m。
- 消失过滤可见范围：5m。

因此，在 3m 到 5m 之间，系统内部存在语义冲突。

对 3m 过滤来说：

> 这个物体太远，我不信任它，不把它作为观测。

对消失过滤来说：

> 这个物体不远，我应该能看到它；如果没看到，就说明它可能消失。

这两个判断同时成立，就会造成误删。

### 10.8 “小时候回访过滤”的更准确保护方式

如果这里的“小时候回访过滤”指的是“物体消失后，后续小时刻/后续时间点回访确认”的过滤，那么最重要的保护是：

> 回访 miss 的判定边界必须和 3m 观测边界一致。

不要让一个被 3m 过滤主动拒绝的检测，又被消失过滤当成“应该看到但没看到”。

具体有三种保护方案。

第一种，最简单：

```text
object3d_disappearance_max_observation_distance_m = 3.0
```

这样只有 3m 内的回访帧才会被算作 visible opportunity。3m 外没看到，不算消失。

第二种，更精确：

在 `_evaluate_object_visibility_opportunity` 里增加 forward depth gate。也就是把 stable object center 投到当前相机坐标系后，如果前向深度 `forward_z > 3.0m`，就直接：

```text
eligible = False
reason = "outside_object3d_depth_gate"
```

这样和 observation estimator 的 `local_points[:, 2]` 逻辑更一致。

第三种，更稳：

保留 5m 的几何可见范围，但把 3m 到 5m 的 miss 只标为 `weak_miss` 或 `untrusted_miss`，不用于删除。只有 3m 内的 miss 才能增加 `consecutive_visible_miss_count`。

推荐第三种，因为它保留了诊断信息：

- 3m 到 5m 没看到，记录下来，但不删除。
- 3m 内没看到，才认为是强证据。
- 连续多个强 miss 后再删除。

如果不想改太多代码，短期最安全的是第一种：

```text
把 disappearance max observation distance 从 5m 改成 3m。
```

同时建议：

```text
object3d_disappearance_min_visible_misses = 2
```

这样即使 3m 内偶发一次 YOLO/SAM/depth 失败，也不会立即删除。

### 10.9 对这两个判别问题的最终判断

针对你的真实问题，可以总结为两句话。

对运动物体过滤：

> 3m 过滤可能导致运动判别证据不足，尤其会漏掉 3m 外或跨越 3m 边界的运动，因此更容易把运动物体误留为静态或 unknown，而不是更容易把静态物体误删为 moving。

对消失后回访过滤：

> 3m 过滤可能导致回访确认失效，尤其当消失过滤认为 5m 内都应该可见时，3m 到 5m 的真实物体会被 3m 挡在 payload 外，导致匹配失败并可能误删。

所以，3m 深度过滤需要重点保护的不是数量，而是判别语义：

- 运动过滤要区分 `static` 和 `unknown`，不要把证据不足当静态。
- 消失过滤要把 visible miss 限制在 3m 可信观测范围内，或者把 3m 外 miss 记为弱证据。

## 11. 导航图入图过滤

Object3D 后处理完成后，会用 stable object 反向过滤每帧 payload。

实现位置：

`_filter_payloads_for_nav_graph`

当前默认：

```python
use_filtered_object3d_for_nav_graph = True
```

含义：

> 最终进入 nav graph 的不是所有单帧 detections，而是必须能匹配到后处理后的 stable Object3D 对象。

核心过程：

- 对每一帧 payload。
- 收集这一帧 active 的 stable objects。
- 遍历 payload 中的 detections。
- 如果 detection 能匹配到 stable object，就保留。
- 如果匹配不到，就从导航图输入中删除。

这里会考虑消失过滤：

- 如果一个 object 被消失过滤删除，它在删除帧之前仍可作为候选。
- 到删除帧以及之后，就不再 active。

相关逻辑：

```python
if removal_frame_index is not None and int(frame_index) >= int(removal_frame_index):
    return False
```

这意味着被 disappearance filter 删除的物体，不会在删除后的帧继续进入导航图。

3m 过滤对这一层的影响是间接但很强的：

- 3m 过滤减少单帧 payload detections。
- Object3D 后处理只基于这些 detections 形成 stable objects。
- nav graph 又只接受能匹配 stable objects 的 detections。

所以 3m 过滤是“源头准入”，nav graph filter 是“最终入图准入”。中间所有过滤都会缩小候选集合。

五帧严格检查中：

```text
object3d_nav_graph_filter:
  enabled: true
  stable_object_count: 4
  input_detection_count: 13
  kept_detection_count: 10
  removed_detection_count: 3
```

这说明 13 个 payload detections 中，10 个能匹配后处理后的 stable object，3 个被导航图入图过滤删除。

## 12. 途径点采样和物体过滤的关系

途径点采样不是物体过滤，但它会影响最终图中有多少 waypoint 和 waypoint-object 边。

当前默认：

```python
waypoint_sampling_enabled = True
waypoint_min_distance_m = 0.8
waypoint_min_yaw_deg = 25.0
waypoint_keep_first_last = True
waypoint_keep_topology_change = True
```

实现位置：

`_sample_waypoint_payloads`

逻辑是：

- 第一帧保留。
- 最后一帧保留。
- 如果距离上一个保留 waypoint 超过 0.8m，保留。
- 如果 yaw 变化超过 25 度，保留。
- 如果拓扑标签变化，保留。

它和物体过滤的关系是：

- 物体 detections 先被过滤。
- payload 再被 waypoint sampling 选择。
- 如果某个物体只在被删除的 waypoint 帧里出现，最终图里可能看不到它的观测边。
- 但 stable Object3D 本身的形成发生在全量帧上，不是只在采样 waypoint 上形成。

也就是说，waypoint sampling 主要影响导航图表达密度，不直接影响 Object3D tracking 是否稳定。

不过最终 `object_nodes` 数量会受影响，因为 nav graph 是从采样后的 payload 建的。如果一个稳定物体没有在任何保留 waypoint 的 detections 里出现，它可能不会形成最终 object node，或者边数量较少。

这也是为什么“Object3D stable count”和“nav graph object_nodes”不一定相等。

五帧严格检查中：

- `object3d_stable_object_count = 4`
- 导航图 `object_nodes = 2`

这说明有 4 个稳定 Object3D 对象，但最终采样、入图、合并后导航图里是 2 个 object nodes。

## 13. 当前严格过滤配置总表

当前严格版本主要过滤配置如下：

| 过滤层 | 当前值 | 作用 |
|---|---:|---|
| YOLO min confidence | 0.25 | 太低置信度检测不进 Object3D |
| YOLO min bbox area | 48 px | 太小 bbox 不进 Object3D |
| SAM mask area threshold | 18 | mask 太小不进 Object3D |
| Object3D min points | 18 | 点云太少不进 Object3D |
| DBSCAN remove noise | True | 清洗点云噪声 |
| DBSCAN eps | 0.15 | 点云聚类半径 |
| DBSCAN min points | 10 | 点云簇最少点数 |
| observation max depth | 3.0m | mask 点云中位深度超过 3m 删除 |
| tracking max assignment distance | 1.3m | 跨帧合并最大空间距离 |
| min consecutive frames | 2 | 至少连续 2 帧出现才稳定 |
| overlap filter | True | 同类 3D 重叠去重 |
| overlap IoU threshold | 0.05 | 同类 bbox IoU 超过即冲突 |
| overlap min ratio threshold | 0.35 | 交集占小体积比例超过即冲突 |
| motion filter | True | 移动物体过滤 |
| motion max center span | 1.0m | 中心跨度过大可能 moving |
| motion max median step | 0.35m | 连续步长中位数过大可能 moving |
| motion max single step | 1.2m | 单步跳变过大可能 moving |
| disappearance filter | True | 回访可见但未匹配则删除 |
| disappearance max distance | 5.0m | 5m 内才算可见机会 |
| disappearance position tolerance | 1.0m | 需接近过去观察位置 |
| disappearance match distance | 1.0m | 同类中心 1m 内算匹配 |
| disappearance min visible misses | 1 | 一次有效 miss 即可删除 |
| disappearance fov margin | 8 deg | 视场边缘安全余量 |
| use filtered object3d for nav graph | True | 只让后处理稳定物体入图 |
| waypoint min distance | 0.8m | 途径点距离采样阈值 |
| waypoint min yaw | 25 deg | 途径点转角采样阈值 |

## 14. 典型场景分析

### 14.1 只跑一帧

现象：

- payload 中可能有 detections。
- Object3D raw objects 可能非 0。
- stable object count 是 0。
- nav graph object nodes 是 0。

原因：

- `min_consecutive_frames=2`
- 一帧无法满足连续 2 帧。

这不是 3m 过滤的问题。

如果只想看第一帧 3D 投影是否正确，应该看 step0 的 `info.json` 里的 detections，或者看单帧回放，而不是看最终 nav graph object_nodes。

### 14.2 物体一直在 3m 内，连续被检测

这是最理想的情况。

过程：

- YOLO 检到。
- SAM mask 正常。
- depth 正常。
- 中位深度小于等于 3m。
- 进入 Object3D。
- 连续 2 帧以上出现。
- 通过稳定过滤。
- 如果没有重复重叠、没有明显运动、没有消失 miss，进入导航图。

这种物体最可能稳定保留。

### 14.3 物体有时 2.9m，有时 3.1m

这是 3m 阈值边界最危险的情况。

过程：

- 2.9m 帧通过。
- 3.1m 帧删除。
- 2.8m 帧通过。

风险：

- 连续帧 run 被打断。
- tracking 可能分裂。
- 稳定过滤可能失败。
- 后续消失过滤可能认为“应该看到但没看到”。

保护建议：

- 将 3m 改成带滞回的逻辑，比如首次准入 3.0m，已存在 track 可延续到 3.3m。
- 或者连续帧稳定允许小缺口，比如 3 帧窗口内出现 2 帧。
- 或者消失过滤只在 3m 内计算 visible miss。

### 14.4 物体先近后远

假设物体前面在 2m 被确认，后面相机离远到 4m。

如果消失过滤的 visible opportunity 仍然使用 5m，那么 4m 处可能被认为“应该能看到”。

但 3m 过滤会阻止 4m 检测进入 payload。

结果：

- 如果那一帧几何上满足可见机会。
- 并且 payload 中没有匹配 detection。
- 当前 `min_visible_misses=1`。
- 那么物体可能被消失过滤删除。

这就是 3m 与 disappearance 5m 不一致带来的风险。

### 14.5 物体先远后近

假设门一开始在 5m，后来机器人靠近到 2m。

过程：

- 5m 时 YOLO 可能检测到，但 3m 过滤删除，不进入 Object3D。
- 2m 时通过 3m，开始进入 Object3D。
- 需要连续 2 帧后成为 stable。

这符合 3m 设计目标：只在近距离建立可靠节点。

代价是地图不会提前记录远处物体。

### 14.6 物体短暂遮挡

假设一个椅子被人挡了一帧。

如果遮挡帧不是有效可见机会，或者没有满足回访条件，不会被消失过滤删除。

如果遮挡帧满足有效可见机会，且当前 `min_visible_misses=1`，那么它可能被删除。

这里是否误删取决于：

- 可见机会判断是否合理。
- 物体中心和相机位姿是否准确。
- 视场角判断是否过宽。
- 遮挡是否被显式建模。

当前代码没有显式遮挡推理，因此 `min_visible_misses=1` 偏严格。

### 14.7 物体消失后又出现

如果物体被消失过滤删除后又出现，当前逻辑不会把之前删除的 stable object 重新激活成同一个生命周期。后续出现可能作为新的 track 或新的 object node 参与构建。

这对“动态变化地图”来说有一个语义问题：

- 它可以表达“旧对象被删除”。
- 也可以表达“新对象出现”。
- 但未必能表达“同一个对象消失后又回来”。

如果业务需要“物体可暂时消失、后面恢复”，需要把 disappearance filter 从 hard delete 改成状态机：

- active
- suspected_missing
- temporarily_missing
- removed
- reactivated

当前实现更接近 hard delete。

## 15. 3m 过滤可能造成的误删路径

下面列出最常见的误删路径。

### 15.1 3m 导致连续帧断裂

路径：

1. 真实物体存在。
2. 某些帧中位深度超过 3m。
3. 这些帧被 early return。
4. Object3D track 的 frame_ids 不连续。
5. `longest_consecutive_frame_run < 2`。
6. 稳定过滤删除。

表现：

- `raw_object_count` 可能有。
- `stable_object_count` 低。
- `removed_objects` 中出现 longest run 不足。

排查：

- 看 `object3d_tracking_summary.json` 的 removed_objects。
- 看每个 step 的 `info.json` detections 深度。
- 对比 YOLO 原始检测数量与 Object3D payload detections 数量。

### 15.2 3m 导致消失过滤误删

路径：

1. 物体前期在 3m 内，成为 stable object。
2. 后期相机回访，从几何上满足 visible opportunity。
3. 实际检测因为中位深度超过 3m，没有进入 payload。
4. `_find_matching_detection` 找不到匹配。
5. `consecutive_visible_miss_count >= 1`。
6. disappearance filter 删除该物体。

表现：

- `object3d_disappearance_removed_object_count > 0`
- removed object 的 `removal_stage = disappearance_filter`
- `disappearance_decision.missed_frames` 中有可见 miss。

排查：

- 看 missed frame 对应 RGB。
- 看 missed frame 是否 YOLO 原始检测到了该物体。
- 看该帧该物体是否因为 3m 被挡掉。
- 看物体中心到相机距离是否在 3m 到 5m 之间。

### 15.3 mask 背景污染导致中位深度超过 3m

路径：

1. 物体真实在 2m。
2. YOLO 框偏大。
3. SAM mask 包含背景。
4. 背景深度占 mask 有效点多数，比如 4m。
5. 中位深度超过 3m。
6. 观测被删除。

表现：

- RGB 看物体很近，但 `info.json` 中没有 detection。
- 或 detection 深度接近背景距离。

排查：

- 可视化 bbox + mask + depth。
- 在 bbox/mask 内统计深度分布，而不是只看最终 detection。
- 看 mask 点云是否贴在背景平面。

### 15.4 depth/RGB 未对齐导致误判

路径：

1. RGB 中 bbox 在物体上。
2. depth 图同一像素位置其实对应别的区域。
3. 回投点云错位。
4. 中位深度错误。
5. 3D bbox 错误或 3m 过滤误删。

表现：

- 物体全局坐标明显偏到轨迹中间、右侧、墙后。
- 深度可视化和 RGB 边缘不重合。
- 左侧物体被投到右侧。

排查：

- 用第一帧 bbox 中心和 mask 点云单独投影。
- 检查相机内参、深度分辨率、RGB 分辨率是否一致。
- 检查 depth 文件命名和 RGB 时间戳是否对应。
- 检查 optical 坐标到项目世界坐标的轴转换。

当前我们已经修过一个坐标轴问题：相机 optical 坐标 `x-right, y-down, z-forward` 映射到项目世界轴 `x-right, y-forward, z-up`。修正后门的位置从错误侧移动到了更合理的左前方。

## 16. 推荐的保护方案

### 16.1 让消失过滤距离与 3m 对齐

当前最直接的保护是：

```python
object3d_disappearance_max_observation_distance_m = 3.0
```

这样后续帧只有在 3m 内才被认为是有效可见机会。它和 `observation_max_depth_m=3.0` 保持一致，可以避免 3m 到 5m 区间的逻辑冲突。

优点：

- 简单。
- 风险小。
- 和当前 3m 设计一致。

缺点：

- 3m 外不会触发消失检查。
- 远距离回访无法证明物体消失。

如果我们的设计目标是“只信任 3m 内的物体观测”，那么这个缺点是可以接受的。

### 16.2 对已稳定物体使用宽松延续阈值

可以使用两个阈值：

- 新物体准入：3.0m。
- 已稳定物体延续：3.3m 或 3.5m。

直觉是：

> 一个从未见过的新物体，必须足够近才创建；但一个已经稳定的物体，后续稍微超过 3m 也可以用于确认它还存在。

这叫 hysteresis，滞回保护。

优点：

- 减少 3m 边界抖动。
- 避免 2.99m/3.01m 来回跳导致 track 断裂。

缺点：

- 需要区分 new observation 和 existing track confirmation。
- 当前 Object3D observation estimator 是早期过滤，不知道这个 detection 是否属于已稳定物体，因此实现要调整结构。

### 16.3 消失过滤里增加“3m 可观测性”条件

不要只用 camera-to-object distance <= 5m 判断可见机会，而是增加：

> 如果当前预测物体相机前向深度超过 3m，则不计为 visible miss。

这比简单把 max distance 改成 3m 更精确，因为 3m 过滤实际用的是相机前向深度 `local_points[:,2]`，而消失过滤现在用的是相机到物体的欧氏距离。

推荐逻辑：

```text
if forward_z > observation_max_depth_m + margin:
    eligible = False
    reason = "outside_object3d_depth_gate"
```

margin 可以是 0.2m 到 0.5m。

优点：

- 精确对齐 3m 深度门控。
- 避免 3m 过滤挡掉检测，但消失过滤仍记 miss。

缺点：

- 需要把 `observation_max_depth_m` 配置传到 disappearance filter。

### 16.4 提高消失过滤 miss 次数

当前：

```python
object3d_disappearance_min_visible_misses = 1
```

这很严格。可以改成 2 或 3。

含义：

- 不是一次可见 miss 就删除。
- 连续 2 次或 3 次应该看到但没看到，才删除。

优点：

- 抗 YOLO 偶发漏检。
- 抗 SAM 偶发失败。
- 抗 depth 瞬时错误。
- 抗 3m 边界抖动。

缺点：

- 真正消失的物体会晚一点删除。
- 地图会更保守。

如果业务更重视“不要误删静态物体”，建议至少设为 2。

### 16.5 连续帧稳定过滤允许小缺口

当前稳定过滤是最长连续 run。

可以改为窗口统计：

> 在 N 帧窗口内出现 K 帧就算稳定。

例如：

- 3 帧窗口出现 2 帧。
- 5 帧窗口出现 3 帧。

这样可以容忍：

- 3m 边界抖动。
- YOLO 单帧漏检。
- SAM 单帧失败。
- depth 单帧无效。

优点：

- 更符合真实传感器不稳定性。

缺点：

- 单帧误检如果在短窗口内重复，也可能更容易通过。
- 需要保留时间窗口和 frame index 逻辑。

### 16.6 增加过滤统计日志

这是最建议马上做的工程保护。

现在我们需要知道每帧：

- YOLO 原始检测数。
- YOLO adapter 后检测数。
- SAM mask 成功数。
- mask area 过滤删除数。
- min points 过滤删除数。
- DBSCAN 后点数不足删除数。
- 3m 深度过滤删除数。
- Object3D observation 保留数。
- tracking raw object 数。
- consecutive 删除数。
- overlap 删除数。
- motion 删除数。
- disappearance 删除数。
- nav graph filter 删除数。

这样一旦最终没有 object node，我们不用猜，直接看是哪一层删掉了。

推荐输出：

`object3d_filter_diagnostics.json`

结构可以按 frame 存：

```json
{
  "frame_index": 0,
  "timestamp": "654518187680",
  "yolo_raw_count": 8,
  "adapter_count": 8,
  "sam_mask_count": 8,
  "object3d_observation_count": 3,
  "depth_gate_removed_count": 5,
  "kept_depths_m": [2.444, 1.966, 2.981]
}
```

以及全局 summary：

```json
{
  "total_yolo_raw": 1064,
  "total_object3d_observations": 347,
  "total_depth_gate_removed": 123,
  "stable_objects": 41,
  "nav_graph_objects": 26
}
```

这个改动不会改变算法结果，但会极大提升排查效率。

## 17. 推荐当前项目采用的策略

结合现在的目标，我建议分成两个模式。

### 17.1 严格建图模式

用于最终生成比较干净的拓扑图。

建议：

- 保留 3m 深度过滤。
- 保留连续帧稳定过滤，`min_consecutive_frames=2`。
- 保留重叠过滤。
- 保留运动过滤。
- 保留消失过滤。
- 将消失过滤距离从 5m 改为 3m，或增加 depth gate eligible 判断。
- 将 `min_visible_misses` 从 1 改为 2。
- 保留 `use_filtered_object3d_for_nav_graph=True`。
- 保留 waypoint sampling。

这个模式下物体更少，但更可信。

### 17.2 调试回放模式

用于看“为什么某个物体没保留”。

建议：

- 保留 3m 深度过滤。
- 保留每帧 diagnostics。
- 可以临时关闭 nav graph stable filter，只用于可视化 raw observations。
- 回放中同时展示：
  - YOLO 原框。
  - SAM mask。
  - depth median。
  - 是否通过 3m。
  - object3d track id。
  - 是否 stable。
  - 是否被 overlap/motion/disappearance 删除。

这个模式不是最终建图，而是解释算法行为。

## 18. 如何判断最终没有物体是哪一层导致的

如果某次运行结果里 `object_nodes=0`，推荐按下面顺序排查。

第一步，看 `step*/info.json`：

- 如果每帧 detections 都是 0，说明问题在 YOLO/SAM/depth/3m 单帧观测阶段。
- 如果每帧 detections 有值，说明早期观测阶段不是全空。

第二步，看 `nav_graph_stats.json`：

- `object3d_raw_object_count`
- `object3d_stable_object_count`
- `object3d_overlap_removed_object_count`
- `object3d_motion_removed_object_count`
- `object3d_disappearance_removed_object_count`
- `object3d_nav_graph_filter`
- `stats.object_nodes`

第三步，判断：

- raw 有，stable 为 0：多半是连续帧稳定过滤。
- stable 有，overlap 删除多：可能重复 bbox 或 bbox 过大。
- motion 删除多：可能坐标抖动、深度错位、真实运动物体。
- disappearance 删除多：可能回访逻辑太严格，或者 3m 与 5m 阈值冲突。
- stable 有，nav graph object_nodes 少：可能 nav graph 入图过滤、waypoint sampling、object 合并导致。

第四步，对比一帧和多帧：

- 一帧没有 object_nodes 是正常的。
- 至少跑 2 到 5 帧才能判断稳定过滤是否合理。

## 19. 对当前一帧问题的明确回答

用户问：

> 先一帧吧，我们新增的深度3m是否需要保护一下，是否会因为这个深度3m的新增过滤导致我们后面的过滤将前面内容全部删除，导致没有物体保留？

当前回答：

一帧下最终没有物体，不是 3m 过滤直接导致，而是严格模式的连续帧稳定过滤导致。因为 `object3d_min_consecutive_frames=2`，只跑一帧时所有 object track 的最长连续出现帧数最多是 1，所以后处理 stable object 必然是 0。

实际一帧里，3m 过滤后仍保留了 3 个 chair detection，深度分别是 2.444m、1.966m、2.981m。这说明 3m 没有把单帧观测全删掉。

五帧验证里，严格过滤后仍然有 4 个 stable object，最终导航图中有 2 个 object node。这说明 3m 不会天然导致后面所有过滤把内容全部删除。

但从长期完整回放看，3m 确实需要和消失过滤做保护，特别是当前 disappearance max observation distance 是 5m，而 Object3D observation max depth 是 3m。这两个阈值不一致，会导致 3m 到 5m 范围内出现“消失过滤认为应该看到，但 3m 过滤不允许观测入场”的潜在误删。

所以建议保护，但保护点不是“关闭 3m”，而是：

1. 让消失过滤的可见距离与 3m 对齐。
2. 或让消失过滤只在 3m depth gate 内计 visible miss。
3. 或把 `min_visible_misses` 从 1 调到 2。
4. 同时增加每层过滤统计日志。

## 20. 推荐下一步改动清单

短期建议，优先级从高到低：

1. 增加 `object3d_filter_diagnostics.json`，记录每帧每层过滤数量。
2. 将 disappearance 的有效可见距离从 5m 调整为 3m，避免和 3m depth gate 冲突。
3. 将 `object3d_disappearance_min_visible_misses` 从 1 调整到 2，减少单帧漏检误删。
4. 在回放 HTML 中显示每个物体的生命周期状态：raw、stable、overlap_removed、motion_removed、disappearance_removed、active。
5. 如果后续发现 3m 边界抖动明显，再考虑加入 3.0m/3.3m 滞回策略。

中期建议：

1. 把 3m 阈值从硬编码 `observation_max_depth_m=3.0` 改成 config 参数。
2. 把 disappearance filter 的 depth gate 也纳入 config。
3. 将连续帧稳定过滤升级为“窗口内出现次数”过滤，允许小缺口。
4. 为每个 removed object 保存 removal reason 和关键证据帧截图。

长期建议：

1. 建立 Object3D 生命周期状态机，而不是简单 hard delete。
2. 支持 temporarily_missing 和 reactivated。
3. 对消失判断加入遮挡推理或深度遮挡检查。
4. 对稳定物体使用更强的跨帧 3D 融合，而不是只依赖单帧 bbox。

## 21. 最终建议

如果目标是“生成可信拓扑地图”，3m 深度过滤应该保留。它可以明显减少远距离错误点云和不可靠物体节点。

但它必须和后续消失过滤保持一致。当前最大的问题不是 3m 本身，而是：

> 观测准入是 3m，消失可见性判断是 5m。

这个不一致会导致某些 3m 到 5m 的回访帧被消失过滤当作有效 miss，而这些帧的检测又可能因为 3m depth gate 无法进入 payload。

因此，最稳妥的工程策略是：

```text
Object3D 观测最大深度 = 3m
消失过滤可见机会最大深度 = 3m 或 3m + 小 margin
消失删除至少需要连续 2 次有效 miss
每帧记录过滤诊断
```

这样可以让“近距离可信观测”和“近距离可信消失判断”保持同一套物理边界，减少算法内部自相矛盾。

当前一帧和五帧验证已经说明：3m 过滤不会天然导致全空；真正需要保护的是它和消失过滤、连续帧稳定过滤之间的边界交互。
