# Object3D 坐标系问题说明

生成时间：2026-05-28

项目路径：`/home/zyf/code_made`

相关数据集：`/home/zyf/Desktop/dataset_test3`

相关代码：

- `/home/zyf/code_made/run_semantic_topomap.py`
- `/home/zyf/Desktop/3d/object3d_engine/core/pointcloud_service.py`
- `/home/zyf/Desktop/3d/object3d_engine/domain/value_objects.py`

## 1. 之前遇到的现象

之前在看 `object3d_global_map.html`、`nav_graph_visualization_3d.html`、拓扑回放时，出现过非常明显的不合理现象：

- RGB 里门一直在相机左侧，但 3D 地图里被投到了轨迹右侧。
- 左右两侧的物体都有可能被挤到轨迹中间。
- 物体节点不像是在真实墙边或门边，而像贴着相机轨迹漂浮。
- 同一帧里 bbox 看起来是对的，但回投后的全局坐标不符合直觉。

这个问题不是 YOLO 分类问题，也不是 SAM 是否生成 mask 的问题。YOLO 框在 RGB 上能看见，mask 也能抠出点云，真正出错的是：

> 相机局部坐标系的点云，被错误地解释成项目世界坐标系方向。

一句话概括：

> Object3D 点云使用的是相机 optical 坐标系，但我们一开始直接把它乘上轨迹姿态，没有先做 optical 坐标轴到项目坐标轴的转换，导致左右/前后/上下关系被错误解释。

## 2. 这条链路里有哪些坐标系

当前链路里至少有三套坐标概念。

第一套是图像像素坐标：

```text
u: 图像横向，向右为正
v: 图像纵向，向下为正
```

YOLO bbox 和 SAM mask 都在这套坐标里。

第二套是相机 optical 局部坐标，也就是 depth 回投后得到的局部 3D 点：

```text
x: 向右
y: 向下
z: 向前
```

这是典型 pinhole camera / optical frame 约定。代码在 `/home/zyf/Desktop/3d/object3d_engine/core/pointcloud_service.py`：

```python
z = depth[valid_mask]
x = (u_coords - intr.cx) * z / intr.fx
y = (v_coords - intr.cy) * z / intr.fy
points = np.stack((x, y, z), axis=-1)
```

所以 `local_points` 绝对不是“项目世界坐标”，而是相机 optical 坐标。

第三套是项目使用的世界/轨迹坐标。当前项目里，我们按这个含义理解：

```text
x: 右
y: 前
z: 上
```

轨迹 `position` 和 `rotation` 描述的是相机/机器人在这个项目世界坐标里的位置和朝向。

问题正是发生在第二套坐标到第三套坐标之间。

## 3. 错误的本质

Object3D engine 的点云转换逻辑是：

`/home/zyf/Desktop/3d/object3d_engine/domain/value_objects.py`

```python
rotated = points @ self.rotation.T
return rotated + self.translation
```

也就是说，Object3D 会把 `local_points` 乘上 `frame.pose.rotation`，再加上 `frame.pose.translation`。

因此传给 Object3D 的 `pose_matrix` 必须表达：

> 如何把 Object3D local_points 所在的相机 optical 坐标，转换到项目世界坐标。

一开始的问题是，我们把轨迹的 `rotation` 直接当成了 local_points 的旋转。这样等价于假设 local_points 已经使用项目局部坐标约定：

```text
x: 右
y: 前
z: 上
```

但事实上 local_points 是：

```text
x: 右
y: 下
z: 前
```

这就发生了轴语义错位。

最关键的错位是：

- 相机 optical 的 `z` 是前方。
- 项目世界里的前方应该映射到 `y`。
- 相机 optical 的 `y` 是向下。
- 项目世界里的向上应该是 `z`，所以 optical 向下应该映射成 world 负 z。

如果不做这一步，物体本来在画面左侧/前方/下方的关系，会被错误旋到别的位置。于是你会看到：

- 左边的门出现在右边。
- 前方的东西变成高度或侧向偏移。
- 物体点云贴着轨迹或集中到中间。

## 4. 正确的轴映射

我们需要把相机 optical 坐标：

```text
camera optical:
x_cam = right
y_cam = down
z_cam = forward
```

映射到项目局部坐标：

```text
project local/world convention:
x_proj = right
y_proj = forward
z_proj = up
```

因此：

```text
x_proj =  x_cam
y_proj =  z_cam
z_proj = -y_cam
```

写成矩阵就是：

```text
[x_proj]   [1  0  0] [x_cam]
[y_proj] = [0  0  1] [y_cam]
[z_proj]   [0 -1  0] [z_cam]
```

也就是：

```python
optical_to_world_axes = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, -1.0, 0.0],
    ],
    dtype=np.float64,
)
```

注意这个矩阵不是完整世界旋转，它只是“相机 optical 轴到项目局部轴”的固定轴变换。

完整位姿应该是：

```python
pose[:3, :3] = world_rotation @ optical_to_world_axes
pose[:3, 3] = position
```

其中：

- `world_rotation` 来自轨迹 quaternion。
- `optical_to_world_axes` 修正相机 optical 坐标轴约定。
- `position` 是相机/机器人在世界坐标中的位置。

## 5. 当前代码里的修复

当前修复点在：

`/home/zyf/code_made/run_semantic_topomap.py`

函数：

```python
def _build_object3d_pose_matrix(self, position, rotation):
    pose = np.eye(4, dtype=np.float64)
    world_rotation = np.asarray(quaternion.as_rotation_matrix(rotation), dtype=np.float64)
    optical_to_world_axes = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, -1.0, 0.0],
        ],
        dtype=np.float64,
    )
    pose[:3, :3] = world_rotation @ optical_to_world_axes
    pose[:3, 3] = np.asarray(position, dtype=np.float64)
    return pose
```

这个函数现在用于两个地方：

1. Object3D 单帧观测生成时，把 mask+depth 点云放到全局坐标。
2. 消失回访判断时，把全局物体中心反投到当前相机坐标，判断是否在相机前方、是否在 3m 内、是否在视场内。

第二点也很重要。我们后面做了“3m 内才做消失回访”的逻辑，如果这个 pose 方向错了，消失过滤也会错。例如：

- 真实在左前方的物体被反算到右前方。
- 真实在前方的物体被反算到侧后方。
- 本来应该可见的物体被判断为不可见。
- 本来不可见的物体被判断为可见并计 miss。

所以坐标修复不仅影响可视化，也影响过滤逻辑。

## 6. 为什么会出现“左侧门被投到右侧”

以门在图像左侧为例。

在相机 optical 坐标里，图像左侧意味着：

```text
x_cam < 0
z_cam > 0
```

也就是点在相机左前方。

正确映射后：

```text
x_proj = x_cam  -> 仍然是负，表示左侧
y_proj = z_cam  -> 正，表示前方
z_proj = -y_cam -> 根据图像上下决定高度
```

所以门应该落在轨迹左前方。

如果不做 optical 轴映射，系统可能把 `z_cam` 当成项目 z，或者把 `y_cam` 当成项目 y。这样“前方”会被解释成“上方”或其他轴向，“向下”会被解释成“前后”，最终物体就会跑到不符合直觉的位置。

这就是为什么你当时看到：

> 门明明一直在左侧，却被定位到右侧或轨迹中间。

本质不是门的检测错了，而是检测点云从相机局部坐标到世界坐标时轴没对上。

## 7. 为什么物体会集中到轨迹中间

物体集中到轨迹中间通常有几个叠加原因：

1. 坐标轴映射错，前方深度没有变成世界前方距离。
2. bbox/mask 深度点云被错误旋转后，中心落到相机附近。
3. 轨迹 position 正确，但局部点云方向错，所有物体都围绕相机轨迹错误展开。
4. 如果 depth/RGB 对齐再有一点误差，点云中心会更容易漂到背景或轨迹附近。

修复 optical 轴映射后，至少可以保证：

```text
图像左侧 -> 世界左侧
图像前方深度 -> 世界前方距离
图像向下 -> 世界负高度
```

这样物体不会因为坐标定义错而系统性地挤到轨迹中间。

## 8. 第一帧如何验证

验证坐标修复时，不建议一开始就看完整图，因为完整图会叠加 tracking、过滤、合并、waypoint sampling 等影响。

最稳的是看第一帧单个检测。

建议步骤：

1. 打开第一帧 RGB。
2. 找一个直觉明确的物体，例如左侧门。
3. 查看该检测 bbox 是否确实在图像左侧。
4. 看该检测的 `bbox_3d_center` 或 `global_position`。
5. 如果相机初始姿态近似正向，左侧物体的 x 应该是负或在轨迹左侧。
6. 前方物体的 y 应该为正。
7. 高度 z 不应该被前方深度撑得很大。

当时修复后的一个关键验证现象是：

```text
door centroid 从错误侧移动到左前方
```

之前错误结果类似：

```text
[+1.772, +1.661, 0.110]
```

修复后更合理：

```text
[-1.856, +1.788, 0.096]
```

这里最关键的是 x 从正变成负，符合“门在左侧”的直觉。

## 9. 如何区分坐标系问题和深度/RGB 对齐问题

这两个问题很容易混在一起，但它们的表现不完全一样。

坐标系问题通常是系统性的：

- 左右经常反。
- 前后/上下经常混。
- 多数物体都以类似方式偏。
- 换一帧还是同样方向错误。

深度/RGB 对齐问题通常是局部性的：

- 某些物体点云落到背景。
- 某些 bbox 深度明显不对。
- 同一帧里有些物体对，有些物体错。
- mask 边缘、玻璃、门框、桌面等区域更容易出错。

排查顺序建议：

1. 先确认坐标轴映射是否正确，因为这是全局基础。
2. 再确认 RGB 和 depth 分辨率是否一致。
3. 再确认 RGB 文件和 depth 文件时间戳是否一一对应。
4. 再看 bbox/mask 内深度分布是否合理。
5. 最后再看 Object3D tracking 和过滤。

## 10. 对消失回访过滤的影响

坐标系问题不仅影响物体显示位置，也影响消失回访判断。

现在消失回访逻辑会做：

1. 已知物体有一个全局中心 `object_center`。
2. 当前帧有相机 `camera_pose`。
3. 用 `inv(camera_pose)` 把 object center 反算回当前相机局部坐标。
4. 判断：
   - 是否在相机前方。
   - `forward_z <= 3.0m`。
   - 是否在 FOV 内。
   - 是否接近之前观察位置。

如果 `camera_pose` 的轴错了，`forward_z` 就不可信。

可能出现：

- 物体明明在相机前方，却被判断到后方。
- 物体明明超过 3m，却被判断成 3m 内。
- 物体明明在画面左侧，却被判断到右侧或视野外。
- 错误地计入 visible miss，导致误删。

所以我们现在使用同一个 `_build_object3d_pose_matrix()` 做正向投影和反向可见性判断，保证正反两个方向使用同一套坐标约定。

这就是闭环：

```text
mask + depth local optical points
-> _build_object3d_pose_matrix
-> global object center

global object center
-> inv(_build_object3d_pose_matrix)
-> current camera local optical point
-> 3m/FOV visibility check
```

只要这个矩阵定义正确，前后判断就不会自相矛盾。

## 11. 后续避免再犯的规则

以后凡是新增 2D 到 3D、3D 到 2D、或者全局到相机局部的逻辑，都要先回答一个问题：

> 当前点到底在哪个坐标系？

必须明确：

- 是像素坐标，还是相机 optical 坐标？
- 是机器人 base 坐标，还是项目 world 坐标？
- quaternion 表达的是哪个坐标系到哪个坐标系？
- 矩阵左乘/右乘约定是什么？
- 点是 row vector 还是 column vector？

当前 Object3D engine 里 `Pose3D.transform_points()` 使用：

```python
rotated = points @ self.rotation.T
return rotated + self.translation
```

这等价于：

```text
global_point = R * local_point + t
```

只是代码里 points 是 row-vector 形式，所以写成 `points @ R.T`。

因此 `pose[:3, :3]` 必须是 local-to-global 的旋转。

## 12. 快速检查清单

如果后面又出现“物体位置不对”，先按这个清单检查：

- 第一帧左侧物体的全局 x 是否在左侧。
- 前方物体是否主要体现在世界前向轴，而不是高度轴。
- `local_points[:, 2]` 是否是合理深度。
- RGB shape 和 depth shape 是否一致。
- `depth_ts` 或 depth 文件名是否和 RGB 帧对应。
- `_build_object3d_pose_matrix()` 是否仍然包含 `optical_to_world_axes`。
- 消失回访里是否使用同一个 `_build_object3d_pose_matrix()` 做反算。
- 是否有人把 `pose[:3, :3] = world_rotation` 又改回去了。

最重要的红线：

> 不要直接用轨迹 rotation 去变换相机 optical 点云。必须先做 optical-to-project 轴映射。

## 13. 最终结论

之前的坐标问题根因是：

> 相机 depth 回投出的 Object3D local points 是 optical 坐标 `x右、y下、z前`，但项目世界/轨迹坐标按 `x右、y前、z上` 使用。直接用轨迹 quaternion 变换 local points，会把前方/上下/左右关系解释错。

当前修复是：

```python
optical_to_world_axes = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, -1.0, 0.0],
    ],
    dtype=np.float64,
)
pose[:3, :3] = world_rotation @ optical_to_world_axes
```

修复后，图像左侧物体会投到世界左侧，前方深度会投到世界前方，高度不会被错误当成距离。这个修复同时保证了 Object3D 建图和消失回访判断使用同一套坐标闭环。

