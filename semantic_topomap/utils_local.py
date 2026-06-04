import numpy as np
import cv2
import quaternion


def calculate_focal_length_px(fov_degrees: float, img_size: int) -> float:
    return img_size / (2 * np.tan(np.radians(fov_degrees / 2)))


def calculate_focal_length_from_hfov_vfov(hfov, vfov, img_width, img_height):
    h_focal_px = calculate_focal_length_px(hfov, img_width)
    v_focal_px = calculate_focal_length_px(vfov, img_height)
    return h_focal_px, v_focal_px


def get_avg_depth_in_box(depth_map, x1, y1, x2, y2, edge_ratio=0.1):
    h_map, w_map = depth_map.shape[:2]

    x1, y1, x2, y2 = map(float, [x1, y1, x2, y2])
    x1, x2 = sorted([x1, x2])
    y1, y2 = sorted([y1, y2])

    w = x2 - x1
    h = y2 - y1
    if w <= 0 or h <= 0:
        return None

    sample_points = [
        (x1 + 0.5 * w, y1 + 0.5 * h),
        (x1 + edge_ratio * w, y1 + edge_ratio * h),
        (x1 + (1 - edge_ratio) * w, y1 + edge_ratio * h),
        (x1 + edge_ratio * w, y1 + (1 - edge_ratio) * h),
        (x1 + (1 - edge_ratio) * w, y1 + (1 - edge_ratio) * h),
    ]

    depths = []
    for x, y in sample_points:
        xi, yi = int(round(x)), int(round(y))
        if 0 <= xi < w_map and 0 <= yi < h_map:
            d = depth_map[yi, xi]
            if np.isfinite(d) and d > 0:
                depths.append(float(d))

    if not depths:
        return None
    return float(np.mean(depths))


def remove_boxes_near_border(detection_objects, img_width, img_height, margin_ratio=0.05):
    margin_x = img_width * margin_ratio
    margin_y = img_height * margin_ratio
    filtered_objects = []

    for det in detection_objects:
        bbox = det[1]
        x1, y1, x2, y2 = bbox
        if x1 >= margin_x and y1 >= margin_y and x2 <= img_width - margin_x and y2 <= img_height - margin_y:
            filtered_objects.append(det)
    return filtered_objects


def draw_filtered_detections(img, filtered_objects):
    out = img.copy()
    for det in filtered_objects:
        class_name, bbox, confidence, _, depth = det
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{class_name} {confidence:.2f} D={depth:.2f}"
        cv2.putText(out, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return out


def image_to_local_1(x_pixel, y_pixel, depth, resolution, focal_length_x, focal_length_y):
    y_normalized = -(x_pixel - resolution[1] / 2) / focal_length_x
    z_normalized = (y_pixel - resolution[0] / 2) / focal_length_y

    x_3d = depth
    y_3d = y_normalized * depth
    z_3d = z_normalized * depth
    return np.array([x_3d, y_3d, z_3d])


def local_to_global_1(position, orientation, local_point):
    rotated_point = quaternion.rotate_vectors(orientation, local_point)
    return rotated_point + position
