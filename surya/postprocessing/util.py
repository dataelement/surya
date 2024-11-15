import copy
import math
from collections import defaultdict
from typing import List


def get_line_angle(x1, y1, x2, y2):
    slope = (y2 - y1) / (x2 - x1)

    angle_radians = math.atan(slope)
    angle_degrees = math.degrees(angle_radians)

    return angle_degrees


def rescale_bbox(bbox, processor_size, image_size):
    page_width, page_height = processor_size

    img_width, img_height = image_size
    width_scaler = img_width / page_width
    height_scaler = img_height / page_height

    new_bbox = copy.deepcopy(bbox)
    new_bbox[0] = int(new_bbox[0] * width_scaler)
    new_bbox[1] = int(new_bbox[1] * height_scaler)
    new_bbox[2] = int(new_bbox[2] * width_scaler)
    new_bbox[3] = int(new_bbox[3] * height_scaler)
    return new_bbox


def rescale_bboxes(bboxes, orig_size, new_size):
    return [rescale_bbox(bbox, orig_size, new_size) for bbox in bboxes]


def rescale_point(point, processor_size, image_size):
    # Point is in x, y format
    page_width, page_height = processor_size

    img_width, img_height = image_size
    width_scaler = img_width / page_width
    height_scaler = img_height / page_height

    new_point = copy.deepcopy(point)
    new_point[0] = int(new_point[0] * width_scaler)
    new_point[1] = int(new_point[1] * height_scaler)
    return new_point


def rescale_points(points, processor_size, image_size):
    return [rescale_point(point, processor_size, image_size) for point in points]


def xyxy2xyxyxyxy(xyxy: List[float]) -> List[List[float]]:
    pt1 = [xyxy[0], xyxy[1]]
    pt2 = [xyxy[2], xyxy[1]]
    pt3 = [xyxy[2], xyxy[3]]
    pt4 = [xyxy[0], xyxy[3]]
    return [pt1, pt2, pt3, pt4]


def intersection_over_min_area(box1, box2):
    """
    计算两个box的交集面积与较小box面积的比值

    Args:
        box1: List[float], [x1, y1, x2, y2] 格式的第一个box坐标 (左上角x1,y1, 右下角x2,y2)
        box2: List[float], [x1, y1, x2, y2] 格式的第二个box坐标 (左上角x1,y1, 右下角x2,y2)

    Returns:
        float: 交集面积与较小box面积的比值
    """
    # 计算交集区域的坐标
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # 计算交集面积
    if x2 <= x1 or y2 <= y1:  # 没有交集
        return 0.0
    intersection = (x2 - x1) * (y2 - y1)

    # 计算两个box的面积
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # 返回交集面积除以较小box的面积
    return intersection / min(area1, area2)


def remove_overlapping_boxes(boxes: List['LayoutBox'], iou_threshold: float = 0.9) -> List['LayoutBox']:
    """
    移除重叠度高的同类别框中较小的那个

    Args:
        boxes: List[LayoutBox], 边界框列表
        iou_threshold: float, 交集面积与较小box面积比值的阈值

    Returns:
        List[LayoutBox]: 处理后的边界框列表
    """
    # 按类别分组
    boxes_by_class = defaultdict(list)
    for box in boxes:
        boxes_by_class[box.label].append(box)

    # 存储需要移除的框的索引
    to_remove = set()

    # 对每个类别分别处理
    for label, class_boxes in boxes_by_class.items():
        for i in range(len(class_boxes)):
            if i in to_remove:
                continue
            for j in range(i + 1, len(class_boxes)):
                if j in to_remove:
                    continue

                box1, box2 = class_boxes[i], class_boxes[j]
                # 计算交集比
                overlap_ratio = intersection_over_min_area(box1.bbox, box2.bbox)

                if overlap_ratio > iou_threshold:
                    # 移除面积较小的框
                    if box1.area < box2.area:
                        to_remove.add(i)
                        break
                    else:
                        to_remove.add(j)

    # 返回未被移除的框
    return [box for idx, box in enumerate(boxes) if idx not in to_remove]
