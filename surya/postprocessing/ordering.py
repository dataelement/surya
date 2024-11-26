from collections import defaultdict
from statistics import median
from typing import Any, Dict, List, Sequence

from surya.schema import LayoutBox, PolygonBox, TextLine


def _has_significant_y_overlap(
    bbox1: Sequence[float], bbox2: Sequence[float], overlap_ratio_threshold: float = 0.8
) -> bool:
    """
    检查两个边界框在y轴上的重叠程度是否超过指定阈值。

    Args:
        bbox1: 第一个边界框，格式为 [x0, y0, x1, y1] 或 (x0, y0, x1, y1)
        bbox2: 第二个边界框，格式为 [x0, y0, x1, y1] 或 (x0, y0, x1, y1)
        overlap_ratio_threshold: 重叠比例阈值，默认为0.8

    Returns:
        bool: 如果重叠区域高度占较小边界框高度的比例超过阈值，返回True；否则返回False

    Note:
        重叠比例 = 重叠区域高度 / min(bbox1高度, bbox2高度)
    """
    _, y0_first, _, y1_first = bbox1
    _, y0_second, _, y1_second = bbox2

    overlap_height = max(0, min(y1_first, y1_second) - max(y0_first, y0_second))
    height_first = y1_first - y0_first
    height_second = y1_second - y0_second
    min_height = min(height_first, height_second)

    return (overlap_height / min_height) > overlap_ratio_threshold


def merge_spans_to_lines(spans: List[PolygonBox]):
    """
    将span列表合并成行。每行包含一组在y轴方向上重叠的spans。

    Args:
        spans (list): span字典的列表。每个span字典包含：
            - bbox (list): [x0, y0, x1, y1] 格式的边界框坐标
            - type (str): span的类型，如文本、公式、图片等

    Returns:
        list: 行的列表，每行是一个span列表。行按照y坐标从上到下排序。

    工作流程:
        1. 首先按y0坐标对spans进行排序（从上到下）
        2. 遍历所有spans，根据以下规则将它们分组成行：
           - 如果当前span是行间公式、图片或表格，或当前行已包含这些类型，
             则开始新行
           - 如果当前span与当前行的最后一个span在y轴上重叠超过50%，
             则添加到当前行
           - 否则开始新行

    示例:
        >>> spans = [
            {'bbox': [0, 0, 10, 10], 'type': 'text'},
            {'bbox': [20, 1, 30, 9], 'type': 'text'},
            {'bbox': [0, 20, 10, 30], 'type': 'text'}
        ]
        >>> lines = merge_spans_to_line(spans)
        >>> # 结果会将前两个span合并到一行，第三个span单独一行
    """
    if len(spans) == 0:
        return []
    else:
        # 按照y0坐标排序
        spans.sort(key=lambda span: span.bbox[1])

        lines = []
        current_line = [spans[0]]
        for span in spans[1:]:
            # # 如果当前的span类型为"interline_equation" 或者 当前行中已经有"interline_equation"
            # # image和table类型，同上
            # if span['type'] in [ContentType.InterlineEquation, ContentType.Image, ContentType.Table] or any(
            #     s['type'] in [ContentType.InterlineEquation, ContentType.Image, ContentType.Table] for s in current_line
            # ):
            #     # 则开始新行
            #     lines.append(current_line)
            #     current_line = [span]
            #     continue

            # 如果当前的span与当前行的最后一个span在y轴上重叠，则添加到当前行
            if _has_significant_y_overlap(span.bbox, current_line[-1].bbox, 0.5):
                current_line.append(span)
            else:
                # 否则，开始新行
                lines.append(current_line)
                current_line = [span]

        # 添加最后一行
        if current_line:
            lines.append(current_line)

        return lines


def sort_spans_horizontally(lines: List[List[PolygonBox]]) -> List[Dict[str, Any]]:
    """
    将每一行中的 spans 按照从左到右的顺序排序，并计算行的边界框。

    Args:
        lines: 包含多行 spans 的列表，每行是一个 PolygonBox 列表

    Returns:
        List[Dict]: 每个字典包含：
            - bbox: 整行的边界框坐标 [x0, y0, x1, y1]
            - spans: 按照从左到右排序后的 spans 列表
    """
    line_objects = []

    for spans in lines:
        if not spans:
            continue

        # 按照 x0 坐标排序
        sorted_spans = sorted(spans, key=lambda span: span.bbox[0])

        # 使用 zip 和 map 优化边界框计算
        x0s, y0s, x1s, y1s = zip(*(span.bbox for span in sorted_spans))
        line_bbox = [min(x0s), min(y0s), max(x1s), max(y1s)]

        line_objects.append(
            {
                'bbox': line_bbox,
                'spans': sorted_spans,
            }
        )

    return line_objects


def create_virtual_lines(
    block_bbox: Sequence[float], line_height: float, page_w: float, page_h: float
) -> List[Dict[str, Any]]:
    """将一个区块(block)划分为多个行(lines)

    这个函数的主要目的是将大的文本区块合理地划分成多个行，以便于后续的文本排序处理。
    划分策略会根据区块的尺寸（高度和宽度）以及页面布局特征（如单列、双列、三列）来决定。

    Args:
        block_bbox (tuple): 区块的边界框坐标 (x0, y0, x1, y1)
        line_height (float): 标准行高
        page_w (float): 页面宽度
        page_h (float): 页面高度

    Returns:
        list: 包含多个行的字典，每个字典包含：
            - bbox: 行的边界框坐标 [x0, y0, x1, y1]
            - spans: 行的 spans 列表, 在这里行和span是等价的
            示例:
            {'bbox': [293, 2462, 2252, 2496], 'spans': [])]}
    """
    # 解析边界框坐标
    x0, y0, x1, y1 = block_bbox

    # 计算区块的高度和宽度
    block_height = y1 - y0
    block_weight = x1 - x0

    # 只有当区块高度大于3倍行高时才考虑划分
    if line_height * 3 < block_height:
        # 情况1: 可能是双列结构
        # 当区块高度超过页面1/4，且宽度在页面1/4到1/2之间时
        if block_height > page_h * 0.25 and page_w * 0.5 > block_weight > page_w * 0.25:
            lines = int(block_height / line_height) + 1  # 按标准行高划分
        else:
            # 情况2: 宽区块（可能是复杂布局）
            if block_weight > page_w * 0.4:
                line_height = (y1 - y0) / 3  # 强制分成3行
                lines = 3
            # 情况3: 中等宽度（可能是三列结构）
            elif block_weight > page_w * 0.25:
                lines = int(block_height / line_height) + 1  # 按标准行高划分
            # 情况4: 窄区块
            else:
                # 检查长宽比，大于1.2认为是细长型
                if block_height / block_weight > 1.2:
                    return [{'bbox': [x0, y0, x1, y1], 'spans': []}]  # 细长的区块不划分
                else:
                    line_height = (y1 - y0) / 2  # 强制分成2行
                    lines = 2

        # 生成行的位置信息
        current_y = y0
        lines_positions = []
        for i in range(lines):
            bbox = [x0, current_y, x1, current_y + line_height]
            lines_positions.append({'bbox': bbox, 'spans': []})
            current_y += line_height
        return lines_positions

    else:
        # 区块高度较小时，不进行划分
        bbox = [x0, y0, x1, y1]
        return [{'bbox': bbox, 'spans': []}]


def assign_spans_to_blocks(blocks: List[LayoutBox], spans: List[PolygonBox]) -> Dict[int, List[PolygonBox]]:
    """Assign text detection spans to layout blocks based on intersection."""
    block2spans = dict()
    used_spans = set()

    for block_idx, block in enumerate(blocks):
        block2spans[block_idx] = []
        for span_idx, span in enumerate(spans):
            if span_idx not in used_spans and span.intersection_pct(block) > 0.5:
                block2spans[block_idx].append(span)
                used_spans.add(span_idx)

    return block2spans
