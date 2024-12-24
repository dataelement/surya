from typing import List, Union

import numpy as np
from PIL import Image

from surya.detection import text_detection_yolo
from surya.input.processing import convert_if_not_rgb
from surya.layout import LayoutBox, LayoutResult
from surya.model.layoutlmv3_order.model import load_model as load_order_model
from surya.model.text_det.model import load_pp_model
from surya.ordering import elem_batch_ordering


class ReadingOrderService:
    def __init__(self):
        # 初始化所需的模型
        self.order_model = load_order_model()
        self.text_det_model = load_pp_model()

    def get_dummy_layout_result(self, image: Image.Image, layout_boxes: List[List[List[float]]]) -> LayoutResult:
        """
        将输入的layout boxes转换为LayoutResult格式
        """
        layout_boxes = [LayoutBox(polygon=box, confidence=1.0, label='Text') for box in layout_boxes]

        return LayoutResult(
            bboxes=layout_boxes, segmentation_map=None, heatmaps=None, image_bbox=[0, 0, image.width, image.height]
        )

    def predict(self, image: Union[Image.Image, str], layout_boxes: List[List[float]]) -> List[dict]:
        """
        预测图片中layout boxes的阅读顺序

        Args:
            image: PIL.Image对象或图片路径
            layout_boxes: Layout boxes列表，每个box格式为(4,2)

        Returns:
            包含位置信息的layout boxes列表，每个元素为dict:
            {
                "bbox": (4,2)
                "position": int (阅读顺序)
            }
        """
        # 如果输入是图片路径，加载图片
        if isinstance(image, str):
            image = Image.open(image)

        # 确保图片是RGB格式
        images = convert_if_not_rgb([image])

        # 准备layout result
        layout_result = [self.get_dummy_layout_result(image, layout_boxes)]

        # 文本检测
        span_predictions = text_detection_yolo(images, self.text_det_model)

        # 预测阅读顺序
        order_predictions = elem_batch_ordering(
            images=images, model=self.order_model, text_det_results=span_predictions, layout_results=layout_result
        )[0]

        return [order_pred.position for order_pred in order_predictions.bboxes]
