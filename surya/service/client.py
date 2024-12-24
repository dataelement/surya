import base64
import io
from typing import List, Union

import requests
from PIL import Image


class ReadingOrderClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        初始化客户端

        Args:
            base_url: API服务的基础URL
        """
        self.base_url = base_url.rstrip("/")

    def _convert_image_to_base64(self, image: Union[str, Image.Image]) -> str:
        """将图片转换为base64编码"""
        if isinstance(image, str):
            image = Image.open(image)

        # 将图片转换为bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format=image.format or 'PNG')
        img_byte_arr = img_byte_arr.getvalue()

        # 转换为base64
        return base64.b64encode(img_byte_arr).decode('utf-8')

    def predict(self, image: Union[str, Image.Image], layout_boxes: List[List[float]]) -> List[dict]:
        """
        预测图片中layout boxes的阅读顺序

        Args:
            image: 图片路径或PIL.Image对象
            layout_boxes: Layout boxes列表，每个box格式为[x1,y1,x2,y2,x3,y3,x4,y4]

        Returns:
            包含位置信息的layout boxes列表
        """
        # 准备请求数据
        image_base64 = self._convert_image_to_base64(image)
        payload = {"image_base64": image_base64, "layout_boxes": layout_boxes}

        # 发送请求
        response = requests.post(f"{self.base_url}/predict", json=payload)

        # 检查响应
        if response.status_code != 200:
            raise Exception(f"API请求失败: {response.text}")

        return response.json()["results"]

    def health_check(self) -> bool:
        """检查服务是否健康"""
        try:
            response = requests.get(f"{self.base_url}/health")
            return response.status_code == 200
        except:
            return False