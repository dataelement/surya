import base64
import io
from typing import List

from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel

from surya.service.reading_order import ReadingOrderService

# 初始化FastAPI和ReadingOrderService
app = FastAPI(title="Reading Order Service")
service = ReadingOrderService()


class OrderRequest(BaseModel):
    """请求体模型"""

    image_base64: str  # base64编码的图片
    layout_boxes: List[List[List[float]]]  # layout boxes坐标列表


class OrderResponse(BaseModel):
    """响应体模型"""

    results: List[int]  # 预测结果


@app.post("/predict", response_model=OrderResponse)
async def predict_order(request: OrderRequest):
    """
    预测阅读顺序的API端点

    Args:
        request: 包含base64图片和layout boxes的请求体

    Returns:
        包含预测结果的响应体
    """
    try:
        # 解码base64图片
        image_data = base64.b64decode(request.image_base64)
        image = Image.open(io.BytesIO(image_data))

        # 调用服务进行预测
        results = service.predict(image, request.layout_boxes)

        return OrderResponse(results=results)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """健康检查接口"""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=9199)
