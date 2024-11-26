import torch
from rapidocr_onnxruntime import RapidOCR
from ultralytics import YOLO

from surya.settings import settings

# def load_model(ckpt=settings.YOLO_DETECTOR_CHECKPOINT) -> YOLO:
#     layout_det_yolo = YOLO(ckpt)
#     print(f"Loaded text detection model {ckpt}")
#     return layout_det_yolo


def load_pp_model():
    model = RapidOCR()
    print(f"Loaded text detection model {model}")
    return model
