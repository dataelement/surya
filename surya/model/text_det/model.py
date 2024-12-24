import torch
from rapidocr_paddle import RapidOCR
from ultralytics import YOLO

from surya.settings import settings

def load_yolo_model(ckpt=settings.YOLO_DETECTOR_CHECKPOINT) -> YOLO:
    layout_det_yolo = YOLO(ckpt)
    print(f"Loaded text detection model {ckpt}")
    return layout_det_yolo


def load_pp_model():
    model = RapidOCR(det_use_cuda=True)
    print(f"Loaded text detection model {model}")
    return model
