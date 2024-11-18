import torch
from ultralytics import YOLO

from surya.settings import settings


def load_model(ckpt=settings.LAYOUT_YOLO_CHECKPOINT) -> YOLO:
    layout_det_yolo = YOLO(ckpt)
    print(f"Loaded layout model {ckpt}")
    return layout_det_yolo
