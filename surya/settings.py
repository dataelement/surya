import os
from typing import Dict, Optional

import torch
from dotenv import find_dotenv
from pydantic import computed_field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # General
    TORCH_DEVICE: Optional[str] = None
    IMAGE_DPI: int = 96  # Used for detection, layout, reading order
    IMAGE_DPI_HIGHRES: int = 192  # Used for OCR, table rec
    IN_STREAMLIT: bool = False  # Whether we're running in streamlit
    ENABLE_EFFICIENT_ATTENTION: bool = True  # Usually keep True, but if you get CUDA errors, setting to False can help
    ENABLE_CUDNN_ATTENTION: bool = (
        False  # Causes issues on many systems when set to True, but can improve performance on certain GPUs
    )
    FLATTEN_PDF: bool = True  # Flatten PDFs by merging form fields before processing

    # Paths
    DATA_DIR: str = "data"
    RESULT_DIR: str = "results"
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    FONT_DIR: str = os.path.join(BASE_DIR, "static", "fonts")

    @computed_field
    def TORCH_DEVICE_MODEL(self) -> str:
        if self.TORCH_DEVICE is not None:
            return self.TORCH_DEVICE

        if torch.cuda.is_available():
            return "cuda"

        if torch.backends.mps.is_available():
            return "mps"

        return "cpu"

    # Text detection
    DETECTOR_BATCH_SIZE: Optional[int] = None  # Defaults to 2 for CPU/MPS, 32 otherwise
    DETECTOR_MODEL_CHECKPOINT: str = "vikp/surya_det3"
    DETECTOR_BENCH_DATASET_NAME: str = "vikp/doclaynet_bench"
    DETECTOR_IMAGE_CHUNK_HEIGHT: int = 1400  # Height at which to slice images vertically
    DETECTOR_TEXT_THRESHOLD: float = 0.6  # Threshold for text detection (above this is considered text)
    DETECTOR_BLANK_THRESHOLD: float = 0.35  # Threshold for blank space (below this is considered blank)
    DETECTOR_POSTPROCESSING_CPU_WORKERS: int = min(8, os.cpu_count())  # Number of workers for postprocessing
    DETECTOR_MIN_PARALLEL_THRESH: int = 3  # Minimum number of images before we parallelize
    COMPILE_DETECTOR: bool = False

    # Text recognition
    RECOGNITION_MODEL_CHECKPOINT: str = "vikp/surya_rec2"
    RECOGNITION_MAX_TOKENS: int = 175
    RECOGNITION_BATCH_SIZE: Optional[int] = None  # Defaults to 8 for CPU/MPS, 256 otherwise
    RECOGNITION_IMAGE_SIZE: Dict = {"height": 256, "width": 896}
    RECOGNITION_RENDER_FONTS: Dict[str, str] = {
        "all": os.path.join(FONT_DIR, "GoNotoCurrent-Regular.ttf"),
        "zh": os.path.join(FONT_DIR, "GoNotoCJKCore.ttf"),
        "ja": os.path.join(FONT_DIR, "GoNotoCJKCore.ttf"),
        "ko": os.path.join(FONT_DIR, "GoNotoCJKCore.ttf"),
    }
    RECOGNITION_FONT_DL_BASE: str = "https://github.com/satbyy/go-noto-universal/releases/download/v7.0"
    RECOGNITION_BENCH_DATASET_NAME: str = "vikp/rec_bench"
    RECOGNITION_PAD_VALUE: int = 255  # Should be 0 or 255
    COMPILE_RECOGNITION: bool = False  # Static cache for torch compile
    RECOGNITION_ENCODER_BATCH_DIVISOR: int = 1  # Divisor for batch size in decoder

    # Layout
    LAYOUT_MODEL_CHECKPOINT: str = "vikp/surya_layout3"
    LAYOUT_BENCH_DATASET_NAME: str = "vikp/publaynet_bench"
    COMPILE_LAYOUT: bool = False

    # Ordering
    ORDER_MODEL_CHECKPOINT: str = "vikp/surya_order"
    ORDER_IMAGE_SIZE: Dict = {"height": 1024, "width": 1024}
    ORDER_MAX_BOXES: int = 256
    ORDER_BATCH_SIZE: Optional[int] = None  # Defaults to 4 for CPU/MPS, 32 otherwise
    ORDER_BENCH_DATASET_NAME: str = "vikp/order_bench"

    # Table Rec
    TABLE_REC_MODEL_CHECKPOINT: str = "vikp/surya_tablerec"
    TABLE_REC_IMAGE_SIZE: Dict = {"height": 640, "width": 640}
    TABLE_REC_MAX_BOXES: int = 512
    TABLE_REC_MAX_ROWS: int = 384
    TABLE_REC_BATCH_SIZE: Optional[int] = None
    TABLE_REC_BENCH_DATASET_NAME: str = "vikp/fintabnet_bench"
    COMPILE_TABLE_REC: bool = False

    # Tesseract (for benchmarks only)
    TESSDATA_PREFIX: Optional[str] = None
    COMPILE_ALL: bool = False

    @computed_field
    def DETECTOR_STATIC_CACHE(self) -> bool:
        return self.COMPILE_ALL or self.COMPILE_DETECTOR

    @computed_field
    def RECOGNITION_STATIC_CACHE(self) -> bool:
        return self.COMPILE_ALL or self.COMPILE_RECOGNITION

    @computed_field
    def LAYOUT_STATIC_CACHE(self) -> bool:
        return self.COMPILE_ALL or self.COMPILE_LAYOUT

    @computed_field
    def TABLE_REC_STATIC_CACHE(self) -> bool:
        return self.COMPILE_ALL or self.COMPILE_TABLE_REC

    @computed_field
    @property
    def MODEL_DTYPE(self) -> torch.dtype:
        return torch.float32 if self.TORCH_DEVICE_MODEL == "cpu" else torch.float16

    class Config:
        env_file = find_dotenv("local.env")
        extra = "ignore"


class ElemSettings(Settings):
    # Only include fields that are different from parent Settings class
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODEL_DIR: str = "/workspace/models/surya"
    BENCH_DIR: str = "/workspace/datasets/surya_benchmarks"
    FONT_DIR: str = os.path.join(BASE_DIR, "static", "fonts")

    # Update paths that depend on MODEL_DIR and BENCH_DIR
    DETECTOR_MODEL_CHECKPOINT: str = os.path.join(MODEL_DIR, "surya_det3")
    DETECTOR_BENCH_DATASET_NAME: str = os.path.join(BENCH_DIR, "doclaynet_bench")

    RECOGNITION_MODEL_CHECKPOINT: str = os.path.join(MODEL_DIR, "surya_rec2")
    RECOGNITION_BENCH_DATASET_NAME: str = os.path.join(BENCH_DIR, "rec_bench")

    LAYOUT_MODEL_CHECKPOINT: str = os.path.join(MODEL_DIR, "surya_layout3")

    # layout yolo
    # LAYOUT_YOLO_CHECKPOINT: str = '/workspace/models/hantian/yolo-doclaynet/yolov10b-doclaynet.pt'
    # LAYOUT_YOLO_CHECKPOINT: str = '/workspace/youjiachen/workspace/ultralytics/layout_yolo11l_doclaynet_2/yolo11l_doclaynet_2_epoch50_imgsz1024_bs642/weights/best.pt'
    LAYOUT_YOLO_CHECKPOINT: str = (
        '/workspace/youjiachen/workspace/ultralytics/layout_yolo11l_doclaynet_2_from_pretrain/last_doclaynet_2_epoch50_imgsz1024_bs64/weights/best.pt'
    )
    # LAYOUT_YOLO_CHECKPOINT: str = (
    #     "/workspace/models/DocLayout-YOLO/DocLayout-YOLO-DocStructBench/doclayout_yolo_docstructbench_imgsz1024.onnx"
    # )
    LAYOUT_IMGSZ: int = 1024
    LAYOUT_DETECTOR_CONF: float = 0.4
    LAYOUT_DETECTOR_MAX_DET: int = 300
    LAYOUT_DETETOR_NUMS_THRESHOLD: float = 0.45
    ID2LABEL: Dict[int, str] = {
        0: "Caption",
        1: "Footnote",
        2: "Formula",
        3: "List-item",
        4: "Page-footer",
        5: "Page-header",
        6: "Picture",
        7: "Section-header",
        8: "Table",
        9: "Text",
        10: "Title",
    }
    # ID2LABEL: Dict[int, str] = {
    #     0: "title",
    #     1: "plain text",
    #     2: "abandon",
    #     3: "figure",
    #     4: "figure_caption",
    #     5: "table",
    #     6: "table_caption",
    #     7: "table_footnote",
    #     8: "isolate_formula",
    #     9: "formula_caption",
    # }

    # ID2LABEL: Dict[int, str] = {
    #     0: "Title",
    #     1: "Text",
    #     2: "Page-header",
    #     3: "Figure",
    #     4: "Figure-caption",
    #     5: "Table",
    #     6: "Table-caption",
    #     7: "Table-footnote",
    #     8: "Isolate-formula",
    #     9: "Formula-caption",
    # }

    LAYOUT_BENCH_DATASET_NAME: str = os.path.join(BENCH_DIR, "publaynet_bench")
    ELEM_LAYOUT_BENCH_DATASET_NAME: str = '/workspace/datasets/layout/dataelem_layout/yolo_format_merge_all'
    DOCLAYNET_BENCH_DATASET_PATH: str = '/workspace/datasets/layout/DocLayout-YOLO/layout_data/doclaynet'

    ORDER_MODEL_CHECKPOINT: str = os.path.join(MODEL_DIR, "surya_order")
    ORDER_LAYOUTLMV3_MODEL_CHECKPOINT: str = "/workspace/models/reading_order/elem_reading_order_v1"
    ORDER_LAYOUTLMV3_MODEL_CHECKPOINT: str = "/workspace/models/hantian/layoutreader"
    ORDER_BENCH_DATASET_NAME: str = os.path.join(BENCH_DIR, "order_bench")

    TABLE_REC_MODEL_CHECKPOINT: str = os.path.join(MODEL_DIR, "surya_tablerec")
    TABLE_REC_BENCH_DATASET_NAME: str = os.path.join(BENCH_DIR, "fintabnet_bench")

    # Other specific changes
    RECOGNITION_RENDER_FONTS: Dict[str, str] = {
        "all": os.path.join(FONT_DIR, "GoNotoCJKCore.ttf"),
        "zh": os.path.join(FONT_DIR, "GoNotoCJKCore.ttf"),
        "ja": os.path.join(FONT_DIR, "GoNotoCJKCore.ttf"),
        "ko": os.path.join(FONT_DIR, "GoNotoCJKCore.ttf"),
    }


# settings = Settings()
settings = ElemSettings()
