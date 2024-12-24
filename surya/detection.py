import contextlib
from concurrent.futures import ThreadPoolExecutor
from typing import Generator, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from rapidocr_paddle import RapidOCR
from tqdm import tqdm
from ultralytics import YOLO

from surya.input.processing import (
    get_total_splits,
    prepare_image_detection,
    split_image,
)
from surya.model.detection.model import EfficientViTForSemanticSegmentation
from surya.postprocessing.affinity import get_vertical_lines
from surya.postprocessing.heatmap import get_and_clean_boxes
from surya.postprocessing.util import xyxy2xyxyxyxy
from surya.schema import PolygonBox, TextDetectionResult
from surya.settings import settings
from surya.util.parallel import FakeExecutor


def get_batch_size():
    batch_size = settings.DETECTOR_BATCH_SIZE
    if batch_size is None:
        batch_size = 8
        if settings.TORCH_DEVICE_MODEL == "mps":
            batch_size = 8
        if settings.TORCH_DEVICE_MODEL == "cuda":
            batch_size = 36
    return batch_size


def pad_to_batch_size(tensor, batch_size):
    current_batch_size = tensor.shape[0]
    if current_batch_size >= batch_size:
        return tensor

    pad_size = batch_size - current_batch_size
    padding = (0, 0) * (tensor.dim() - 1) + (0, pad_size)

    return F.pad(tensor, padding, mode='constant', value=0)


def batch_detection(
    images: List, model: EfficientViTForSemanticSegmentation, processor, batch_size=None, static_cache=False
) -> Generator[Tuple[List[List[np.ndarray]], List[Tuple[int, int]]], None, None]:
    assert all([isinstance(image, Image.Image) for image in images])
    if batch_size is None:
        batch_size = get_batch_size()
    heatmap_count = model.config.num_labels

    orig_sizes = [image.size for image in images]
    splits_per_image = [get_total_splits(size, processor) for size in orig_sizes]

    batches = []
    current_batch_size = 0
    current_batch = []
    for i in range(len(images)):
        if current_batch_size + splits_per_image[i] > batch_size:
            if len(current_batch) > 0:
                batches.append(current_batch)
            current_batch = []
            current_batch_size = 0
        current_batch.append(i)
        current_batch_size += splits_per_image[i]

    if len(current_batch) > 0:
        batches.append(current_batch)

    for batch_idx in tqdm(range(len(batches)), desc="Detecting bboxes"):
        batch_image_idxs = batches[batch_idx]
        batch_images = [images[j].convert("RGB") for j in batch_image_idxs]

        split_index = []
        split_heights = []
        image_splits = []
        for image_idx, image in enumerate(batch_images):
            image_parts, split_height = split_image(image, processor)
            image_splits.extend(image_parts)
            split_index.extend([image_idx] * len(image_parts))
            split_heights.extend(split_height)

        image_splits = [prepare_image_detection(image, processor) for image in image_splits]
        # Batch images in dim 0
        batch = torch.stack(image_splits, dim=0).to(model.dtype).to(model.device)
        if static_cache:
            batch = pad_to_batch_size(batch, batch_size)

        with torch.inference_mode():
            pred = model(pixel_values=batch)

        logits = pred.logits
        correct_shape = [processor.size["height"], processor.size["width"]]
        current_shape = list(logits.shape[2:])
        if current_shape != correct_shape:
            logits = F.interpolate(logits, size=correct_shape, mode='bilinear', align_corners=False)

        logits = logits.cpu().detach().numpy().astype(np.float32)
        preds = []
        for i, (idx, height) in enumerate(zip(split_index, split_heights)):
            # If our current prediction length is below the image idx, that means we have a new image
            # Otherwise, we need to add to the current image
            if len(preds) <= idx:
                preds.append([logits[i][k] for k in range(heatmap_count)])
            else:
                heatmaps = preds[idx]
                pred_heatmaps = [logits[i][k] for k in range(heatmap_count)]

                if height < processor.size["height"]:
                    # Cut off padding to get original height
                    pred_heatmaps = [pred_heatmap[:height, :] for pred_heatmap in pred_heatmaps]

                for k in range(heatmap_count):
                    heatmaps[k] = np.vstack([heatmaps[k], pred_heatmaps[k]])
                preds[idx] = heatmaps

        yield preds, [orig_sizes[j] for j in batch_image_idxs]


def parallel_get_lines(preds, orig_sizes, include_maps=False):
    heatmap, affinity_map = preds
    heat_img, aff_img = None, None
    if include_maps:
        heat_img = Image.fromarray((heatmap * 255).astype(np.uint8))
        aff_img = Image.fromarray((affinity_map * 255).astype(np.uint8))
    affinity_size = list(reversed(affinity_map.shape))
    heatmap_size = list(reversed(heatmap.shape))
    bboxes = get_and_clean_boxes(heatmap, heatmap_size, orig_sizes)
    vertical_lines = get_vertical_lines(affinity_map, affinity_size, orig_sizes)

    result = TextDetectionResult(
        bboxes=bboxes,
        vertical_lines=vertical_lines,
        heatmap=heat_img,
        affinity_map=aff_img,
        image_bbox=[0, 0, orig_sizes[0], orig_sizes[1]],
    )
    return result


def batch_text_detection(
    images: List, model, processor, batch_size=None, include_maps=False
) -> List[TextDetectionResult]:
    detection_generator = batch_detection(
        images, model, processor, batch_size=batch_size, static_cache=settings.DETECTOR_STATIC_CACHE
    )

    postprocessing_futures = []
    max_workers = min(settings.DETECTOR_POSTPROCESSING_CPU_WORKERS, len(images))
    parallelize = not settings.IN_STREAMLIT and len(images) >= settings.DETECTOR_MIN_PARALLEL_THRESH
    executor = ThreadPoolExecutor if parallelize else FakeExecutor
    with executor(max_workers=max_workers) as e:
        for preds, orig_sizes in detection_generator:
            for pred, orig_size in zip(preds, orig_sizes):
                postprocessing_futures.append(e.submit(parallel_get_lines, pred, orig_size, include_maps))

    return [future.result() for future in postprocessing_futures]


def text_detection_yolo(images: List, model, batch_size=None, include_maps=False) -> List[TextDetectionResult]:
    results = []
    for image in tqdm(images, desc="Detecting bboxes"):
        if isinstance(model, YOLO):
            preds = model.predict(
                image,
                conf=settings.TEXT_DETECTOR_CONF,
                max_det=settings.TEXT_DETECTOR_MAX_DET,
                task='obb',
                verbose=False,
                save=False,
                imgsz=settings.TEXT_DETECTOR_IMGSZ,
            )

            bboxes = []
            for pred in preds:
                # Convert polygon coordinates to integer type
                polygons = pred.obb.xyxyxyxy.cpu().numpy().astype(int)
                confs = pred.obb.conf.tolist()
                for polygon, conf in zip(polygons, confs):
                    bboxes.append(PolygonBox(polygon=polygon, confidence=conf))
        elif isinstance(model, RapidOCR):
            preds, elapse = model(image, use_cls=False, use_rec=False, use_cuda=True)
            bboxes = []
            for pred in preds:
                # bboxes.append(
                #     PolygonBox(polygon=np.array(pred[0], dtype=np.int32).tolist(), text=pred[1], confidence=pred[2])
                # )
                bboxes.append(PolygonBox(polygon=pred, text='x', confidence=1.0))
        else:
            raise ValueError(f"Unknown model type: {type(model)}")

        vertical_lines = []
        heat_img = None
        affinity_map = None
        result = TextDetectionResult(
            bboxes=bboxes,
            vertical_lines=vertical_lines,
            heatmap=heat_img,
            affinity_map=affinity_map,
            image_bbox=[0, 0, image.width, image.height],
        )
        results.append(result)
    return results
