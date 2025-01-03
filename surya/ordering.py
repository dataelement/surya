import logging
import logging as logger
import os
import shutil
from collections import defaultdict
from copy import deepcopy
from statistics import median
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import LayoutLMv3ForTokenClassification

from surya.model.ordering.encoderdecoder import OrderVisionEncoderDecoderModel
from surya.postprocessing.ordering import (
    assign_spans_to_blocks,
    create_virtual_lines,
    merge_spans_to_lines,
    sort_spans_horizontally,
)
from surya.postprocessing.visualization import visualize_bbox
from surya.schema import (
    LayoutBox,
    LayoutResult,
    OrderBox,
    OrderResult,
    PolygonBox,
    TextDetectionResult,
    TextLine,
)
from surya.settings import settings
from surya.util.reading_order import boxes2inputs, parse_logits, prepare_inputs


def get_batch_size():
    batch_size = settings.ORDER_BATCH_SIZE
    if batch_size is None:
        batch_size = 8
        if settings.TORCH_DEVICE_MODEL == "mps":
            batch_size = 8
        if settings.TORCH_DEVICE_MODEL == "cuda":
            batch_size = 32
    return batch_size


def rank_elements(arr):
    enumerated_and_sorted = sorted(enumerate(arr), key=lambda x: x[1])
    rank = [0] * len(arr)

    for rank_value, (original_index, value) in enumerate(enumerated_and_sorted):
        rank[original_index] = rank_value

    return rank


def _clip_and_validate_box(
    left: float, top: float, right: float, bottom: float, page_width: float, page_height: float
) -> tuple[float, float, float, float]:
    """Clip box coordinates to page boundaries and validate the results."""
    # Clip coordinates
    left = max(0, min(left, page_width))
    right = max(0, min(right, page_width))
    top = max(0, min(top, page_height))
    bottom = max(0, min(bottom, page_height))

    # Log warnings for out-of-bounds coordinates
    if any([left < 0, right > page_width, top < 0, bottom > page_height]):
        logging.warning(
            f'Box coordinates out of bounds: '
            f'({left}, {top}, {right}, {bottom}) for page size {page_width}x{page_height}'
        )

    return left, top, right, bottom


def _scale_box(box: tuple[float, float, float, float], x_scale: float, y_scale: float) -> list[int]:
    """Scale and round box coordinates."""
    left, top, right, bottom = box
    return [round(left * x_scale), round(top * y_scale), round(right * x_scale), round(bottom * y_scale)]


def batch_ordering(
    images: List, bboxes: List[List[List[float]]], model: OrderVisionEncoderDecoderModel, processor, batch_size=None
) -> List[OrderResult]:
    assert all([isinstance(image, Image.Image) for image in images])
    assert len(images) == len(bboxes)
    if batch_size is None:
        batch_size = get_batch_size()

    output_order = []
    for i in tqdm(range(0, len(images), batch_size), desc="Finding reading order"):
        batch_bboxes = deepcopy(bboxes[i : i + batch_size])
        batch_images = images[i : i + batch_size]
        batch_images = [image.convert("RGB") for image in batch_images]  # also copies the images

        orig_sizes = [image.size for image in batch_images]
        model_inputs = processor(images=batch_images, boxes=batch_bboxes)

        batch_pixel_values = model_inputs["pixel_values"]
        batch_bboxes = model_inputs["input_boxes"]
        batch_bbox_mask = model_inputs["input_boxes_mask"]
        batch_bbox_counts = model_inputs["input_boxes_counts"]

        batch_bboxes = torch.from_numpy(np.array(batch_bboxes, dtype=np.int32)).to(model.device)
        batch_bbox_mask = torch.from_numpy(np.array(batch_bbox_mask, dtype=np.int32)).to(model.device)
        batch_pixel_values = torch.tensor(np.array(batch_pixel_values), dtype=model.dtype).to(model.device)
        batch_bbox_counts = torch.tensor(np.array(batch_bbox_counts), dtype=torch.long).to(model.device)

        token_count = 0
        past_key_values = None
        encoder_outputs = None
        batch_predictions = [[] for _ in range(len(batch_images))]
        done = torch.zeros(len(batch_images), dtype=torch.bool, device=model.device)

        with torch.inference_mode():
            while token_count < settings.ORDER_MAX_BOXES:
                return_dict = model(
                    pixel_values=batch_pixel_values,
                    decoder_input_boxes=batch_bboxes,
                    decoder_input_boxes_mask=batch_bbox_mask,
                    decoder_input_boxes_counts=batch_bbox_counts,
                    encoder_outputs=encoder_outputs,
                    past_key_values=past_key_values,
                )
                logits = return_dict["logits"].detach()

                last_tokens = []
                last_token_mask = []
                min_val = torch.finfo(model.dtype).min
                for j in range(logits.shape[0]):
                    label_count = batch_bbox_counts[j, 1] - batch_bbox_counts[j, 0] - 1  # Subtract 1 for the sep token
                    new_logits = logits[j, -1]
                    new_logits[batch_predictions[j]] = (
                        min_val  # Mask out already predicted tokens, we can only predict each token once
                    )
                    new_logits[label_count:] = min_val  # Mask out all logit positions above the number of bboxes
                    pred = int(torch.argmax(new_logits, dim=-1).item())

                    # Add one to avoid colliding with the 1000 height/width token for bboxes
                    last_tokens.append([[pred + processor.box_size["height"] + 1] * 4])
                    if len(batch_predictions[j]) == label_count - 1:  # Minus one since we're appending the final label
                        last_token_mask.append([0])
                        batch_predictions[j].append(pred)
                        done[j] = True
                    elif len(batch_predictions[j]) < label_count - 1:
                        last_token_mask.append([1])
                        batch_predictions[j].append(pred)  # Get rank prediction for given position
                    else:
                        last_token_mask.append([0])

                if done.all():
                    break

                past_key_values = return_dict["past_key_values"]
                encoder_outputs = (return_dict["encoder_last_hidden_state"],)

                batch_bboxes = torch.tensor(last_tokens, dtype=torch.long).to(model.device)
                token_bbox_mask = torch.tensor(last_token_mask, dtype=torch.long).to(model.device)
                batch_bbox_mask = torch.cat([batch_bbox_mask, token_bbox_mask], dim=1)
                token_count += 1

        for j, row_pred in enumerate(batch_predictions):
            row_bboxes = bboxes[i + j]
            assert len(row_pred) == len(
                row_bboxes
            ), f"Mismatch between logits and bboxes. Logits: {len(row_pred)}, Bboxes: {len(row_bboxes)}"

            orig_size = orig_sizes[j]
            ranks = [0] * len(row_bboxes)

            for box_idx in range(len(row_bboxes)):
                ranks[row_pred[box_idx]] = box_idx

            order_boxes = []
            for row_bbox, rank in zip(row_bboxes, ranks):
                order_box = OrderBox(
                    bbox=row_bbox,
                    position=rank,
                )
                order_boxes.append(order_box)

            result = OrderResult(
                bboxes=order_boxes,
                image_bbox=[0, 0, orig_size[0], orig_size[1]],
            )
            output_order.append(result)
    return output_order


def elem_batch_ordering(
    images: List,
    model: LayoutLMv3ForTokenClassification,
    text_det_results: List[TextDetectionResult],
    layout_results: List[LayoutResult],
    debug=False,
    batch_size=None,
) -> List[OrderResult]:
    """Process document layout and determine reading order of blocks."""
    assert all([isinstance(image, Image.Image) for image in images])
    assert len(images) == len(text_det_results)

    order_results = []

    if debug:
        debug_dir = "results/debug/ordering"
        if os.path.exists(debug_dir):
            shutil.rmtree(debug_dir)
        os.makedirs(debug_dir, exist_ok=True)

    for page_idx, layout_result in enumerate(layout_results):
        if debug:
            images[page_idx].save(f"{debug_dir}/{page_idx}.png")
            layout_boxes = []
            layout_labels = []
            for block in layout_result.bboxes:
                block_label = block.label
                layout_boxes.append(block.bbox)
                layout_labels.extend([block_label] * len(layout_boxes))
            img = visualize_bbox(images[page_idx], layout_boxes, layout_labels)
            cv2.imwrite(f"{debug_dir}/{page_idx}_layout.png", img)
            print(f"Saved layout to {debug_dir}/{page_idx}_layout.png")

        # 1. Split blocks into valid and invalid
        valid_blocks = [block for block in layout_result.bboxes]
        if not valid_blocks:
            continue

        # 2. Assign spans to blocks
        spans = text_det_results[page_idx].bboxes
        new_spans = []

        # process_spans
        # 删除高度或者宽度小于等于0或者置信度低于 0.05 的spans
        for span in spans:
            x1, y1, x2, y2 = span.bbox
            span_score = span.confidence
            if x2 - x1 <= 0 or y2 - y1 <= 0 or span_score <= 0.05:
                continue
            new_spans.append(span)
        logger.info(f"Discarded {len(spans) - len(new_spans)} spans with low confidence")

        # 删除 iou>0.9中置信度较低的那个
        discard_spans = set()
        for i, span in enumerate(new_spans):
            for j in range(i + 1, len(new_spans)):
                span2 = new_spans[j]
                iou = span.intersection_area(span2) / (span.area + span2.area - span.intersection_area(span2))
                if iou > 0.9:
                    if span.confidence < span2.confidence:
                        discard_spans.add(i)
                    else:
                        discard_spans.add(j)
        new_spans_2 = [span for i, span in enumerate(new_spans) if i not in discard_spans]
        # logger.info(f"Discarded {len(discard_spans)} spans with high iou")

        # 删除重复的 span
        new_spans_3 = []
        used_spans = []
        for span in new_spans_2:
            box_set = set(span.bbox)
            if box_set not in used_spans:
                new_spans_3.append(span)
                used_spans.append(box_set)
        # logger.info(f"Discarded {len(new_spans_2) - len(new_spans_3)} duplicate spans")

        block2spans = assign_spans_to_blocks(valid_blocks, new_spans_3)

        if debug:
            boxes = []
            labels = []
            for block_idx, block_spans in block2spans.items():
                block_label = valid_blocks[block_idx].label
                span_boxes = [span.bbox for span in block_spans]
                boxes.extend(span_boxes)
                labels.extend([block_label] * len(span_boxes))
            debug_image = visualize_bbox(images[page_idx], boxes, labels)
            cv2.imwrite(f"{debug_dir}/{page_idx}_spans.png", debug_image)
            print(f"Saved spans to {debug_dir}/{page_idx}_spans.png")

        # 3. Merge spans into lines for each block
        block2lines = {}
        _line_heights = []
        for block_idx, block_spans in block2spans.items():
            current_block = valid_blocks[block_idx]
            block_label = current_block.label
            if block_label in {'Table', 'Picture', 'Figure'}:
                block_lines = []
            else:
                block_lines = merge_spans_to_lines(block_spans)
                block_lines = sort_spans_horizontally(block_lines)
            block2lines[block_idx] = block_lines
            _line_heights.extend([line['bbox'][3] - line['bbox'][1] for line in block_lines])
        median_line_height = median(_line_heights) if _line_heights else 10

        # 4. Create virtual lines for table, picture, and figure blocks
        all_line_boxes: List[Dict] = []
        for block_idx, block_lines in block2lines.items():
            current_block = valid_blocks[block_idx]
            block_label = current_block.label
            if block_label in {'Table', 'Picture', 'Figure'} or not block_lines:
                block_lines = create_virtual_lines(
                    current_block.bbox, median_line_height, layout_result.image_bbox[2], layout_result.image_bbox[3]
                )
                block2lines[block_idx] = block_lines
            all_line_boxes.extend([line['bbox'] for line in block_lines])

        all_line_boxes.sort()

        if debug:
            debug_image = visualize_bbox(images[page_idx], all_line_boxes, ['all_lines'] * len(all_line_boxes))
            cv2.imwrite(f"{debug_dir}/{page_idx}_all_lines.png", debug_image)

        # 5. Prepare lines for reading order model
        page_w, page_h = layout_result.image_bbox[2:]

        x_scale = 1000.0 / page_w
        y_scale = 1000.0 / page_h
        boxes = []
        # logger.info(f"Scale: {x_scale}, {y_scale}, Boxes len: {len(page_line_list)}")
        for left, top, right, bottom in all_line_boxes:

            if left < 0:
                logger.warning(
                    f'left < 0, left: {left}, right: {right}, top: {top}, bottom: {bottom}, page_w: {page_w}, page_h: {page_h}'
                )  # noqa: E501
                left = 0
            if right > page_w:
                logger.warning(
                    f'right > page_w, left: {left}, right: {right}, top: {top}, bottom: {bottom}, page_w: {page_w}, page_h: {page_h}'
                )  # noqa: E501
                right = page_w
            if top < 0:
                logger.warning(
                    f'top < 0, left: {left}, right: {right}, top: {top}, bottom: {bottom}, page_w: {page_w}, page_h: {page_h}'
                )  # noqa: E501
                top = 0
            if bottom > page_h:
                logger.warning(
                    f'bottom > page_h, left: {left}, right: {right}, top: {top}, bottom: {bottom}, page_w: {page_w}, page_h: {page_h}'
                )  # noqa: E501
                bottom = page_h

            left = round(left * x_scale)
            top = round(top * y_scale)
            right = round(right * x_scale)
            bottom = round(bottom * y_scale)
            assert (
                1000 >= right >= left >= 0 and 1000 >= bottom >= top >= 0
            ), f'Invalid box. right: {right}, left: {left}, bottom: {bottom}, top: {top}'  # noqa: E126, E121
            boxes.append([left, top, right, bottom])

        # 6. Get reading order predictions
        inputs = boxes2inputs(boxes)
        inputs = prepare_inputs(inputs, model)
        logits = model(**inputs).logits.cpu().squeeze(0)
        line_predictions = parse_logits(logits, len(boxes))

        if debug:
            img = visualize_bbox(images[page_idx], all_line_boxes, list(map(str, line_predictions)))
            cv2.imwrite(f"{debug_dir}/{page_idx}_order_pred.png", img)

        # 7. Calculate block orders based on their lines
        box_order = [all_line_boxes[i] for i in line_predictions]
        block2order = defaultdict(list)
        for block_idx, block_lines in block2lines.items():
            for line in block_lines:
                line_box = line['bbox']
                block2order[block_idx].append(box_order.index(line_box))
        for block_idx, block_order in block2order.items():
            block2order[block_idx] = median(block_order)

        sorted_block2order = sorted(block2order.items(), key=lambda x: x[1])
        order_mapping = {i: idx for idx, (i, _) in enumerate(sorted_block2order)}

        # 8. Create final order result
        page_order_boxes = []

        # Add valid blocks with their calculated order
        for block_idx, block in enumerate(valid_blocks):
            page_order_boxes.append(OrderBox(bbox=block.bbox, position=order_mapping[block_idx]))

        # Create page result
        order_results.append(OrderResult(bboxes=page_order_boxes, image_bbox=[0, 0, page_w, page_h]))

    return order_results
