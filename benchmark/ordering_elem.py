import argparse
import collections
import copy
import json
import os
import time
from statistics import median

import cv2
import datasets
import numpy as np

from surya.benchmark.metrics import rank_accuracy
from surya.detection import batch_text_detection, text_detection_yolo
from surya.input.processing import convert_if_not_rgb
from surya.layout import (
    LayoutBox,
    LayoutResult,
    batch_layout_detection,
    batch_layout_detection_yolo,
)
from surya.model.detection.model import load_model as load_det_model
from surya.model.detection.model import load_processor as load_det_processor
from surya.model.layout.model import load_model as load_layout_model
from surya.model.layout.model import load_processor as load_layout_processor
from surya.model.layout.yolo_model import load_model as load_layout_model_yolo
from surya.model.layoutlmv3_order.model import load_model as load_order_model
from surya.model.text_det.model import load_pp_model
from surya.ordering import batch_ordering, elem_batch_ordering
from surya.postprocessing.util import xyxy2xyxyxyxy
from surya.postprocessing.visualization import visualize_bbox

# from surya.layout
from surya.settings import settings


def get_dummy_layout_result(dataset):
    results = []
    for i, row in enumerate(dataset):
        # breakpoint()

        image = row["image"]
        bboxes = row["bboxes"]
        polygons = [np.array(xyxy2xyxyxyxy(bbox), dtype=np.float32).reshape(4, 2).tolist() for bbox in bboxes]
        layout_boxes = [LayoutBox(polygon=polygon, confidence=1.0, label='Text') for polygon in polygons]
        results.append(
            LayoutResult(
                bboxes=layout_boxes, segmentation_map=None, heatmaps=None, image_bbox=[0, 0, image.width, image.height]
            )
        )
    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark surya reading order model.")
    parser.add_argument(
        "--results_dir",
        type=str,
        help="Path to JSON file with benchmark results.",
        default=os.path.join(settings.RESULT_DIR, "benchmark"),
    )
    parser.add_argument("--max", type=int, help="Maximum number of images to run benchmark on.", default=None)
    parser.add_argument("--debug", action="store_true", help="Run in debug mode.", default=False)
    args = parser.parse_args()

    pathname = "order_bench"
    folder_name = os.path.basename(pathname).split(".")[0]
    result_path = os.path.join(args.results_dir, folder_name)
    os.makedirs(result_path, exist_ok=True)

    # These have already been shuffled randomly, so sampling from the start is fine
    split = "train"
    if args.max is not None:
        split = f"train[:{args.max}]"
    dataset = datasets.load_dataset(settings.ORDER_BENCH_DATASET_NAME, split=split)
    images = list(dataset["image"])
    images = convert_if_not_rgb(images)
    bboxes = list(dataset["bboxes"])
    dummy_layout_predictions = get_dummy_layout_result(dataset)

    # span_predictions = batch_text_detection(images, load_det_model(), load_det_processor())
    span_predictions = text_detection_yolo(images, load_pp_model())

    start = time.time()
    order_predictions = elem_batch_ordering(
        images=images,
        model=load_order_model(),
        text_det_results=span_predictions,
        layout_results=dummy_layout_predictions,
    )
    surya_time = time.time() - start
    print(f"Took {surya_time:.2f} seconds to order.")

    if args.debug:
        for i, span_pred in enumerate(span_predictions):
            polys = [i.polygon for i in span_pred.bboxes]
            bboxes = [i.bbox for i in span_pred.bboxes]
            confs = [i.confidence for i in span_pred.bboxes]
            span_labels = [str(i) for i in range(len(bboxes))]

            # PIL to numpy array and convert RGB to BGR
            img = np.array(images[i])
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            for poly, label in zip(polys, span_labels):
                pts = np.array(poly).reshape((-1, 1, 2)).astype(np.int32)
                cv2.polylines(img, [pts], True, (0, 0, 255), 2)
                cv2.arrowedLine(img, tuple(pts[0][0]), tuple(pts[2][0]), (0, 255, 0), 2)

            cv2.imwrite(os.path.join(result_path, f"{i}_spans.png"), img)
            print(f"Saved {i}_spans.png to {result_path}")
        for i, order_pred in enumerate(order_predictions):
            img = visualize_bbox(
                images[i],
                [i.bbox for i in order_pred.bboxes],
                [str(i.position) for i in order_pred.bboxes],
            )
            cv2.imwrite(os.path.join(result_path, f"{i}_order.png"), img)
            print(f"Saved {i}_order.png to {result_path}")

    folder_name = os.path.basename(pathname).split(".")[0]
    result_path = os.path.join(args.results_dir, folder_name)
    os.makedirs(result_path, exist_ok=True)

    page_metrics = collections.OrderedDict()
    mean_accuracy = 0
    for idx, order_pred in enumerate(order_predictions):
        row = dataset[idx]

        gt_boxes = row["bboxes"]
        gt_order = row["labels"]

        pred_box_dict = {tuple(pred_box.bbox): pred_box.position for pred_box in order_pred.bboxes}
        pred_order = [pred_box_dict[tuple(gt_box)] for gt_box in gt_boxes]

        accuracy = rank_accuracy(pred_order, gt_order)
        mean_accuracy += accuracy
        page_results = {"accuracy": accuracy, "box_count": len(gt_order)}

        page_metrics[idx] = page_results

    mean_accuracy /= len(order_predictions)

    out_data = {"time": surya_time, "mean_accuracy": mean_accuracy, "page_metrics": page_metrics}

    with open(os.path.join(result_path, "results.json"), "w+") as f:
        json.dump(out_data, f, indent=4)

    print(f"Mean accuracy is {mean_accuracy:.2f}.")
    print(f"Took {surya_time / len(images):.2f} seconds per image, and {surya_time:.1f} seconds total.")
    print(f"Wrote results to {result_path}")


if __name__ == "__main__":
    main()
