import argparse
import collections
import copy
import json
import os
import time
from statistics import median

import cv2
import datasets

from surya.benchmark.metrics import rank_accuracy
from surya.detection import batch_text_detection
from surya.input.processing import convert_if_not_rgb
from surya.layout import batch_layout_detection_yolo
from surya.model.detection.model import load_model as load_det_model
from surya.model.detection.model import load_processor as load_det_processor
from surya.model.layout.yolo_model import load_model as load_layout_model
from surya.model.layoutlmv3_order.model import load_model as load_order_model
from surya.ordering import batch_ordering, elem_batch_ordering
from surya.postprocessing.visualization import visualize_bbox

# from surya.layout
from surya.settings import settings


def main():
    parser = argparse.ArgumentParser(description="Benchmark surya reading order model.")
    parser.add_argument(
        "--results_dir",
        type=str,
        help="Path to JSON file with benchmark results.",
        default=os.path.join(settings.RESULT_DIR, "benchmark"),
    )
    parser.add_argument("--max", type=int, help="Maximum number of images to run benchmark on.", default=1)
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

    span_det_processor = load_det_processor()
    span_det_model = load_det_model()
    layout_model = load_layout_model()
    order_model = load_order_model()

    span_predictions = batch_text_detection(images, span_det_model, span_det_processor)
    layout_predictions = batch_layout_detection_yolo(images, layout_model, span_predictions)
    start = time.time()
    order_predictions = elem_batch_ordering(
        images=images,
        model=order_model,
        text_det_results=span_predictions,
        layout_results=layout_predictions,
    )
    surya_time = time.time() - start
    print(f"Took {surya_time:.2f} seconds to order.")

    if args.debug:
        for i, span_pred in enumerate(span_predictions):
            bboxes = [i.bbox for i in span_pred.bboxes]
            confs = [i.confidence for i in span_pred.bboxes]
            span_labels = [str(i) for i in range(len(bboxes))]
            img = visualize_bbox(images[i], bboxes, span_labels, confs)
            cv2.imwrite(os.path.join(result_path, f"{i}_spans.png"), img)
            print(f"Wrote {i}_spans.png")
        for i, order_pred in enumerate(order_predictions):
            img = visualize_bbox(
                images[i],
                [i.bbox for i in order_pred.bboxes],
                [str(i.position) for i in order_pred.bboxes],
            )
            cv2.imwrite(os.path.join(result_path, f"{i}_order.png"), img)
            print(f"Wrote {i}_order.png")

    # folder_name = os.path.basename(pathname).split(".")[0]
    # result_path = os.path.join(args.results_dir, folder_name)
    # os.makedirs(result_path, exist_ok=True)

    # page_metrics = collections.OrderedDict()
    # mean_accuracy = 0
    # for idx, order_pred in enumerate(order_predictions):
    #     row = dataset[idx]
    #     pred_labels = [str(l.position) for l in order_pred.bboxes]
    #     labels = row["labels"]
    #     accuracy = rank_accuracy(pred_labels, labels)
    #     mean_accuracy += accuracy
    #     page_results = {"accuracy": accuracy, "box_count": len(labels)}

    #     page_metrics[idx] = page_results

    # mean_accuracy /= len(order_predictions)

    # out_data = {"time": surya_time, "mean_accuracy": mean_accuracy, "page_metrics": page_metrics}

    # with open(os.path.join(result_path, "results.json"), "w+") as f:
    #     json.dump(out_data, f, indent=4)

    # print(f"Mean accuracy is {mean_accuracy:.2f}.")
    # print(f"Took {surya_time / len(images):.2f} seconds per image, and {surya_time:.1f} seconds total.")
    # print("Mean accuracy is the % of correct ranking pairs.")
    # print(f"Wrote results to {result_path}")


if __name__ == "__main__":
    main()
