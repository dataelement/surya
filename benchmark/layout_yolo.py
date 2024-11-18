import argparse
import collections
import copy
import json
import os
import time
from enum import Enum
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from PIL import Image, ImageFile

from surya.postprocessing.visualization import visualize_bbox

ImageFile.LOAD_TRUNCATED_IMAGES = True
from tabulate import tabulate
from tqdm import tqdm

from surya.benchmark.metrics import precision_recall_v2
from surya.input.processing import convert_if_not_rgb
from surya.layout import batch_layout_detection, batch_layout_detection_yolo
from surya.model.layout.model import load_model as load_surya_layout_model
from surya.model.layout.model import load_processor as load_surya_layout_processor
from surya.model.layout.yolo_model import load_model as load_layout_yolo_model
from surya.postprocessing.heatmap import draw_bboxes_on_image
from surya.settings import settings


class ModelType(str, Enum):
    DOCLAYOUT_YOLO = "doclayout_yolo"
    ELEM_YOLO = "elem_yolo"
    SURYA = "surya"


class DatasetType(str, Enum):
    ELEM = "elem"
    DOCLAYNET = "doclaynet"


def get_model_and_processor(model_type: ModelType):
    """Get model and processor based on model type"""
    if model_type == ModelType.DOCLAYOUT_YOLO:
        return load_layout_yolo_model(), None
    elif model_type == ModelType.ELEM_YOLO:
        return load_layout_yolo_model(), None
    elif model_type == ModelType.SURYA:
        return load_surya_layout_model(), load_surya_layout_processor()


def get_label_alignment(dataset_type: DatasetType, model_type: ModelType):
    """Get label alignment based on dataset and model type"""
    alignments = {
        DatasetType.ELEM: {
            ModelType.DOCLAYOUT_YOLO: {  # elem layout to doclayout_yolo
                "Image": [["Picture"], ["figure"]],
                "Table": [["Table"], ["table"]],
                "Text": [["Text"], ["plain text"]],
                "Title": [["Title"], ["title"]],
            },
            ModelType.SURYA: {  # elem layout to surya
                "Image": [["Picture"], ["Picture", "Figure"]],
                "Table": [["Table"], ["Table"]],
                "Text": [["Text"], ["Text"]],
                "Title": [["Title"], ["Section-header", "Title"]],
            },
            ModelType.ELEM_YOLO: {  # elem layout to elem_yolo
                "Image": [["Picture"], ["Picture", "Figure"]],
                "Table": [["Table"], ["Table"]],
                "Text": [["Text"], ["Text"]],
                "Title": [["Title"], ["Section-header", "Title"]],
            },
        },
        DatasetType.DOCLAYNET: {
            ModelType.SURYA: {  # doclaynet to surya
                "Image": [["Picture"], ["Picture"]],
                "Table": [["Table"], ["Table"]],
                "Text": [["Text"], ["Text"]],
                "Title": [["Section-header", "Title"], ["Section-header", "Title"]],
            },
            ModelType.ELEM_YOLO: {  # elem layout to elem_yolo
                "Image": [["Picture"], ["Picture", "Figure"]],
                "Table": [["Table"], ["Table"]],
                "Text": [["Text"], ["Text"]],
                "Title": [["Title", "Section-header"], ["Section-header", "Title"]],
            },
            ModelType.DOCLAYOUT_YOLO: {  # elem layout to doclayout_yolo
                "Image": [["Picture"], ["figure"]],
                "Table": [["Table"], ["table"]],
                "Text": [["Text"], ["plain text"]],
                "Title": [["Title", "Section-header"], ["title"]],
            },
        },
    }
    return alignments[dataset_type][model_type]


def get_dataset_id2label(dataset_type: DatasetType) -> Dict[int, str]:
    """Get id2label mapping based on dataset type"""
    mappings = {
        DatasetType.ELEM: {
            1: 'Seal',
            2: 'Picture',
            3: 'Title',
            4: 'Text',
            5: 'Table',
            6: 'Page-header',
            7: 'Page-number',
            8: 'Page-footer',
        },
        DatasetType.DOCLAYNET: {
            0: 'Caption',
            1: 'Footnote',
            2: 'Formula',
            3: 'List-item',
            4: 'Page-footer',
            5: 'Page-header',
            6: 'Picture',
            7: 'Section-header',
            8: 'Table',
            9: 'Text',
            10: 'Title',
        },
    }
    return mappings[dataset_type]


def load_layout_dataset(dataset_path: str, dataset_type: DatasetType, max_samples: int) -> pd.DataFrame:
    """
    Yolo format dataset is stored in a directory with the following structure:
    dataset_path/
        images/
            *.jpg/png/etc
        labels/
            *.txt
        train.txt
        val.txt
    """
    # elem layout id2name
    id2label = get_dataset_id2label(dataset_type)

    with open(Path(dataset_path) / 'val.txt', 'r') as f:
        lines = [line.strip() for line in f]

    data = []
    for idx, line in enumerate(tqdm(lines[:max_samples], desc="Loading dataset")):
        row = dict(id=idx)
        image_path = Path(dataset_path) / f'{line}'
        label_path = Path(dataset_path) / 'labels' / f'{Path(line).stem}.txt'
        row['image'] = Image.open(image_path)
        width, height = row['image'].size

        bboxes = []
        category_ids = []
        labels = []
        with open(label_path, 'r') as f:
            for line in f:
                cls_id, *n_bbox = map(float, line.split())
                xyxy = [n_bbox[0] * width, n_bbox[1] * height, n_bbox[4] * width, n_bbox[5] * height]
                bboxes.append(xyxy)
                category_ids.append(int(cls_id))
                labels.append(id2label[int(cls_id)])

        row['bboxes'] = bboxes
        row['category_ids'] = category_ids
        row['labels'] = labels
        data.append(row)

    return pd.DataFrame(data, index=range(len(data)))


def main():
    parser = argparse.ArgumentParser(description="Benchmark layout models.")
    parser.add_argument(
        "--results_dir",
        type=str,
        help="Path to results directory",
        default=os.path.join(settings.RESULT_DIR, "benchmark"),
    )
    parser.add_argument("--max", type=int, help="Maximum number of images to run benchmark on.", default=1000)
    parser.add_argument("--debug", action="store_true", help="Run in debug mode.", default=False)
    parser.add_argument(
        "--model_type",
        type=str,
        choices=[e.value for e in ModelType],
        default=ModelType.DOCLAYOUT_YOLO.value,
        help="Type of model to use",
    )
    parser.add_argument(
        "--dataset_type",
        type=str,
        choices=[e.value for e in DatasetType],
        default=DatasetType.ELEM.value,
        help="Type of dataset to use",
    )
    parser.add_argument("--dataset_path", type=str, help="Path to dataset", default=None)
    args = parser.parse_args()

    # Convert string arguments to enum types
    model_type = ModelType(args.model_type)
    dataset_type = DatasetType(args.dataset_type)

    # Set dataset path if not provided
    if dataset_type == DatasetType.ELEM:
        dataset_path = settings.ELEM_LAYOUT_BENCH_DATASET_NAME
    elif dataset_type == DatasetType.DOCLAYNET:
        dataset_path = settings.DOCLAYNET_BENCH_DATASET_PATH
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    pathname = f"layout_bench_{model_type.value}_{dataset_type.value}"

    # Load dataset
    dataset = load_layout_dataset(dataset_path, dataset_type, max_samples=args.max)
    images = list(dataset["image"])
    images = convert_if_not_rgb(images)

    # Load model and processor
    model, processor = get_model_and_processor(model_type)

    start = time.time()

    if model_type == ModelType.DOCLAYOUT_YOLO:
        layout_predictions = batch_layout_detection_yolo(images=images, model=model)
    elif model_type == ModelType.ELEM_YOLO:
        layout_predictions = batch_layout_detection_yolo(images=images, model=model)
    else:
        layout_predictions = batch_layout_detection(images=images, model=model, processor=processor)

    surya_time = time.time() - start

    folder_name = os.path.basename(pathname).split(".")[0]
    result_path = os.path.join(args.results_dir, folder_name)
    os.makedirs(result_path, exist_ok=True)

    # Get label alignment based on dataset and model type
    label_alignment = get_label_alignment(dataset_type, model_type)

    page_metrics = collections.OrderedDict()
    for idx, pred in enumerate(layout_predictions):
        row = dataset.iloc[idx]
        all_correct_bboxes = []
        all_correct_labels = []
        all_pred_bboxes = []
        all_pred_labels = []
        all_pred_scores = []
        page_results = {}
        for label_name in label_alignment:
            correct_cats, surya_cats = label_alignment[label_name]
            correct_bboxes = [b for b, l in zip(row["bboxes"], row["labels"]) if l in correct_cats]
            all_correct_bboxes.extend(correct_bboxes)
            all_correct_labels.extend([l for l in row["labels"] if l in correct_cats])
            pred_bboxes = [b.bbox for b in pred.bboxes if b.label in surya_cats]
            all_pred_bboxes.extend(pred_bboxes)
            all_pred_labels.extend([l.label for l in pred.bboxes if l.label in surya_cats])
            all_pred_scores.extend([b.confidence for b in pred.bboxes if b.label in surya_cats])
            metrics = precision_recall_v2(pred_bboxes, correct_bboxes)
            # metrics = precision_recall_v2([], [])
            weight = len(correct_bboxes)
            metrics["weight"] = weight
            page_results[label_name] = metrics

        page_metrics[idx] = page_results

        if args.debug:
            gt_image = visualize_bbox(
                image_path=copy.deepcopy(images[idx]),
                bboxes=all_correct_bboxes,
                classes=all_correct_labels,
                scores=[1] * len(all_correct_bboxes),
            )
            pred_image = visualize_bbox(
                image_path=copy.deepcopy(images[idx]),
                bboxes=all_pred_bboxes,
                classes=all_pred_labels,
                scores=all_pred_scores,
            )
            # concat gt and pred images
            concat_image = np.concatenate([gt_image, pred_image], axis=1)
            Image.fromarray(concat_image).save(os.path.join(result_path, f"{idx}_layout.png"))

    mean_metrics = collections.defaultdict(dict)
    layout_types = sorted(page_metrics[0].keys())
    metric_types = sorted(page_metrics[0][layout_types[0]].keys())
    metric_types.remove("weight")
    for l in layout_types:
        for m in metric_types:
            metric = []
            total = 0
            for page in page_metrics:
                metric.append(page_metrics[page][l][m] * page_metrics[page][l]["weight"])
                total += page_metrics[page][l]["weight"]

            value = sum(metric)
            if value > 0:
                value /= total
            mean_metrics[l][m] = value

    out_data = {"time": surya_time, "metrics": mean_metrics, "page_metrics": page_metrics}

    with open(os.path.join(result_path, "results.json"), "w+") as f:
        json.dump(out_data, f, indent=4)

    table_headers = ["Layout Type"] + metric_types
    table_data = []
    for layout_type in layout_types:
        table_data.append([layout_type] + [f"{mean_metrics[layout_type][m]:.5f}" for m in metric_types])

    print(tabulate(table_data, headers=table_headers, tablefmt="markdown"))
    print(f"Took {surya_time / len(images):.5f} seconds per image, and {surya_time:.5f} seconds total.")
    print(
        "Precision and recall are over the mutual coverage of the detected boxes and the ground truth boxes at a .5 threshold."
    )
    print(f"Wrote results to {result_path}")


if __name__ == "__main__":
    main()
