from concurrent.futures import ProcessPoolExecutor
from functools import partial
from itertools import repeat

import numpy as np
import Polygon as plg

from surya.postprocessing.util import remove_overlapping_boxes, xyxy2xyxyxyxy


def intersection_area(box1, box2):
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    return (x_right - x_left) * (y_bottom - y_top)


def box_area(box):
    return (box[2] - box[0]) * (box[3] - box[1])


def calculate_iou(box1, box2, box1_only=False):
    intersection = intersection_area(box1, box2)
    union = box_area(box1)
    if not box1_only:
        union += box_area(box2) - intersection

    if union == 0:
        return 0
    return intersection / union


def match_boxes(preds, references):
    num_actual = len(references)
    num_predicted = len(preds)

    iou_matrix = np.zeros((num_actual, num_predicted))
    for i, actual in enumerate(references):
        for j, pred in enumerate(preds):
            iou_matrix[i, j] = calculate_iou(actual, pred, box1_only=True)

    sorted_indices = np.argsort(iou_matrix, axis=None)[::-1]
    sorted_ious = iou_matrix.flatten()[sorted_indices]
    actual_indices, predicted_indices = np.unravel_index(sorted_indices, iou_matrix.shape)

    assigned_actual = set()
    assigned_pred = set()

    matches = []
    for idx, iou in zip(zip(actual_indices, predicted_indices), sorted_ious):
        i, j = idx
        if i not in assigned_actual and j not in assigned_pred:
            iou_val = iou_matrix[i, j]
            if iou_val > 0.95:  # Account for rounding on box edges
                iou_val = 1.0
            matches.append((i, j, iou_val))
            assigned_actual.add(i)
            assigned_pred.add(j)

    unassigned_actual = set(range(num_actual)) - assigned_actual
    unassigned_pred = set(range(num_predicted)) - assigned_pred
    matches.extend([(i, None, -1.0) for i in unassigned_actual])
    matches.extend([(None, j, 0.0) for j in unassigned_pred])

    return matches


def penalized_iou_score(preds, references):
    matches = match_boxes(preds, references)
    iou = sum([match[2] for match in matches]) / len(matches)
    return iou


def intersection_pixels(box1, box2):
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return set()

    x_left, x_right = int(x_left), int(x_right)
    y_top, y_bottom = int(y_top), int(y_bottom)

    coords = np.meshgrid(np.arange(x_left, x_right), np.arange(y_top, y_bottom))
    pixels = set(zip(coords[0].flat, coords[1].flat))

    return pixels


def calculate_coverage(box, other_boxes, penalize_double=False):
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    if box_area == 0:
        return 0

    # find total coverage of the box
    covered_pixels = set()
    double_coverage = list()
    for other_box in other_boxes:
        ia = intersection_pixels(box, other_box)
        double_coverage.append(list(covered_pixels.intersection(ia)))
        covered_pixels = covered_pixels.union(ia)

    # Penalize double coverage - having multiple bboxes overlapping the same pixels
    double_coverage_penalty = len(double_coverage)
    if not penalize_double:
        double_coverage_penalty = 0
    covered_pixels_count = max(0, len(covered_pixels) - double_coverage_penalty)
    return covered_pixels_count / box_area


def calculate_coverage_fast(box, other_boxes, penalize_double=False):
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    if box_area == 0:
        return 0

    total_intersect = 0
    for other_box in other_boxes:
        total_intersect += intersection_area(box, other_box)

    return min(1, total_intersect / box_area)


def precision_recall(preds, references, threshold=0.5, workers=8, penalize_double=True):
    if len(references) == 0:
        return {
            "precision": 1,
            "recall": 1,
            "f1": 1,
        }

    if len(preds) == 0:
        return {
            "precision": 0,
            "recall": 0,
            "f1": 0,
        }

    # If we're not penalizing double coverage, we can use a faster calculation
    coverage_func = calculate_coverage_fast
    if penalize_double:
        coverage_func = calculate_coverage

    with ProcessPoolExecutor(max_workers=workers) as executor:
        precision_func = partial(coverage_func, penalize_double=penalize_double)
        precision_iou = executor.map(precision_func, preds, repeat(references))
        reference_iou = executor.map(coverage_func, references, repeat(preds))

    precision_classes = [1 if i > threshold else 0 for i in precision_iou]
    precision = sum(precision_classes) / len(precision_classes)

    recall_classes = [1 if i > threshold else 0 for i in reference_iou]
    recall = sum(recall_classes) / len(recall_classes)

    f1 = 0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def mean_coverage(preds, references):
    coverages = []

    for box1 in references:
        coverage = calculate_coverage(box1, preds)
        coverages.append(coverage)

    for box2 in preds:
        coverage = calculate_coverage(box2, references)
        coverages.append(coverage)

    # Calculate the average coverage over all comparisons
    if len(coverages) == 0:
        return 0
    coverage = sum(coverages) / len(coverages)
    return {"coverage": coverage}


def rank_accuracy(preds, references):
    # Preds and references need to be aligned so each position refers to the same bbox
    pairs = []
    for i, pred in enumerate(preds):
        for j, pred2 in enumerate(preds):
            if i == j:
                continue
            pairs.append((i, j, pred > pred2))

    # Find how many of the prediction rankings are correct
    correct = 0
    for i, ref in enumerate(references):
        for j, ref2 in enumerate(references):
            if (i, j, ref > ref2) in pairs:
                correct += 1

    return correct / len(pairs)


class MetricIou:
    def __init__(self, iou_thresh=0.5):
        self.iou_thresh = iou_thresh

    def __call__(self, gt_boxes_list, boxes_list):
        detMatched_list = []
        numDetCare_list = []
        numGtCare_list = []
        for i in range(len(gt_boxes_list)):
            gt_boxes = gt_boxes_list[i]
            boxes = boxes_list[i]
            detMatched, numDetCare, numGtCare = self.eval(gt_boxes, boxes)
            detMatched_list.append(detMatched)
            numDetCare_list.append(numDetCare)
            numGtCare_list.append(numGtCare)
        matchedSum = np.sum(np.array(detMatched_list))
        numGlobalCareDet = np.sum(np.array(numDetCare_list))
        numGlobalCareGt = np.sum(np.array(numGtCare_list))
        methodRecall = 0 if numGlobalCareGt == 0 else float(matchedSum) / numGlobalCareGt
        methodPrecision = 0 if numGlobalCareDet == 0 else float(matchedSum) / numGlobalCareDet
        methodHmean = (
            0
            if methodRecall + methodPrecision == 0
            else 2 * methodRecall * methodPrecision / (methodRecall + methodPrecision)
        )
        return methodPrecision, methodRecall, methodHmean

    def eval(self, gt_boxes, boxes):
        detMatched = 0
        numDetCare = 0
        numGtCare = 0
        if gt_boxes is None:
            return 0, 0, 0

        gtPols = []
        detPols = []
        detDontCarePolsNum = []
        iouMat = np.empty([1, 1])
        for i in range(len(gt_boxes)):
            gt_box = gt_boxes[i]
            gtPols.append(self.polygon_from_box(gt_box))

        if boxes is None:
            return 0, 0, len(gtPols)

        for box in boxes:
            detPol = self.polygon_from_box(box)
            detPols.append(detPol)

        if len(gtPols) > 0 and len(detPols) > 0:
            outputShape = [len(gtPols), len(detPols)]
            iouMat = np.empty(outputShape)
            gtRectMat = np.zeros(len(gtPols), np.int8)
            detRectMat = np.zeros(len(detPols), np.int8)
            pairs = []
            detMatchedNums = []
            for gtNum in range(len(gtPols)):
                for detNum in range(len(detPols)):
                    pG = gtPols[gtNum]
                    pD = detPols[detNum]
                    iouMat[gtNum, detNum] = self.get_intersection_over_union(pD, pG)

            for gtNum in range(len(gtPols)):
                for detNum in range(len(detPols)):
                    if gtRectMat[gtNum] == 0 and detRectMat[detNum] == 0 and detNum not in detDontCarePolsNum:
                        if iouMat[gtNum, detNum] > self.iou_thresh:
                            gtRectMat[gtNum] = 1
                            detRectMat[detNum] = 1
                            detMatched += 1
                            pairs.append({'gt': gtNum, 'det': detNum})
                            detMatchedNums.append(detNum)

        numGtCare = len(gtPols)
        numDetCare = len(detPols) - len(detDontCarePolsNum)
        return detMatched, numDetCare, numGtCare

    def get_intersection(self, pD, pG):
        pInt = pD & pG
        if len(pInt) == 0:
            return 0
        return pInt.area()

    def get_union(self, pD, pG):
        areaA = pD.area()
        areaB = pG.area()
        return areaA + areaB - self.get_intersection(pD, pG)

    def get_intersection_over_union(self, pD, pG):
        try:
            return self.get_intersection(pD, pG) / self.get_union(pD, pG)
        except Exception:
            return 0

    def polygon_from_box(self, box):
        resBoxes = np.empty([1, 8], dtype='int32')
        resBoxes[0, 0] = int(box[0][0])
        resBoxes[0, 4] = int(box[0][1])
        resBoxes[0, 1] = int(box[1][0])
        resBoxes[0, 5] = int(box[1][1])
        resBoxes[0, 2] = int(box[2][0])
        resBoxes[0, 6] = int(box[2][1])
        resBoxes[0, 3] = int(box[3][0])
        resBoxes[0, 7] = int(box[3][1])
        pointMat = resBoxes[0].reshape([2, 4]).T
        return plg.Polygon(pointMat)


def precision_recall_v2(preds, references, threshold=0.5):
    """
    more faster than v1
    """
    eval_iou = MetricIou(iou_thresh=threshold)
    if len(references) == 0:
        return {"precision": 1, "recall": 1, "f1": 1}
    if len(preds) == 0:
        return {"precision": 0, "recall": 0, "f1": 0}

    references = [xyxy2xyxyxyxy(ref) for ref in references]
    preds = [xyxy2xyxyxyxy(pred) for pred in preds]
    references = np.array(references).reshape(1, -1, 4, 2)
    preds = np.array(preds).reshape(1, -1, 4, 2)

    precision, recall, f1 = eval_iou(references, preds)
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}
