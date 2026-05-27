import numpy as np
from utils.bbox_utils import BBoxUtils
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
from utils.helpers import get_files, get_text_files

class MetricsCalculator:
    """
    Calculates object detection metrics (precision, recall, F1) given bounding box predictions and ground truth in YOLO format.
    """

    def __init__(self, iou_threshold=0.5):
        """
        Initializes the MetricsCalculator.

        Args:
            iou_threshold (float, optional): The Intersection over Union (IoU) threshold for considering a detection a true positive. Defaults to 0.5.
        """
        self.iou_threshold = iou_threshold

    def _box_iou(self, box1, box2):
        """
        Calculates the Intersection over Union (IoU) of two bounding boxes.

        Args:
            box1 (list): A bounding box in YOLO format [c, x_c, y_c, w, h].
            box2 (list): A bounding box in YOLO format [c, x_c, y_c, w, h].

        Returns:
            float: The IoU of the two boxes.
        """
        _, x1_c, y1_c, w1, h1 = box1
        _, x2_c, y2_c, w2, h2 = box2

        # Convert center coordinates to corner coordinates
        x1_min = x1_c - w1 / 2
        y1_min = y1_c - h1 / 2
        x1_max = x1_c + w1 / 2
        y1_max = y1_c + h1 / 2

        x2_min = x2_c - w2 / 2
        y2_min = y2_c - h2 / 2
        x2_max = x2_c + w2 / 2
        y2_max = y2_c + h2 / 2

        # Calculate intersection area
        x_left = max(x1_min, x2_min)
        y_top = max(y1_min, y2_min)
        x_right = min(x1_max, x2_max)
        y_bottom = min(y1_max, y2_max)

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection_area = (x_right - x_left) * (y_bottom - y_top)

        # Calculate union area
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - intersection_area

        return intersection_area / union_area

    def _match_detections(self, predictions, ground_truths):
        """
        Matches predictions to ground truth boxes based on IoU (standard greedy matching).
        No class constraint is applied here - matching is purely geometric.

        Args:
            predictions (list of list): A list of predicted bounding boxes in YOLO format [[c, x_c, y_c, w, h], ...].
            ground_truths (list of list): A list of ground truth bounding boxes in YOLO format [[c, x_c, y_c, w, h], ...].

        Returns:
            list of tuples: A list of matched pairs (pred_idx, gt_idx) where IoU >= iou_threshold.
        """
        matches = []
        used_gt_indices = set()

        for pred_idx, pred_box in enumerate(predictions):
            best_iou = 0.0
            best_gt_idx = None

            for gt_idx, gt_box in enumerate(ground_truths):
                if gt_idx not in used_gt_indices:
                    iou = self._box_iou(pred_box, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx

            if best_iou >= self.iou_threshold:
                matches.append((pred_idx, best_gt_idx))
                used_gt_indices.add(best_gt_idx)

        return matches

    def calculate_class_wise_metrics(self, predictions, ground_truths):
        """
        Calculates class-wise precision, recall, and F1 score.

        Args:
            predictions (list of list): A list of predicted bounding boxes in YOLO format [[c, x_c, y_c, w, h], ...].
            ground_truths (list of list): A list of ground truth bounding boxes in YOLO format [[c, x_c, y_c, w, h], ...].

        Returns:
            dict: A dictionary where keys are class labels and values are dictionaries
                  containing 'precision', 'recall', and 'f1' for each class.
                  Returns empty dict if there are no ground truths.
        """

        class_metrics = {}
        if not ground_truths:
            return {}

        classes = set(int(gt_box[0]) for gt_box in ground_truths)

        for c in classes:
            class_predictions = [p for p in predictions if int(p[0]) == c]
            class_ground_truths = [gt for gt in ground_truths if int(gt[0]) == c]

            tp = fp = fn = 0
            if class_ground_truths:
                matches = self._match_detections(class_predictions, class_ground_truths)
                tp = len(matches)
                fp = len(class_predictions) - tp
                fn = len(class_ground_truths) - tp

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            class_metrics[c] = {'precision': precision, 'recall': recall, 'f1': f1}

        return class_metrics

    def calculate_overall_metrics(self, predictions, ground_truths, method='macro'):
        """
        Calculates overall precision, recall, and F1 score using either micro or macro averaging.

        Args:
            predictions (list of list): A list of predicted bounding boxes in YOLO format [[c, x_c, y_c, w, h], ...].
            ground_truths (list of list): A list of ground truth bounding boxes in YOLO format [[c, x_c, y_c, w, h], ...].
            method (str, optional): The averaging method ('micro' or 'macro'). Defaults to 'macro'.

        Returns:
            dict: A dictionary containing 'precision', 'recall', and 'f1'.
        """
        if not ground_truths:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}

        if method not in ['micro', 'macro']:
            raise ValueError("Method must be either 'micro' or 'macro'")

        if method == 'micro':
            tp = 0
            fp = 0
            fn = 0
            matches = self._match_detections(predictions, ground_truths)
            tp = len(matches)
            fp = len(predictions) - tp
            fn = len(ground_truths) - tp

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            return {'precision': precision, 'recall': recall, 'f1': f1}
        else:  # macro
            class_metrics = self.calculate_class_wise_metrics(predictions, ground_truths)
            if not class_metrics:
                return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
            precisions = [metrics['precision'] for metrics in class_metrics.values()]
            recalls = [metrics['recall'] for metrics in class_metrics.values()]
            f1_scores = [metrics['f1'] for metrics in class_metrics.values()]

            precision = sum(precisions) / len(precisions) if precisions else 0.0
            recall = sum(recalls) / len(recalls) if recalls else 0.0
            f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
            return {'precision': precision, 'recall': recall, 'f1': f1}

    def calculate_class_wise_recall(self, predictions, ground_truths):
        """
        Calculates class-wise recall.

        Args:
            predictions (list of list): Predicted bounding boxes.
            ground_truths (list of list): Ground truth bounding boxes.

        Returns:
            dict: Class-wise recall values.
        """
        class_metrics = self.calculate_class_wise_metrics(predictions, ground_truths)
        return {c: metrics['recall'] for c, metrics in class_metrics.items()} if class_metrics else {}

    def calculate_class_wise_precision(self, predictions, ground_truths):
        """
        Calculates class-wise precision.

        Args:
            predictions (list of list): Predicted bounding boxes.
            ground_truths (list of list): Ground truth bounding boxes.

        Returns:
            dict: Class-wise precision values.
        """
        class_metrics = self.calculate_class_wise_metrics(predictions, ground_truths)
        return {c: metrics['precision'] for c, metrics in class_metrics.items()} if class_metrics else {}

    def calculate_class_wise_f1(self, predictions, ground_truths):
        """
        Calculates class-wise F1 score.

        Args:
            predictions (list of list): Predicted bounding boxes.
            ground_truths (list of list): Ground truth bounding boxes.

        Returns:
            dict: Class-wise F1 score values.
        """
        class_metrics = self.calculate_class_wise_metrics(predictions, ground_truths)
        return {c: metrics['f1'] for c, metrics in class_metrics.items()} if class_metrics else {}

    def calculate_overall_recall(self, predictions, ground_truths, method='macro'):
        """
        Calculates overall recall.

        Args:
            predictions (list of list): Predicted bounding boxes.
            ground_truths (list of list): Ground truth bounding boxes.
            method (str, optional): Averaging method ('micro' or 'macro').

        Returns:
            float: Overall recall.
        """
        return self.calculate_overall_metrics(predictions, ground_truths, method)['recall']

    def calculate_overall_precision(self, predictions, ground_truths, method='macro'):
        """
        Calculates overall precision.

        Args:
            predictions (list of list): Predicted bounding boxes.
            ground_truths (list of list): Ground truth bounding boxes.
            method (str, optional): Averaging method ('micro' or 'macro').

        Returns:
            float: Overall precision.
        """
        return self.calculate_overall_metrics(predictions, ground_truths, method)['precision']

    def calculate_overall_f1(self, predictions, ground_truths, method='macro'):
        """
        Calculates overall F1 score.

        Args:
            predictions (list of list): Predicted bounding boxes.
            ground_truths (list of list): Ground truth bounding boxes.
            method (str, optional): Averaging method ('micro' or 'macro').

        Returns:
            float: Overall F1 score.
        """
        return self.calculate_overall_metrics(predictions, ground_truths, method)['f1']
    
    def calculate_class_wise_metrics_for_dataset_dir(self, predictions_dir, ground_truths_dir):
        """
        Calculates class-wise metrics for all files in the given directories.

        Args:
            predictions_dir (str): Directory containing prediction files.
            ground_truths_dir (str): Directory containing ground truth files.

        Returns:
            dict: Class-wise metrics for each file.
        """
        predictions_files = get_text_files(predictions_dir)
        ground_truths_files = get_text_files(ground_truths_dir)

        if len(predictions_files) != len(ground_truths_files):
            raise ValueError("Number of prediction files and ground truth files must match.")

        metrics = {}
        for pred_file, gt_file in zip(predictions_files, ground_truths_files):
            # check if the filenames are the same
            if pred_file.name != gt_file.name:
                raise ValueError(f"Prediction file {pred_file.name} does not match ground truth file {gt_file.name}.")
            
            predictions = BBoxUtils.get_bboxes_array_from_file(pred_file)
            ground_truths = BBoxUtils.get_bboxes_array_from_file(gt_file)
            metrics[pred_file.name] = self.calculate_class_wise_metrics(predictions, ground_truths)

        return metrics
    
    def calculate_overall_metrics_for_dataset_dir(
        self,
        predictions_dir,
        ground_truths_dir,
        method='micro',
        class_aware=False,
        valid_class_ids=None,
        ignore_predictions_on_excluded_gt=False,
    ):
        """
        Calculates overall metrics for all files in the given directories.

        For 'micro': aggregates TP/FP/FN across all files, then computes P/R/F1.
        For 'macro': averages per-sheet P/R/F1 (legacy behavior).
        When class_aware=True, TP requires both IoU >= threshold AND matching class.
        When valid_class_ids is provided, GT outside the set can be excluded from the
        closed-set metric; predictions that overlap excluded GT can optionally be ignored.

        Args:
            predictions_dir (str): Directory containing prediction files.
            ground_truths_dir (str): Directory containing ground truth files.
            method (str, optional): Averaging method ('micro' or 'macro').
            class_aware (bool, optional): If True, check class after IoU matching.
                                          For class-aware detection/classification (Stage2).
                                          Defaults to False (class-agnostic, Stage1).
            valid_class_ids (Iterable[int], optional): If provided, only these GT classes
                                          participate in evaluation.
            ignore_predictions_on_excluded_gt (bool, optional): If True and valid_class_ids
                                          is set, predictions overlapping excluded GT are
                                          ignored instead of counting as FP.

        Returns:
            dict: Overall metrics for each file (sheet_wise) and global aggregated metrics.
        """
        predictions_files = get_text_files(predictions_dir)
        ground_truths_files = get_text_files(ground_truths_dir)

        if len(predictions_files) != len(ground_truths_files):
            raise ValueError("Number of prediction files and ground truth files must match.")

        sheet_wise = {}
        total_tp = total_fp = total_fn = 0

        for pred_file, gt_file in zip(predictions_files, ground_truths_files):
            if pred_file.name != gt_file.name:
                raise ValueError(f"Prediction file {pred_file.name} does not match ground truth file {gt_file.name}.")

            predictions = BBoxUtils.get_bboxes_array_from_file(pred_file)
            ground_truths = BBoxUtils.get_bboxes_array_from_file(gt_file)

            if class_aware:
                allowed_ids = None if valid_class_ids is None else {int(cid) for cid in valid_class_ids}
                if allowed_ids is not None:
                    eval_ground_truths = [gt for gt in ground_truths if int(gt[0]) in allowed_ids]
                    excluded_ground_truths = [gt for gt in ground_truths if int(gt[0]) not in allowed_ids]
                else:
                    eval_ground_truths = list(ground_truths)
                    excluded_ground_truths = []

                if allowed_ids is not None and len(eval_ground_truths) == 0:
                    tp = fp = fn = 0
                    total_tp += tp
                    total_fp += fp
                    total_fn += fn
                    if method == 'micro':
                        sheet_wise[pred_file.name] = {
                            'precision': 0.0,
                            'recall': 0.0,
                            'f1': 0.0,
                            'tp': 0,
                            'fp': 0,
                            'fn': 0,
                        }
                    else:
                        sheet_wise[pred_file.name] = {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
                    continue

                # class-aware 严格匹配：类别错了算 FP，GT 仍然保留为 FN 候选
                used_gt = set()

                tp = fp = 0
                for pred_idx, pred_box in enumerate(predictions):
                    # class_id == -1 → unknown/拒识，直接跳过，
                    # 不计入 TP/FP，也不占用 GT（GT 后续会计入 FN）
                    if int(pred_box[0]) == -1:
                        continue

                    if ignore_predictions_on_excluded_gt and excluded_ground_truths:
                        best_excluded_iou = 0.0
                        for gt_box in excluded_ground_truths:
                            best_excluded_iou = max(best_excluded_iou, self._box_iou(pred_box, gt_box))
                        if best_excluded_iou >= self.iou_threshold:
                            continue

                    best_iou = 0.0
                    best_gt_idx = None
                    for gt_idx, gt_box in enumerate(eval_ground_truths):
                        if gt_idx in used_gt:
                            continue
                        if int(pred_box[0]) != int(gt_box[0]):
                            continue
                        iou = self._box_iou(pred_box, gt_box)
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_idx = gt_idx
                    if best_iou >= self.iou_threshold and best_gt_idx is not None:
                        used_gt.add(best_gt_idx)
                        tp += 1
                    else:
                        fp += 1
                # FN = 所有 eval_ground_truths 中没有任何已知预测（class_id != -1）成功匹配到的
                fn = len(eval_ground_truths) - len(used_gt)
            else:
                # Class-agnostic: only IoU matching (for object detection / Stage1)
                matches = self._match_detections(predictions, ground_truths)
                tp = len(matches)
                used_gt = set(m[1] for m in matches)
                fp = len(predictions) - tp
                fn = len(ground_truths) - len(used_gt)

            total_tp += tp
            total_fp += fp
            total_fn += fn

            if method == 'micro':
                p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
                sheet_wise[pred_file.name] = {'precision': p, 'recall': r, 'f1': f, 'tp': tp, 'fp': fp, 'fn': fn}
            else:
                om = self.calculate_overall_metrics(predictions, ground_truths, method='macro')
                sheet_wise[pred_file.name] = om

        # Global micro aggregation
        global_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        global_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        global_f = 2 * global_p * global_r / (global_p + global_r) if (global_p + global_r) > 0 else 0.0

        return {
            'sheet_wise': sheet_wise,
            'overall': {
                'precision': global_p,
                'recall': global_r,
                'f1': global_f,
                'tp': total_tp,
                'fp': total_fp,
                'fn': total_fn,
            }
        }
   
    
   


if __name__ == "__main__":

    gt_file = Path(r"C:\Users\mgupta70.ASURITE\Dropbox (ASU)\ASU\PhD\Courses\Github_projects\PID-Final-Cursor\data\raw\labels_class_agnostic\5.txt")
    pred_file = Path(r"C:\Users\mgupta70.ASURITE\Dropbox (ASU)\ASU\PhD\Courses\Github_projects\PID-Final-Cursor\results\5.txt")
    iou_threshold = 0.5
    metrics_calculator = MetricsCalculator(iou_threshold=iou_threshold)
    # Load predictions and ground truths from files
    predictions = BBoxUtils.get_bboxes_array_from_file(pred_file)
    # drop last column (confidence score) if it exists
    new_predictions = []
    for pred in predictions:
        if len(pred) > 5:
            new_predictions.append(pred[:-1])
        else:
            new_predictions.append(pred)
    predictions = new_predictions
    ground_truths = BBoxUtils.get_bboxes_array_from_file(gt_file)
    class_wise_metrics = metrics_calculator.calculate_class_wise_metrics(predictions, ground_truths)
    overall_metrics = metrics_calculator.calculate_overall_metrics(predictions, ground_truths)
    print("Class-wise Metrics:", class_wise_metrics)
    print("Overall Metrics:", overall_metrics)
