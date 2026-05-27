"""
Sweep nearest-prototype rejection thresholds for open-set predictions.

This script does not rerun inference. It loads one or more detailed
predictions_*.json files, remaps each prediction with:

    distance > threshold -> predicted_class = -1
    distance <= threshold -> predicted_class = nearest_class

Then it reuses the same IoU matching convention and metric implementation used
by EvaluationPipeline. The goal is to compare operating points quickly before
changing the training or inference code.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from copy import deepcopy
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pipeline.evaluation import EvaluationPipeline
from utils.bbox_utils import BBoxUtils
from utils.openset_metrics import compute_ten_metrics


UNKNOWN_GT_ID = 999
IOU_THRESHOLD = 0.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline threshold sweep for open-set detailed predictions."
    )
    parser.add_argument(
        "--detailed_json",
        type=Path,
        nargs="+",
        required=True,
        help="One or more predictions_*.json files.",
    )
    parser.add_argument(
        "--names",
        type=str,
        nargs="*",
        default=None,
        help="Optional display names aligned with --detailed_json.",
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="*",
        default=[0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30],
        help="Reject if distance > threshold.",
    )
    parser.add_argument(
        "--gt_dir",
        type=Path,
        default=PROJECT_ROOT / "data/inference/stage2/labels",
        help="GT YOLO labels directory.",
    )
    parser.add_argument(
        "--train_dir",
        type=Path,
        default=PROJECT_ROOT / "data/processed/stage_2/train",
        help="Known class train directory; subdirectory names define known IDs.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=PROJECT_ROOT / "results/stage_2/threshold_sweeps",
        help="Directory for CSV/JSON outputs.",
    )
    parser.add_argument(
        "--write_json",
        action="store_true",
        help="Also write re-thresholded predictions JSON files for every threshold.",
    )
    return parser.parse_args()


def load_known_ids(train_dir: Path) -> List[int]:
    if not train_dir.exists():
        raise FileNotFoundError(f"train_dir does not exist: {train_dir}")
    return sorted(
        int(path.name)
        for path in train_dir.iterdir()
        if path.is_dir() and path.name.isdigit()
    )


def load_gt(gt_dir: Path) -> Dict[str, List[Tuple[int, List[float]]]]:
    if not gt_dir.exists():
        raise FileNotFoundError(f"gt_dir does not exist: {gt_dir}")
    gt_map: Dict[str, List[Tuple[int, List[float]]]] = {}
    for txt_file in sorted(gt_dir.glob("*.txt")):
        try:
            boxes = BBoxUtils.get_bboxes_array_from_file(txt_file)
            gt_map[txt_file.stem] = [
                (int(b[0]), list(b[1:5]))
                for b in boxes
                if len(b) >= 5
            ]
        except Exception:
            gt_map[txt_file.stem] = []
    return gt_map


def yolo_iou(bbox1: Iterable[float], bbox2: Iterable[float]) -> float:
    x1c, y1c, w1, h1 = [float(v) for v in bbox1]
    x2c, y2c, w2, h2 = [float(v) for v in bbox2]
    x1_min, y1_min = x1c - w1 / 2, y1c - h1 / 2
    x1_max, y1_max = x1c + w1 / 2, y1c + h1 / 2
    x2_min, y2_min = x2c - w2 / 2, y2c - h2 / 2
    x2_max, y2_max = x2c + w2 / 2, y2c + h2 / 2
    inter_w = max(0.0, min(x1_max, x2_max) - max(x1_min, x2_min))
    inter_h = max(0.0, min(y1_max, y2_max) - max(y1_min, y2_min))
    inter = inter_w * inter_h
    union = w1 * h1 + w2 * h2 - inter
    return inter / union if union > 0 else 0.0


def bbox_from_prediction(sample: dict) -> List[float]:
    bbox = sample.get("bbox", {})
    if isinstance(bbox, dict):
        return [
            float(bbox.get("xc", 0.0)),
            float(bbox.get("yc", 0.0)),
            float(bbox.get("w", 0.0)),
            float(bbox.get("h", 0.0)),
        ]
    if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
        return [float(v) for v in bbox[:4]]
    return [0.0, 0.0, 0.0, 0.0]


def prediction_distance(sample: dict) -> float:
    value = sample.get("distance", sample.get("min_dist", 0.0))
    if isinstance(value, list):
        value = value[0] if value else 0.0
    return float(value or 0.0)


def top1_from_top3(sample: dict) -> Optional[int]:
    top3 = sample.get("top3", [])
    if not top3:
        return None
    first = top3[0]
    if isinstance(first, dict):
        for key in ("class", "class_id", "class_name", "id"):
            if key in first:
                try:
                    return int(first[key])
                except Exception:
                    return None
    if isinstance(first, (list, tuple)) and first:
        try:
            return int(first[0])
        except Exception:
            return None
    try:
        return int(first)
    except Exception:
        return None


def nearest_class(sample: dict) -> int:
    for key in ("nearest_class", "nearest_class_id"):
        if sample.get(key) is not None:
            try:
                return int(sample[key])
            except Exception:
                pass
    from_top3 = top1_from_top3(sample)
    if from_top3 is not None:
        return from_top3
    pred = int(sample.get("predicted_class", -1))
    if pred != -1:
        return pred
    return -1


def class_at_threshold(sample: dict, threshold: float) -> int:
    if prediction_distance(sample) > threshold:
        return -1
    return nearest_class(sample)


def group_predictions(predictions: List[dict]) -> Dict[str, List[dict]]:
    grouped: Dict[str, List[dict]] = {}
    for sample in predictions:
        stem = Path(sample.get("image_name", "")).stem
        if stem:
            grouped.setdefault(stem, []).append(sample)
    return grouped


def match_once(
    predictions: List[dict],
    gt_map: Dict[str, List[Tuple[int, List[float]]]],
) -> Tuple[List[dict], List[dict], List[Tuple[int, List[float]]]]:
    grouped = group_predictions(predictions)
    matched: List[dict] = []
    unmatched_predictions: List[dict] = []
    unmatched_gts: List[Tuple[int, List[float]]] = []

    for stem, gt_list in gt_map.items():
        stem_preds = grouped.get(stem, [])
        if not stem_preds:
            unmatched_gts.extend(gt_list)
            continue
        if not gt_list:
            # Match EvaluationPipeline exactly: predictions on images without
            # GT boxes are ignored by current open-set evaluation.
            continue

        pred_bboxes = [bbox_from_prediction(sample) for sample in stem_preds]
        iou_matrix = np.zeros((len(stem_preds), len(gt_list)), dtype=np.float64)
        for i, pred_bbox in enumerate(pred_bboxes):
            for j, (_, gt_bbox) in enumerate(gt_list):
                iou_matrix[i, j] = yolo_iou(pred_bbox, gt_bbox)

        row_ind, col_ind = EvaluationPipeline._hungarian_match(-iou_matrix)
        used_gts = set()
        for pred_idx, gt_idx in zip(row_ind, col_ind):
            pred_idx = int(pred_idx)
            gt_idx = int(gt_idx)
            best_iou = float(iou_matrix[pred_idx, gt_idx])
            if best_iou >= IOU_THRESHOLD:
                gt_cls, gt_bbox = gt_list[gt_idx]
                matched.append(
                    {
                        "sample": stem_preds[pred_idx],
                        "gt_class": int(gt_cls),
                        "pred_bbox": pred_bboxes[pred_idx],
                        "gt_bbox": gt_bbox,
                        "distance": prediction_distance(stem_preds[pred_idx]),
                        "iou": best_iou,
                    }
                )
                used_gts.add(gt_idx)
            else:
                unmatched_predictions.append(stem_preds[pred_idx])

        for idx, gt in enumerate(gt_list):
            if idx not in used_gts:
                unmatched_gts.append(gt)

    return matched, unmatched_predictions, unmatched_gts


def evaluate_threshold(
    matched: List[dict],
    unmatched_predictions: List[dict],
    unmatched_gts: List[Tuple[int, List[float]]],
    known_class_ids: List[int],
    threshold: float,
) -> dict:
    matched_pairs = [
        (
            class_at_threshold(row["sample"], threshold),
            int(row["gt_class"]),
            row["pred_bbox"],
            row["gt_bbox"],
        )
        for row in matched
    ]
    unmatched_preds = [
        (
            class_at_threshold(sample, threshold),
            bbox_from_prediction(sample),
        )
        for sample in unmatched_predictions
    ]
    results = compute_ten_metrics(
        matched_pairs=matched_pairs,
        unmatched_preds=unmatched_preds,
        unmatched_gts=unmatched_gts,
        known_class_ids=known_class_ids,
        UNKNOWN_GT_ID=UNKNOWN_GT_ID,
        IOU_THRESHOLD=IOU_THRESHOLD,
    )
    return results


def write_rethresholded_json(
    input_data: dict,
    threshold: float,
    output_path: Path,
) -> None:
    output_data = deepcopy(input_data)
    for sample in output_data.get("predictions", []):
        pred_cls = class_at_threshold(sample, threshold)
        sample["predicted_class"] = int(pred_cls)
        sample["is_unknown"] = pred_cls == -1
    summary = output_data.setdefault("summary", {})
    summary["rejection_threshold"] = float(threshold)
    summary["rethresholded_by"] = Path(__file__).name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)


def metric_subset(results: dict) -> dict:
    keys = ["Macro-F1", "FRR", "URR", "PS", "RS", "PR", "RR", "Acc"]
    return {key: results.get(key) for key in keys}


def safe_float(value) -> float:
    if value is None:
        return float("nan")
    try:
        return float(value)
    except Exception:
        return float("nan")


def print_best(rows: List[dict], name: str) -> None:
    print(f"\n[{name}] best operating points")
    for metric in ["Acc", "Macro-F1", "RR", "PR", "URR"]:
        best = max(rows, key=lambda row: safe_float(row.get(metric)))
        print(
            f"  best {metric:<8} thr={best['threshold']:.3f} "
            f"Acc={best['Acc']:.4f} Macro-F1={best['Macro-F1']:.4f} "
            f"PR={best['PR']:.4f} RR={best['RR']:.4f} "
            f"PS={best['PS']:.4f} RS={best['RS']:.4f} "
            f"FRR={best['FRR']:.4f} URR={best['URR']:.4f}"
        )


def save_csv(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run",
        "threshold",
        "Macro-F1",
        "FRR",
        "URR",
        "PS",
        "RS",
        "PR",
        "RR",
        "Acc",
        "total_matched",
        "unmatched_preds",
        "unmatched_gts",
        "detailed_json",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_one(
    detailed_json: Path,
    run_name: str,
    thresholds: List[float],
    gt_map: Dict[str, List[Tuple[int, List[float]]]],
    known_class_ids: List[int],
    output_dir: Path,
    write_json: bool,
) -> List[dict]:
    with detailed_json.open(encoding="utf-8") as f:
        input_data = json.load(f)
    predictions = input_data.get("predictions", [])
    if not predictions:
        raise ValueError(f"No predictions found in {detailed_json}")

    matched, unmatched_predictions, unmatched_gts = match_once(predictions, gt_map)
    rows: List[dict] = []
    print(
        f"\n=== {run_name} ===\n"
        f"json={detailed_json}\n"
        f"matched={len(matched)} unmatched_preds={len(unmatched_predictions)} "
        f"unmatched_gts={len(unmatched_gts)}"
    )
    print("thr    Acc     MacroF1  PS      RS      PR      RR      FRR     URR")

    for threshold in thresholds:
        results = evaluate_threshold(
            matched=matched,
            unmatched_predictions=unmatched_predictions,
            unmatched_gts=unmatched_gts,
            known_class_ids=known_class_ids,
            threshold=threshold,
        )
        row = {
            "run": run_name,
            "threshold": float(threshold),
            **metric_subset(results),
            "total_matched": len(matched),
            "unmatched_preds": len(unmatched_predictions),
            "unmatched_gts": len(unmatched_gts),
            "detailed_json": str(detailed_json),
        }
        rows.append(row)
        print(
            f"{threshold:>4.2f}  "
            f"{row['Acc']:.4f}  {row['Macro-F1']:.4f}  "
            f"{row['PS']:.4f}  {row['RS']:.4f}  "
            f"{row['PR']:.4f}  {row['RR']:.4f}  "
            f"{row['FRR']:.4f}  {row['URR']:.4f}"
        )

        if write_json:
            out_json = (
                output_dir
                / "rethresholded_json"
                / run_name
                / f"{detailed_json.stem}_thr{threshold:.3f}.json"
            )
            write_rethresholded_json(input_data, threshold, out_json)

    print_best(rows, run_name)
    return rows


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.names is not None and len(args.names) not in {0, len(args.detailed_json)}:
        raise ValueError("--names must be omitted or match --detailed_json length")

    known_class_ids = load_known_ids(args.train_dir)
    gt_map = load_gt(args.gt_dir)

    all_rows: List[dict] = []
    for idx, detailed_json in enumerate(args.detailed_json):
        run_name = (
            args.names[idx]
            if args.names and idx < len(args.names)
            else detailed_json.parent.parent.name
        )
        rows = run_one(
            detailed_json=detailed_json,
            run_name=run_name,
            thresholds=args.thresholds,
            gt_map=gt_map,
            known_class_ids=known_class_ids,
            output_dir=args.output_dir,
            write_json=args.write_json,
        )
        all_rows.extend(rows)

    csv_path = args.output_dir / "threshold_sweep_metrics.csv"
    save_csv(csv_path, all_rows)

    summary_path = args.output_dir / "threshold_sweep_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "gt_dir": str(args.gt_dir),
                "train_dir": str(args.train_dir),
                "known_class_ids": known_class_ids,
                "thresholds": args.thresholds,
                "rows": all_rows,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("\nOutputs:")
    print(f"  CSV: {csv_path}")
    print(f"  Summary: {summary_path}")


if __name__ == "__main__":
    main()
