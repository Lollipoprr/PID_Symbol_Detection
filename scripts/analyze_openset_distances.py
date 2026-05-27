"""
Analyze nearest-prototype distance distributions for open-set predictions.

The script reuses the same IoU matching convention as the evaluation pipeline:
predictions are matched to GT boxes per image, then matched samples are split by
GT class into known classes and unknown class 999. It reports how much the
nearest-distance distributions overlap and saves plots/tables for threshold
diagnosis.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pipeline.evaluation import EvaluationPipeline
from utils.bbox_utils import BBoxUtils


UNKNOWN_GT_ID = 999
IOU_THRESHOLD = 0.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze known/unknown nearest-distance distributions."
    )
    parser.add_argument(
        "--detailed_json",
        type=Path,
        default=None,
        help="Path to predictions_*.json. Defaults to newest JSON under --run_dir.",
    )
    parser.add_argument(
        "--run_dir",
        type=Path,
        default=PROJECT_ROOT / "results/stage_2/svg_structure_v2_best",
        help="Run directory containing detailed_results/predictions_*.json.",
    )
    parser.add_argument(
        "--gt_dir",
        type=Path,
        default=PROJECT_ROOT / "data/inference/stage2/labels",
        help="GT YOLO labels directory. Unknown GT class is 999.",
    )
    parser.add_argument(
        "--train_dir",
        type=Path,
        default=PROJECT_ROOT / "data/processed/stage_2/train",
        help="Known-class train directory; subdirectory names define known IDs.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <run_dir>/distance_analysis.",
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="*",
        default=[0.06, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30],
        help="Thresholds to evaluate as distance > threshold => reject.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=60,
        help="Histogram bin count.",
    )
    return parser.parse_args()


def find_latest_json(run_dir: Path) -> Path:
    detailed_dir = run_dir / "detailed_results"
    files = sorted(detailed_dir.glob("predictions_*.json"), key=lambda p: p.stat().st_mtime)
    if not files:
        raise FileNotFoundError(f"No predictions_*.json found in {detailed_dir}")
    return files[-1]


def load_known_ids(train_dir: Path) -> List[int]:
    if not train_dir.exists():
        raise FileNotFoundError(f"train_dir does not exist: {train_dir}")
    return sorted(
        int(p.name) for p in train_dir.iterdir()
        if p.is_dir() and p.name.isdigit()
    )


def load_gt(gt_dir: Path) -> Dict[str, List[Tuple[int, List[float]]]]:
    if not gt_dir.exists():
        raise FileNotFoundError(f"gt_dir does not exist: {gt_dir}")
    gt_map: Dict[str, List[Tuple[int, List[float]]]] = {}
    for txt_file in sorted(gt_dir.glob("*.txt")):
        try:
            boxes = BBoxUtils.get_bboxes_array_from_file(txt_file)
            gt_map[txt_file.stem] = [(int(b[0]), list(b[1:5])) for b in boxes if len(b) >= 5]
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


def group_predictions(predictions: List[dict]) -> Dict[str, List[dict]]:
    by_stem: Dict[str, List[dict]] = {}
    for sample in predictions:
        stem = Path(sample.get("image_name", "")).stem
        if stem:
            by_stem.setdefault(stem, []).append(sample)
    return by_stem


def match_predictions(
    predictions: List[dict],
    gt_map: Dict[str, List[Tuple[int, List[float]]]],
) -> Tuple[List[dict], List[dict], List[Tuple[int, List[float]]]]:
    by_stem = group_predictions(predictions)
    matched: List[dict] = []
    unmatched_predictions: List[dict] = []
    unmatched_gts: List[Tuple[int, List[float]]] = []

    for stem, gt_list in gt_map.items():
        stem_preds = by_stem.get(stem, [])
        if not stem_preds:
            unmatched_gts.extend(gt_list)
            continue
        if not gt_list:
            unmatched_predictions.extend(stem_preds)
            continue

        pred_bboxes = [bbox_from_prediction(s) for s in stem_preds]
        iou_matrix = np.zeros((len(stem_preds), len(gt_list)), dtype=np.float64)
        for i, pred_bbox in enumerate(pred_bboxes):
            for j, (_, gt_bbox) in enumerate(gt_list):
                iou_matrix[i, j] = yolo_iou(pred_bbox, gt_bbox)

        row_ind, col_ind = EvaluationPipeline._hungarian_match(-iou_matrix)
        used_preds = set()
        used_gts = set()
        for pred_idx, gt_idx in zip(row_ind, col_ind):
            best_iou = float(iou_matrix[pred_idx, gt_idx])
            used_preds.add(int(pred_idx))
            if best_iou >= IOU_THRESHOLD:
                gt_cls, gt_bbox = gt_list[int(gt_idx)]
                pred = stem_preds[int(pred_idx)]
                matched.append({
                    "image_name": pred.get("image_name", ""),
                    "predicted_class": int(pred.get("predicted_class", -1)),
                    "gt_class": int(gt_cls),
                    "distance": float(pred.get("distance", pred.get("min_dist", 0.0)) or 0.0),
                    "gap": float(pred.get("gap", 0.0) or 0.0),
                    "iou": best_iou,
                    "pred_bbox": pred_bboxes[int(pred_idx)],
                    "gt_bbox": gt_bbox,
                    "is_pred_unknown": bool(pred.get("is_unknown", False)),
                })
                used_gts.add(int(gt_idx))
            else:
                unmatched_predictions.append(stem_preds[int(pred_idx)])

        for idx, sample in enumerate(stem_preds):
            if idx not in used_preds:
                unmatched_predictions.append(sample)
        for idx, gt in enumerate(gt_list):
            if idx not in used_gts:
                unmatched_gts.append(gt)

    return matched, unmatched_predictions, unmatched_gts


def describe(values: List[float]) -> dict:
    if not values:
        return {"n": 0}
    arr = np.array(values, dtype=np.float64)
    out = {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }
    for q in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        out[f"p{q}"] = float(np.percentile(arr, q))
    return out


def threshold_table(
    known_distances: List[float],
    unknown_distances: List[float],
    thresholds: List[float],
) -> List[dict]:
    known = np.array(known_distances, dtype=np.float64)
    unknown = np.array(unknown_distances, dtype=np.float64)
    rows = []
    for thr in thresholds:
        known_reject = int((known > thr).sum()) if known.size else 0
        unknown_reject = int((unknown > thr).sum()) if unknown.size else 0
        total_reject = known_reject + unknown_reject
        rows.append({
            "threshold": float(thr),
            "known_accept_rate": float((known <= thr).mean()) if known.size else math.nan,
            "known_false_reject_rate": float((known > thr).mean()) if known.size else math.nan,
            "unknown_recall": float((unknown > thr).mean()) if unknown.size else math.nan,
            "rejection_precision": (
                float(unknown_reject / total_reject) if total_reject else math.nan
            ),
            "known_rejected": known_reject,
            "unknown_rejected": unknown_reject,
            "total_rejected": total_reject,
        })
    return rows


def save_csv(path: Path, rows: List[dict], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_plot(
    output_path: Path,
    known_distances: List[float],
    unknown_distances: List[float],
    thresholds: List[float],
    bins: int,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    all_dist = known_distances + unknown_distances
    if not all_dist:
        return

    max_x = min(max(all_dist), float(np.percentile(np.array(all_dist), 99.5)) * 1.2)
    hist_range = (0.0, max_x if max_x > 0 else 1.0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(
        known_distances,
        bins=bins,
        range=hist_range,
        alpha=0.65,
        density=True,
        label=f"GT known (n={len(known_distances)})",
        color="#2f6fbb",
    )
    axes[0].hist(
        unknown_distances,
        bins=bins,
        range=hist_range,
        alpha=0.60,
        density=True,
        label=f"GT unknown 999 (n={len(unknown_distances)})",
        color="#c74e3b",
    )
    for thr in thresholds:
        if hist_range[0] <= thr <= hist_range[1]:
            axes[0].axvline(thr, color="black", linewidth=0.8, alpha=0.25)
    axes[0].set_title("Nearest distance distribution")
    axes[0].set_xlabel("Nearest prototype distance")
    axes[0].set_ylabel("Density")
    axes[0].legend()
    axes[0].grid(True, alpha=0.25)

    rows = threshold_table(known_distances, unknown_distances, thresholds)
    xs = [r["threshold"] for r in rows]
    axes[1].plot(xs, [r["unknown_recall"] for r in rows], marker="o", label="Unknown recall")
    axes[1].plot(xs, [r["rejection_precision"] for r in rows], marker="o", label="Rejection precision")
    axes[1].plot(xs, [r["known_false_reject_rate"] for r in rows], marker="o", label="Known false reject")
    axes[1].set_title("Threshold trade-off")
    axes[1].set_xlabel("Reject if distance > threshold")
    axes[1].set_ylabel("Rate")
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    axes[1].grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    detailed_json = args.detailed_json or find_latest_json(args.run_dir)
    output_dir = args.output_dir or ((args.run_dir if args.run_dir else detailed_json.parent.parent) / "distance_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    with detailed_json.open(encoding="utf-8") as f:
        pred_data = json.load(f)
    predictions = pred_data.get("predictions", [])
    if not predictions:
        raise ValueError(f"No predictions found in {detailed_json}")

    known_ids = set(load_known_ids(args.train_dir))
    gt_map = load_gt(args.gt_dir)
    matched, unmatched_predictions, unmatched_gts = match_predictions(predictions, gt_map)

    known_rows = [r for r in matched if r["gt_class"] in known_ids]
    unknown_rows = [r for r in matched if r["gt_class"] == UNKNOWN_GT_ID]
    known_distances = [r["distance"] for r in known_rows]
    unknown_distances = [r["distance"] for r in unknown_rows]

    details_path = output_dir / f"{detailed_json.stem}_matched_distances.csv"
    save_csv(
        details_path,
        matched,
        [
            "image_name",
            "gt_class",
            "predicted_class",
            "distance",
            "gap",
            "iou",
            "is_pred_unknown",
            "pred_bbox",
            "gt_bbox",
        ],
    )

    thr_rows = threshold_table(known_distances, unknown_distances, args.thresholds)
    threshold_path = output_dir / f"{detailed_json.stem}_thresholds.csv"
    save_csv(
        threshold_path,
        thr_rows,
        [
            "threshold",
            "known_accept_rate",
            "known_false_reject_rate",
            "unknown_recall",
            "rejection_precision",
            "known_rejected",
            "unknown_rejected",
            "total_rejected",
        ],
    )

    plot_path = output_dir / f"{detailed_json.stem}_distance_hist.png"
    save_plot(plot_path, known_distances, unknown_distances, args.thresholds, args.bins)

    summary = {
        "detailed_json": str(detailed_json),
        "gt_dir": str(args.gt_dir),
        "train_dir": str(args.train_dir),
        "unknown_gt_id": UNKNOWN_GT_ID,
        "iou_threshold": IOU_THRESHOLD,
        "n_predictions": len(predictions),
        "n_gt": sum(len(v) for v in gt_map.values()),
        "n_matched": len(matched),
        "n_unmatched_predictions": len(unmatched_predictions),
        "n_unmatched_gts": len(unmatched_gts),
        "known_distance": describe(known_distances),
        "unknown_distance": describe(unknown_distances),
        "thresholds": thr_rows,
        "outputs": {
            "matched_csv": str(details_path),
            "threshold_csv": str(threshold_path),
            "plot": str(plot_path),
        },
    }

    summary_path = output_dir / f"{detailed_json.stem}_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("=" * 72)
    print(f"Input JSON: {detailed_json}")
    print(f"Matched: {len(matched)} | known={len(known_rows)} unknown={len(unknown_rows)}")
    print(f"Unmatched predictions: {len(unmatched_predictions)}")
    print(f"Unmatched GTs: {len(unmatched_gts)}")
    print("-" * 72)
    for name, stats in [("Known", summary["known_distance"]), ("Unknown", summary["unknown_distance"])]:
        if stats.get("n", 0) == 0:
            print(f"{name}: n=0")
            continue
        print(
            f"{name}: n={stats['n']} mean={stats['mean']:.4f} "
            f"p50={stats['p50']:.4f} p75={stats['p75']:.4f} "
            f"p90={stats['p90']:.4f} p95={stats['p95']:.4f}"
        )
    print("-" * 72)
    print("threshold  known_FRR  unknown_RR  reject_precision  rejected")
    for row in thr_rows:
        print(
            f"{row['threshold']:>9.3f}  "
            f"{row['known_false_reject_rate']:>9.4f}  "
            f"{row['unknown_recall']:>10.4f}  "
            f"{row['rejection_precision']:>16.4f}  "
            f"{row['total_rejected']:>8}"
        )
    print("-" * 72)
    print(f"Summary: {summary_path}")
    print(f"Matched CSV: {details_path}")
    print(f"Threshold CSV: {threshold_path}")
    print(f"Plot: {plot_path}")


if __name__ == "__main__":
    main()
