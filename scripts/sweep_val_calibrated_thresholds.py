"""
Sweep class-adaptive thresholds calibrated from an independent validation split.

Workflow:
1. Load a trained stage-2 checkpoint.
2. Build class prototypes from train_dir.
3. Embed val_dir crops and estimate per-class thresholds from known distances.
4. Apply these thresholds to an existing predictions_*.json.
5. Reuse the same open-set evaluation logic to report metrics.

Unlike sweep_class_adaptive_thresholds.py, this script does not use test GT to
estimate thresholds. It is intended for the final, non-leaky calibration run.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
SCRIPT_ROOT = PROJECT_ROOT / "src/scripts"
for path in (SRC_ROOT, SCRIPT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from infer_svg_structure import (  # noqa: E402
    compute_single_modality_prototypes,
    get_image_files,
    load_model,
    make_transform,
)
from sweep_openset_thresholds import (  # noqa: E402
    bbox_from_prediction,
    load_gt,
    load_known_ids,
    match_once,
    metric_subset,
    nearest_class,
    prediction_score,
    print_best,
)
from utils.openset_metrics import compute_ten_metrics  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Val-calibrated class-adaptive threshold sweep."
    )
    parser.add_argument(
        "--detailed_json",
        type=Path,
        required=True,
        help="Path to test/inference predictions_*.json.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Checkpoint used to embed val crops and build train prototypes.",
    )
    parser.add_argument(
        "--train_dir",
        type=Path,
        default=PROJECT_ROOT / "data/processed/stage_2/train",
        help="Known-class train directory used to build prototypes.",
    )
    parser.add_argument(
        "--val_dir",
        type=Path,
        default=PROJECT_ROOT / "data/processed/stage_2/val",
        help="Independent validation split used for threshold calibration.",
    )
    parser.add_argument(
        "--gt_dir",
        type=Path,
        default=PROJECT_ROOT / "data/inference/stage2/labels",
        help="GT YOLO labels directory for test-time evaluation.",
    )
    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument(
        "--quantiles",
        type=float,
        nargs="*",
        default=[90, 92, 95, 97, 99],
        help="Per-class calibration quantiles.",
    )
    parser.add_argument(
        "--scales",
        type=float,
        nargs="*",
        default=[0.9, 1.0, 1.1, 1.2],
        help="Multipliers applied to per-class quantiles.",
    )
    parser.add_argument(
        "--floors",
        type=float,
        nargs="*",
        default=[0.04, 0.06, 0.08],
        help="Minimum threshold candidates.",
    )
    parser.add_argument(
        "--caps",
        type=float,
        nargs="*",
        default=[0.12, 0.15, 0.18, 0.20, 0.25, 0.30],
        help="Maximum threshold candidates. Use negative value for no cap.",
    )
    parser.add_argument(
        "--min_samples",
        type=int,
        default=5,
        help="Fallback to global threshold if a class has fewer samples.",
    )
    parser.add_argument(
        "--calib_mode",
        choices=["correct_known", "gt_known"],
        default="correct_known",
        help=(
            "correct_known: use val samples only when nearest prototype matches GT class; "
            "gt_known: use all val known samples and group by GT class."
        ),
    )
    parser.add_argument(
        "--score_field",
        choices=["distance"],
        default="distance",
        help="Score used for thresholding. Current val calibration supports distance mode.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Optional run name in outputs.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=PROJECT_ROOT / "results/stage_2/threshold_sweeps/val_calibrated",
        help="Output directory.",
    )
    parser.add_argument(
        "--write_best_json",
        action="store_true",
        help="Write best-acc and best-macro_f1 rethresholded JSON files.",
    )
    return parser.parse_args()


def embed_image_batch(model, image_paths: List[Path], transform, device: torch.device) -> torch.Tensor:
    tensors = []
    for path in image_paths:
        img = Image.open(path).convert("RGB")
        tensors.append(transform(img))
    batch = torch.stack(tensors).to(device)
    with torch.no_grad():
        out = model(batch)
    emb = out["photo_emb"] if isinstance(out, dict) else out
    return F.normalize(emb, p=2, dim=1).cpu()


def build_val_groups(
    model,
    prototypes: dict[int, torch.Tensor],
    val_dir: Path,
    transform,
    device: torch.device,
    batch_size: int,
    calib_mode: str,
) -> Dict[int, List[float]]:
    groups: Dict[int, List[float]] = {cid: [] for cid in sorted(prototypes.keys())}
    sorted_ids = sorted(prototypes.keys())
    proto_matrix = torch.cat([prototypes[cid] for cid in sorted_ids], dim=0).to(device)

    for class_dir in sorted(val_dir.iterdir()):
        if not class_dir.is_dir() or not class_dir.name.isdigit():
            continue
        gt_cls = int(class_dir.name)
        if gt_cls not in groups:
            continue
        image_paths = get_image_files(class_dir)
        if not image_paths:
            continue

        for start in range(0, len(image_paths), batch_size):
            chunk = image_paths[start:start + batch_size]
            embs = embed_image_batch(model, chunk, transform, device)
            dists = torch.cdist(embs.to(device), proto_matrix, p=2)
            min_dists, min_idx = dists.min(dim=1)
            for row_idx in range(len(chunk)):
                nearest_cls = int(sorted_ids[int(min_idx[row_idx].item())])
                min_dist = float(min_dists[row_idx].item())
                if calib_mode == "correct_known":
                    if nearest_cls == gt_cls:
                        groups[gt_cls].append(min_dist)
                elif calib_mode == "gt_known":
                    groups[gt_cls].append(min_dist)
                else:
                    raise ValueError(f"Unsupported calib_mode: {calib_mode}")

    return groups


def make_thresholds(
    groups: Dict[int, List[float]],
    known_ids: List[int],
    quantile: float,
    scale: float,
    floor: float,
    cap: Optional[float],
    min_samples: int,
) -> Tuple[Dict[int, float], float]:
    all_values = [dist for cid in known_ids for dist in groups.get(cid, [])]
    if not all_values:
        raise ValueError("No val calibration distances available")

    global_thr = float(np.percentile(np.asarray(all_values, dtype=np.float64), quantile))
    global_thr *= float(scale)
    global_thr = max(global_thr, floor)
    if cap is not None:
        global_thr = min(global_thr, cap)

    thresholds: Dict[int, float] = {}
    for cid in known_ids:
        values = groups.get(cid, [])
        if len(values) < min_samples:
            thresholds[cid] = global_thr
            continue
        thr = float(np.percentile(np.asarray(values, dtype=np.float64), quantile))
        thr *= float(scale)
        thr = max(thr, floor)
        if cap is not None:
            thr = min(thr, cap)
        thresholds[cid] = thr
    return thresholds, global_thr


def class_with_adaptive_threshold(sample: dict, thresholds: Dict[int, float], global_thr: float) -> int:
    near_cls = nearest_class(sample)
    threshold = thresholds.get(near_cls, global_thr)
    if prediction_score(sample, score_field="distance") > threshold:
        return -1
    return near_cls


def evaluate_adaptive(
    matched: List[dict],
    unmatched_predictions: List[dict],
    unmatched_gts: List[Tuple[int, List[float]]],
    known_class_ids: List[int],
    thresholds: Dict[int, float],
    global_thr: float,
) -> dict:
    matched_pairs = [
        (
            class_with_adaptive_threshold(row["sample"], thresholds, global_thr),
            int(row["gt_class"]),
            row["pred_bbox"],
            row["gt_bbox"],
        )
        for row in matched
    ]
    unmatched_preds = [
        (
            class_with_adaptive_threshold(sample, thresholds, global_thr),
            bbox_from_prediction(sample),
        )
        for sample in unmatched_predictions
    ]
    return compute_ten_metrics(
        matched_pairs=matched_pairs,
        unmatched_preds=unmatched_preds,
        unmatched_gts=unmatched_gts,
        known_class_ids=known_class_ids,
    )


def write_adaptive_json(
    input_data: dict,
    thresholds: Dict[int, float],
    global_thr: float,
    config: dict,
    output_path: Path,
) -> None:
    output_data = deepcopy(input_data)
    for sample in output_data.get("predictions", []):
        pred_cls = class_with_adaptive_threshold(sample, thresholds, global_thr)
        sample["predicted_class"] = int(pred_cls)
        sample["is_unknown"] = pred_cls == -1
        sample["adaptive_threshold"] = float(thresholds.get(nearest_class(sample), global_thr))
    summary = output_data.setdefault("summary", {})
    summary["rejection_strategy"] = "val_calibrated_class_adaptive_distance"
    summary["class_adaptive_thresholds"] = {str(k): float(v) for k, v in thresholds.items()}
    summary["global_fallback_threshold"] = float(global_thr)
    summary["adaptive_threshold_config"] = config
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)


def metric_row(
    run_name: str,
    config: dict,
    results: dict,
    thresholds: Dict[int, float],
    global_thr: float,
    detailed_json: Path,
) -> dict:
    values = list(thresholds.values())
    return {
        "run": run_name,
        "quantile": config["quantile"],
        "scale": config["scale"],
        "floor": config["floor"],
        "cap": config["cap"],
        "global_threshold": global_thr,
        "threshold_min": min(values),
        "threshold_p50": float(np.percentile(values, 50)),
        "threshold_max": max(values),
        **metric_subset(results),
        "detailed_json": str(detailed_json),
    }


def save_val_calibrated_csv(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run",
        "quantile",
        "scale",
        "floor",
        "cap",
        "global_threshold",
        "threshold_min",
        "threshold_p50",
        "threshold_max",
        "Macro-F1",
        "FRR",
        "URR",
        "PS",
        "RS",
        "PR",
        "RR",
        "Acc",
        "detailed_json",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt(row: dict) -> str:
    return (
        f"q={row['quantile']:g} scale={row['scale']:.2f} "
        f"floor={row['floor']:.2f} cap={row['cap']} "
        f"Acc={row['Acc']:.4f} Macro-F1={row['Macro-F1']:.4f} "
        f"PS={row['PS']:.4f} RS={row['RS']:.4f} "
        f"PR={row['PR']:.4f} RR={row['RR']:.4f} "
        f"FRR={row['FRR']:.4f} URR={row['URR']:.4f} "
        f"thr[p50={row['threshold_p50']:.4f}, max={row['threshold_max']:.4f}]"
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    run_name = args.name or args.detailed_json.parent.parent.name
    known_class_ids = load_known_ids(args.train_dir)
    gt_map = load_gt(args.gt_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = make_transform()
    model = load_model(args.checkpoint, args.embed_dim, device)
    prototypes, proto_stats = compute_single_modality_prototypes(
        model=model,
        data_dir=args.train_dir,
        transform=transform,
        device=device,
        batch_size=args.batch_size,
    )

    with args.detailed_json.open(encoding="utf-8") as f:
        input_data = json.load(f)
    predictions = input_data.get("predictions", [])
    if not predictions:
        raise ValueError(f"No predictions found in {args.detailed_json}")

    matched, unmatched_predictions, unmatched_gts = match_once(predictions, gt_map)
    groups = build_val_groups(
        model=model,
        prototypes=prototypes,
        val_dir=args.val_dir,
        transform=transform,
        device=device,
        batch_size=args.batch_size,
        calib_mode=args.calib_mode,
    )

    print(
        f"Run: {run_name}\n"
        f"checkpoint={args.checkpoint}\n"
        f"matched={len(matched)} unmatched_preds={len(unmatched_predictions)} unmatched_gts={len(unmatched_gts)}\n"
        f"val_calib_mode={args.calib_mode} classes_with_samples={sum(1 for v in groups.values() if v)} "
        f"total_val_samples={sum(len(v) for v in groups.values())}\n"
        f"prototype_p50={proto_stats['p50_intra']:.4f} prototype_p90={proto_stats['p90_intra']:.4f}"
    )

    rows: List[dict] = []
    artifacts: List[dict] = []
    caps: List[Optional[float]] = [None if cap < 0 else cap for cap in args.caps]
    for quantile in args.quantiles:
        for scale in args.scales:
            for floor in args.floors:
                for cap in caps:
                    if cap is not None and floor > cap:
                        continue
                    config = {
                        "quantile": float(quantile),
                        "scale": float(scale),
                        "floor": float(floor),
                        "cap": None if cap is None else float(cap),
                        "min_samples": int(args.min_samples),
                        "calib_mode": args.calib_mode,
                    }
                    thresholds, global_thr = make_thresholds(
                        groups=groups,
                        known_ids=known_class_ids,
                        quantile=quantile,
                        scale=scale,
                        floor=floor,
                        cap=cap,
                        min_samples=args.min_samples,
                    )
                    results = evaluate_adaptive(
                        matched=matched,
                        unmatched_predictions=unmatched_predictions,
                        unmatched_gts=unmatched_gts,
                        known_class_ids=known_class_ids,
                        thresholds=thresholds,
                        global_thr=global_thr,
                    )
                    row = metric_row(run_name, config, results, thresholds, global_thr, args.detailed_json)
                    rows.append(row)
                    artifacts.append(
                        {
                            "row": row,
                            "config": config,
                            "thresholds": thresholds,
                            "global_thr": global_thr,
                        }
                    )

    rows_sorted_acc = sorted(rows, key=lambda r: float(r["Acc"]), reverse=True)
    rows_sorted_macro = sorted(rows, key=lambda r: float(r["Macro-F1"]), reverse=True)
    rows_sorted_rr = sorted(rows, key=lambda r: float(r["RR"]), reverse=True)

    print("\nBest by Acc:")
    for row in rows_sorted_acc[:10]:
        print("  " + fmt(row))
    print("\nBest by Macro-F1:")
    for row in rows_sorted_macro[:10]:
        print("  " + fmt(row))
    print("\nBest by RR:")
    for row in rows_sorted_rr[:10]:
        print("  " + fmt(row))

    csv_path = args.output_dir / f"{run_name}_val_calibrated_sweep.csv"
    save_val_calibrated_csv(csv_path, rows)

    summary = {
        "run": run_name,
        "checkpoint": str(args.checkpoint),
        "detailed_json": str(args.detailed_json),
        "train_dir": str(args.train_dir),
        "val_dir": str(args.val_dir),
        "gt_dir": str(args.gt_dir),
        "calib_mode": args.calib_mode,
        "best_by_acc": rows_sorted_acc[:10],
        "best_by_macro_f1": rows_sorted_macro[:10],
        "best_by_rr": rows_sorted_rr[:10],
    }
    summary_path = args.output_dir / f"{run_name}_val_calibrated_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    if args.write_best_json:
        for metric, best_row in [("acc", rows_sorted_acc[0]), ("macro_f1", rows_sorted_macro[0])]:
            artifact = next(item for item in artifacts if item["row"] is best_row)
            out_path = args.output_dir / "rethresholded_json" / f"{run_name}_best_{metric}.json"
            write_adaptive_json(
                input_data=input_data,
                thresholds=artifact["thresholds"],
                global_thr=artifact["global_thr"],
                config=artifact["config"],
                output_path=out_path,
            )
            print(f"Written best-{metric} JSON: {out_path}")

    print("\nOutputs:")
    print(f"  CSV: {csv_path}")
    print(f"  Summary: {summary_path}")


if __name__ == "__main__":
    main()
