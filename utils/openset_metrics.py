"""
Open-Set / Unknown Detection Metrics
====================================
基于 IoU bbox 匹配的检测 + 分类 pipeline 专用指标。

开集检测策略：最近邻匹配 + 距离度量
  - distance > threshold → 拒识（Unknown）
  - distance <= threshold → 接受分类

八个评估指标：
  1. Macro-F1  — 已知类分类质量（TP/FP/FN based on IoU bbox matching）
  2. FRR        — 已知类误拒率（GT=known 且 matched, pred=-1 的比例）
  3. URR        — 未知类召回率（GT=999 且 matched, pred=-1 的比例）
  4. PS         — 符号匹配精度（匹配到图例符号的正确率）
  5. RS         — 符号匹配召回率（正确识别的符号/图例符号总数）
  6. PR         — 拒识精度（正确拒识/总拒识）
  7. RR         — 拒识召回率（正确拒识/应拒识总数）
  8. Acc        — 总体准确率（正确分类样本/总样本）
"""
from __future__ import annotations
import numpy as np


def compute_ten_metrics(
    matched_pairs: list,
    unmatched_preds: list,
    unmatched_gts: list,
    known_class_ids: list,
    UNKNOWN_GT_ID: int = 999,
    IOU_THRESHOLD: float = 0.5,
) -> dict:
    """Compute the eight evaluation metrics.

    八个指标：
      1. Macro-F1  — 已知类分类质量（TP/FP/FN based on IoU bbox matching）
      2. FRR        — 已知类误拒率（GT=known 且 matched, pred=-1 的比例）
      3. URR        — 未知类召回率（GT=999 且 matched, pred=-1 的比例）
      4. PS         — 符号匹配精度（匹配到图例符号的正确率）
      5. RS         — 符号匹配召回率（正确识别的符号/图例符号总数）
      6. PR         — 拒识精度（正确拒识/总拒识）
      7. RR         — 拒识召回率（正确拒识/应拒识总数）
      8. Acc        — 总体准确率（正确分类样本/总样本）

    Args:
        matched_pairs: List of (pred_cls, gt_cls, pred_bbox, gt_bbox) tuples
                       for bbox-matched (IoU >= threshold) predictions.
        unmatched_preds: List of (pred_cls, pred_bbox) tuples for predictions
                         that did not match any GT bbox.
        unmatched_gts: List of (gt_cls, gt_bbox) tuples for GT bboxes that
                      were not matched by any prediction.
        known_class_ids: List of known class IDs.
        UNKNOWN_GT_ID: Class ID used for unknown class in GT (default 999).
        IOU_THRESHOLD: IoU threshold used for bbox matching (default 0.5).
    Returns:
        dict with keys: Macro-F1, FRR, URR, PS, RS, PR, RR, Acc.
    """
    known_ids_set = set(known_class_ids)

    # ── Metric 1: Macro-F1 ──────────────────────────────────────────────────
    TP = {c: 0 for c in known_class_ids}
    FP = {c: 0 for c in known_class_ids}
    FN = {c: 0 for c in known_class_ids}

    for pred_cls, gt_cls, *_ in matched_pairs:
        if gt_cls in known_ids_set:
            if pred_cls == gt_cls:
                TP[gt_cls] += 1
            elif pred_cls in known_ids_set:
                FP[pred_cls] += 1
                FN[gt_cls] += 1
            elif pred_cls == -1:
                FN[gt_cls] += 1
        elif gt_cls == UNKNOWN_GT_ID:
            if pred_cls in known_ids_set:
                FP[pred_cls] += 1

    for pred_cls, *_ in unmatched_preds:
        if pred_cls in known_ids_set:
            FP[pred_cls] += 1

    for gt_cls, *_ in unmatched_gts:
        if gt_cls in known_ids_set:
            FN[gt_cls] += 1

    f1_per_class = {}
    for c in known_class_ids:
        tp, fp, fn = TP[c], FP[c], FN[c]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if precision + recall == 0:
            f1_per_class[c] = 0.0
        else:
            f1_per_class[c] = 2 * precision * recall / (precision + recall)

    classes_in_inference = {
        c for c in known_class_ids
        if TP[c] > 0 or FP[c] > 0 or FN[c] > 0
    }

    f1_list = [f1_per_class[c] for c in sorted(classes_in_inference)]
    macro_f1 = float(np.mean(f1_list)) if f1_list else 0.0

    # ── Metric 2: FRR ───────────────────────────────────────────────────────
    known_matched_total = 0
    known_rejected = 0
    for pred_cls, gt_cls, *_ in matched_pairs:
        if gt_cls in known_ids_set:
            known_matched_total += 1
            if pred_cls == -1:
                known_rejected += 1
    frr = float(known_rejected / known_matched_total) if known_matched_total > 0 else 0.0

    # ── Metric 3: URR ───────────────────────────────────────────────────────
    unknown_matched_total = 0
    unknown_correctly_rejected = 0
    for pred_cls, gt_cls, *_ in matched_pairs:
        if gt_cls == UNKNOWN_GT_ID:
            unknown_matched_total += 1
            if pred_cls == -1:
                unknown_correctly_rejected += 1
    urr = float(unknown_correctly_rejected / unknown_matched_total) if unknown_matched_total > 0 else 0.0

    # ── Metric 4: PS (Symbol Matching Precision) ─────────────────────────────
    # 符号匹配精度：匹配到图例符号的正确率
    # 正确匹配 = GT是已知类 AND 预测是已知类 AND 类别正确
    correct_matches = 0
    total_matched_known = 0
    for pred_cls, gt_cls, *_ in matched_pairs:
        if gt_cls in known_ids_set:
            total_matched_known += 1
            if pred_cls in known_ids_set and pred_cls == gt_cls:
                correct_matches += 1
    ps = float(correct_matches / total_matched_known) if total_matched_known > 0 else 0.0

    # ── Metric 5: RS (Symbol Matching Recall) ───────────────────────────────
    # 符号匹配召回率：正确识别的符号 / 图例符号总数
    # 总图例符号 = 所有已知类 GT bbox 数量
    total_known_gts = 0
    for _, gt_cls, *_ in matched_pairs:
        if gt_cls in known_ids_set:
            total_known_gts += 1
    total_known_gts += len([gt_cls for gt_cls, _ in unmatched_gts if gt_cls in known_ids_set])
    rs = float(correct_matches / total_known_gts) if total_known_gts > 0 else 0.0

    # ── Metric 6: PR (Precision of Rejection) ───────────────────────────────
    # 拒识精度：正确拒识 / 总拒识
    # 正确拒识 = GT是unknown AND 预测是-1
    # 总拒识 = 预测为-1的数量
    correct_rejections = 0
    total_rejections = 0
    for pred_cls, gt_cls, *_ in matched_pairs:
        if pred_cls == -1:
            total_rejections += 1
            if gt_cls == UNKNOWN_GT_ID:
                correct_rejections += 1
    # 未匹配的预测也是拒识
    total_rejections += len(unmatched_preds)
    pr = float(correct_rejections / total_rejections) if total_rejections > 0 else 0.0

    # ── Metric 7: RR (Recall of Rejection) ──────────────────────────────────
    # 拒识召回率：正确拒识 / 应拒识总数
    # 应拒识总数 = 所有 unknown GT 的数量
    total_unknown_gts = 0
    for _, gt_cls, *_ in matched_pairs:
        if gt_cls == UNKNOWN_GT_ID:
            total_unknown_gts += 1
    total_unknown_gts += len([gt_cls for gt_cls, _ in unmatched_gts if gt_cls == UNKNOWN_GT_ID])
    rr = float(correct_rejections / total_unknown_gts) if total_unknown_gts > 0 else 0.0

    # ── Metric 8: Acc (Overall Accuracy) ───────────────────────────────────
    # 总体准确率：正确分类样本 / 总样本
    # 总样本 = 所有 matched + 所有未匹配预测（拒识）
    total_samples = len(matched_pairs) + len(unmatched_preds)
    correct_classifications = correct_matches + correct_rejections
    acc = float(correct_classifications / total_samples) if total_samples > 0 else 0.0

    return {
        "Macro-F1": macro_f1,
        "FRR": frr,
        "URR": urr,
        "PS": ps,
        "RS": rs,
        "PR": pr,
        "RR": rr,
        "Acc": acc,
        "n_classes_in_inference": len(classes_in_inference),
        "n_classes_total": len(known_class_ids),
        "classes_in_inference": sorted(classes_in_inference),
        "f1_per_class": f1_per_class,
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "correct_matches": correct_matches,
        "correct_rejections": correct_rejections,
        "total_matched_known": total_matched_known,
        "total_known_gts": total_known_gts,
        "total_rejections": total_rejections,
        "total_unknown_gts": total_unknown_gts,
    }
