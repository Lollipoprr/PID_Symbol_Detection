from pipeline.base import BasePipeline
from typing import Dict, Any, List, Optional
from pathlib import Path
import logging
from utils.metrics import MetricsCalculator
from utils.helpers import get_text_files
from utils.bbox_utils import BBoxUtils
import json
import numpy as np
from datetime import datetime
from itertools import product


class EvaluationPipeline(BasePipeline):
    """Evaluation pipeline for the model.

    Computes five evaluation metrics:
      1. Stage-1  : class-agnostic  detection  (IoU + Precision/Recall/F1)
      2. Stage-2  : class-aware      classification (IoU + class-correct + P/R/F1)
      3. Five     : Macro-F1 / FRR / URR
    """

    def __init__(self, config_path: str = "configs/config.yaml"):
        super().__init__(config_path)
        self.logger = logging.getLogger(__name__)

    # ---------------------------------------------------------------------
    # Validation
    # ---------------------------------------------------------------------
    def validate(self) -> bool:
        """Validate pipeline stage inputs and configuration.

        部分验证：仅警告缺失文件，不阻塞评估流程。
        各阶段评估会按需跳过。
        """
        model_paths = self.get_model_paths()

        if not model_paths['stage2_weights_dir'].exists():
            self.logger.warning(
                "[Validate] Stage-2 model directory not found — "
                "Stage-2 metrics will be skipped"
            )

        paths = self.get_data_paths()
        stage1_txts = get_text_files(paths['class_agnostic_results_dir'])
        if len(stage1_txts) == 0:
            self.logger.warning(
                "[Validate] Stage-1 prediction txt files not found — "
                "Stage-1 metrics will be skipped"
            )

        return True

    # ---------------------------------------------------------------------
    # Hungarian BBox Matching (pure numpy, no scipy dependency)
    # ---------------------------------------------------------------------
    @staticmethod
    def _hungarian_match(cost: np.ndarray) -> tuple:
        """Kuhn-Munkres (Hungarian) algorithm for optimal assignment.

        Handles rectangular cost matrices (n_pred x n_gt) by padding to square.

        Args:
            cost: 2D numpy array [n_pred x n_gt], cost[i,j] = cost of assigning
                  pred i to GT j. Lower is better.

        Returns:
            (row_ind, col_ind): Arrays of matched indices.
            row_ind[i] = matched pred index, col_ind[i] = matched GT index.
        """
        n_pred, n_gt = cost.shape
        n = max(n_pred, n_gt)

        # Pad to square matrix with zeros (free / never chosen if preds > GTs)
        C = np.zeros((n, n), dtype=np.float64)
        C[:n_pred, :n_gt] = cost

        # Kuhn-Munkres for minimization
        u = np.zeros(n + 1, dtype=np.float64)
        v = np.zeros(n + 1, dtype=np.float64)
        p = np.zeros(n + 1, dtype=np.int32)
        way = np.zeros(n + 1, dtype=np.int32)

        for i in range(1, n + 1):
            p[0] = i
            j0 = 0
            minv = np.full(n + 1, np.inf, dtype=np.float64)
            used = np.zeros(n + 1, dtype=bool)
            while p[j0] != 0:
                used[j0] = True
                i0 = p[j0]
                delta = np.inf
                j1 = 0
                for j in range(1, n + 1):
                    if not used[j]:
                        cur = C[i0 - 1, j - 1] - u[i0] - v[j]
                        if cur < minv[j]:
                            minv[j] = cur
                            way[j] = j0
                        if minv[j] < delta:
                            delta = minv[j]
                            j1 = j
                for j in range(n + 1):
                    if used[j]:
                        u[p[j]] += delta
                        v[j] -= delta
                    else:
                        minv[j] -= delta
                j0 = j1

            # Augmenting
            while True:
                j1 = way[j0]
                p[j0] = p[j1]
                j0 = j1
                if j0 == 0:
                    break

        # Extract valid assignments (only where both pred and GT index < original size)
        row_ind = []
        col_ind = []
        for j in range(1, n + 1):
            if p[j] != 0 and p[j] - 1 < n_pred and j - 1 < n_gt:
                row_ind.append(p[j] - 1)
                col_ind.append(j - 1)
        return np.array(row_ind, dtype=np.int32), np.array(col_ind, dtype=np.int32)

    # ---------------------------------------------------------------------
    # Stage-1 metrics (class-agnostic)
    # ---------------------------------------------------------------------
    def compute_stage1_metrics(self) -> Dict[str, Any]:
        """Compute Stage-1 metrics (class-agnostic) and return summary dict."""
        paths = self.get_data_paths()
        metrics_calculator = MetricsCalculator()

        stage1_txts = get_text_files(paths['class_agnostic_results_dir'])
        if len(stage1_txts) == 0:
            self.logger.warning(
                "[Stage-1] No prediction txt files found — skipping Stage-1 metrics"
            )
            return {'sheet_wise': [], 'overall': {}, 'skipped': True}

        self.logger.info("Computing Stage-1 metrics (class-agnostic)")
        result = metrics_calculator.calculate_overall_metrics_for_dataset_dir(
            paths['class_agnostic_results_dir'],
            paths['stage1_inference_labels_agnostic_dir'],
        )

        self.logger.info("Stage-1 metrics computed successfully")
        return {
            'sheet_wise': result['sheet_wise'],
            'overall': result['overall'],
        }

    # ---------------------------------------------------------------------
    # Stage-2 metrics (class-aware)
    # ---------------------------------------------------------------------
    def compute_stage2_metrics(self) -> Dict[str, Any]:
        """Compute Stage-2 metrics (class-aware) and return summary dict."""
        paths = self.get_data_paths()
        metrics_calculator = MetricsCalculator()

        stage2_preds = get_text_files(paths['stage2_inference_results_dir'])
        if len(stage2_preds) == 0:
            self.logger.warning(
                "[Stage-2] No prediction txt files found — skipping Stage-2 metrics"
            )
            return {'sheet_wise': [], 'overall': {}, 'skipped': True}

        known_class_ids = []
        if paths['train_dir'].exists():
            known_class_ids = sorted(
                int(p.name)
                for p in paths['train_dir'].iterdir()
                if p.is_dir() and p.name.isdigit()
            )

        self.logger.info("Computing Stage-2 metrics (class-aware)")
        result = metrics_calculator.calculate_overall_metrics_for_dataset_dir(
            paths['stage2_inference_results_dir'],
            paths['stage1_inference_labels_aware_dir'],
            class_aware=True,
            valid_class_ids=known_class_ids,
            ignore_predictions_on_excluded_gt=True,
        )

        self.logger.info("Stage-2 metrics computed successfully")
        return {
            'sheet_wise': result['sheet_wise'],
            'overall': result['overall'],
            'known_class_ids': known_class_ids,
        }

    # ---------------------------------------------------------------------
    # Open-Set / Unknown Detection metrics
    # ---------------------------------------------------------------------
    def compute_openset_metrics(
        self,
        detailed_json: Optional[Path] = None,
        gt_dir: Optional[Path] = None,
        source: Optional[str] = None,
    ) -> Dict[str, Any]:
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

        开集检测策略：最近邻匹配 + 距离度量（阈值 0.25）

        数据来源：
          - 预测数据：detailed_json
          - GT 标签：gt_dir（inference/stage2/labels，YOLO格式 [cls x y w h]，999=未知类）
          - 已知类 ID：从 train_dir 子目录名自动检测

        Args:
            detailed_json: Path to predictions JSON. If None, auto-searches
                          based on source parameter.
            gt_dir:        Path to GT label dir. If None, uses
                          stage2_inference_labels_aware_dir from config.
            source:        "baseline". Auto-detected if None.

        Returns:
            Dict with the nine metrics and summary statistics.
        """
        from utils.openset_metrics import compute_ten_metrics

        paths = self.get_data_paths()

        # ── 1. 找到 detailed_json ─────────────────────────────────────────
        if detailed_json is None:
            detailed_dir = paths['stage2_inference_results_dir'].parent / "detailed_results"

            json_files = sorted(detailed_dir.glob("predictions_*.json"))
            if not json_files:
                self.logger.warning(
                    "[OpenSet] 没有找到 detailed predictions JSON，"
                    "请先运行 stage2_inference 推理生成详细结果"
                )
                return {"error": "no detailed predictions found"}
            detailed_json = json_files[-1]
            self.logger.info(f"[OpenSet] 数据源: {detailed_dir.name}")

        self.logger.info(f"[OpenSet] 读取推理结果: {detailed_json}")

        with open(detailed_json, encoding="utf-8") as f:
            pred_data = json.load(f)

        predictions: List[Dict] = pred_data.get("predictions", [])
        if not predictions:
            return {"error": "predictions list is empty"}

        # ── 2. 从推理结果获取拒识策略信息 ─────────────────────────────
        inference_summary = pred_data.get("summary", {})
        rejection_strategy = inference_summary.get("rejection_strategy", "nearest_neighbor_distance")
        self.logger.info(f"[OpenSet] 拒识策略: {rejection_strategy}")

        # ── 3. 加载 GT 标签 ──────────────────────────────────────────────
        # 包含 999=unknown 的 GT（stage2_labels_aware_dir）
        if gt_dir is None:
            gt_dir = Path(paths.get('stage2_inference_labels_aware_dir',
                                    paths['raw_images_dir'].parent / "labels"))

        self.logger.info(f"[OpenSet] 加载 GT 标签 from: {gt_dir}")

        if not gt_dir.exists():
            self.logger.error(f"[OpenSet] GT 目录不存在: {gt_dir}")
            return {"error": f"GT directory not found: {gt_dir}"}

        gt_map: Dict[str, List[tuple]] = {}
        for txt_file in gt_dir.glob("*.txt"):
            try:
                boxes = BBoxUtils.get_bboxes_array_from_file(txt_file)
                # YOLO format: [cls, xc, yc, w, h] → (cls_id, [xc, yc, w, h])
                gt_map[txt_file.stem] = [
                    (int(b[0]), list(b[1:])) for b in boxes
                ]
            except Exception:
                gt_map[txt_file.stem] = []

        # ── 3. 已知类 ID ─────────────────────────────────────────────────
        known_class_ids: List[int] = []
        train_dir = paths['train_dir']
        if train_dir.exists():
            known_class_ids = sorted(
                int(p.name) for p in train_dir.iterdir()
                if p.is_dir() and p.name.isdigit()
            )
        if not known_class_ids:
            self.logger.warning("[OpenSet] 无法从 train_dir 检测已知类，使用空列表")
            known_class_ids = []
        self.logger.info(
            f"[OpenSet] 已知类: {len(known_class_ids)} 个, IDs={known_class_ids}"
        )

        UNKNOWN_GT_ID = 999
        IOU_THRESHOLD = 0.5

        # ── 4. IoU 计算辅助 ─────────────────────────────────────────────
        def _compute_iou_yolo(bbox1: List[float], bbox2: List[float]) -> float:
            x1c, y1c, w1, h1 = bbox1
            x2c, y2c, w2, h2 = bbox2
            x1_min, y1_min = x1c - w1/2, y1c - h1/2
            x1_max, y1_max = x1c + w1/2, y1c + h1/2
            x2_min, y2_min = x2c - w2/2, y2c - h2/2
            x2_max, y2_max = x2c + w2/2, y2c + h2/2
            inter_x_min = max(x1_min, x2_min)
            inter_y_min = max(y1_min, y2_min)
            inter_x_max = min(x1_max, x2_max)
            inter_y_max = min(y1_max, y2_max)
            inter_w = max(0.0, inter_x_max - inter_x_min)
            inter_h = max(0.0, inter_y_max - inter_y_min)
            inter_area = inter_w * inter_h
            area1 = w1 * h1
            area2 = w2 * h2
            union_area = area1 + area2 - inter_area
            return inter_area / union_area if union_area > 0 else 0.0

        # ── 5. Hungarian Bbox Matching（逐图最优分配） ───────────────────────
        matched_by_img: Dict[str, list] = {}
        unmatched_preds_by_img: Dict[str, list] = {}
        unmatched_gts_by_img: Dict[str, list] = {}
        distances_by_img: Dict[str, list] = {}
        gaps_by_img: Dict[str, list] = {}

        for stem in gt_map.keys():
            matched_by_img[stem] = []
            unmatched_preds_by_img[stem] = []
            unmatched_gts_by_img[stem] = []   # 先假设全部GT为unmatched
            distances_by_img[stem] = []
            gaps_by_img[stem] = []

        for stem, gt_list in gt_map.items():
            # 收集该图的预测样本
            stem_preds = [s for s in predictions
                          if Path(s.get("image_name", "")).stem == stem]
            if not stem_preds:
                unmatched_gts_by_img[stem] = [(tc, gb) for tc, gb in gt_list]
                continue
            if not gt_list:
                unmatched_preds_by_img[stem] = []
                continue

            n_pred = len(stem_preds)
            n_gt = len(gt_list)

            # 构建 IoU 代价矩阵 [n_pred x n_gt]
            iou_matrix = np.zeros((n_pred, n_gt), dtype=np.float64)
            pred_bboxes = []
            pred_classes = []
            pred_distances = []
            pred_gaps = []

            for i, sample in enumerate(stem_preds):
                bbox_coords = sample.get("bbox", {})
                if isinstance(bbox_coords, dict):
                    pb = [
                        bbox_coords.get("xc", 0),
                        bbox_coords.get("yc", 0),
                        bbox_coords.get("w", 0),
                        bbox_coords.get("h", 0),
                    ]
                else:
                    pb = [0, 0, 0, 0]
                pred_bboxes.append(pb)

                pc = int(sample.get("predicted_class", -1))
                if pc == -1 and sample.get("is_unknown", True):
                    pc = -1
                pred_classes.append(pc)

                d = sample.get("distance") or sample.get("min_dist") or 0.0
                if isinstance(d, list):
                    d = d[0] if d else 0.0
                pred_distances.append(float(d))

                g = sample.get("gap")
                if g is None:
                    g = 0.0
                if isinstance(g, list):
                    g = g[0] if g else 0.0
                pred_gaps.append(float(g))

                for j, (_, gb) in enumerate(gt_list):
                    iou_matrix[i, j] = _compute_iou_yolo(pb, gb)

            # 匈牙利算法：最大化总 IoU == 最小化 -IoU
            cost_matrix = -iou_matrix
            row_ind, col_ind = EvaluationPipeline._hungarian_match(cost_matrix)

            used_gt_idxs = set()
            for r, c in zip(row_ind, col_ind):
                best_iou = iou_matrix[r, c]
                if best_iou >= IOU_THRESHOLD:
                    matched_by_img[stem].append((
                        pred_classes[r],
                        gt_list[c][0],
                        pred_bboxes[r],
                        gt_list[c][1],
                    ))
                    distances_by_img[stem].append(pred_distances[r])
                    gaps_by_img[stem].append(pred_gaps[r])
                    used_gt_idxs.add(c)
                else:
                    unmatched_preds_by_img[stem].append((
                        pred_classes[r],
                        pred_bboxes[r],
                    ))

            # 未匹配的 GT
            for idx, (tc, gb) in enumerate(gt_list):
                if idx not in used_gt_idxs:
                    unmatched_gts_by_img[stem].append((tc, gb))

        # 日志
        n_matched = sum(len(v) for v in matched_by_img.values())
        n_unmatched_p = sum(len(v) for v in unmatched_preds_by_img.values())
        n_unmatched_g = sum(len(v) for v in unmatched_gts_by_img.values())
        self.logger.info(
            f"[OpenSet] matched={n_matched}, unmatched_preds={n_unmatched_p}, "
            f"unmatched_gts={n_unmatched_g}"
        )

        if n_matched == 0:
            return {"error": "no matched samples with ground truth"}

        # ── 6. 汇总所有图片的数据 ────────────────────────────────────────
        all_matched = []
        all_unmatched_preds = []
        all_unmatched_gts = []
        all_distances = []
        all_gaps = []

        for stem in sorted(matched_by_img.keys()):
            for (pred_cls, gt_cls, pred_bbox, gt_bbox) in matched_by_img[stem]:
                all_matched.append((pred_cls, gt_cls, pred_bbox, gt_bbox))
            all_unmatched_preds.extend(unmatched_preds_by_img.get(stem, []))
            all_unmatched_gts.extend(unmatched_gts_by_img.get(stem, []))
            all_distances.extend(distances_by_img.get(stem, []))
            all_gaps.extend(gaps_by_img.get(stem, []))

        # ── 7. 计算指标 ──────────────────────────────────────────────
        self.logger.info("[OpenSet] 计算指标...")
        try:
            results = compute_ten_metrics(
                matched_pairs=all_matched,
                unmatched_preds=all_unmatched_preds,
                unmatched_gts=all_unmatched_gts,
                known_class_ids=known_class_ids,
                UNKNOWN_GT_ID=UNKNOWN_GT_ID,
                IOU_THRESHOLD=IOU_THRESHOLD,
            )
        except Exception as e:
            self.logger.error(f"[OpenSet] 指标计算失败: {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e)}

        for key in ["Macro-F1", "FRR", "URR", "PS", "RS", "PR", "RR", "Acc"]:
            val = results.get(key)
            if val is not None:
                self.logger.info(f"[OpenSet] {key} = {val:.4f}")
            else:
                self.logger.warning(f"[OpenSet] {key} = None")

        n_classes_inf = results.get("n_classes_in_inference", "?")
        n_classes_tot = results.get("n_classes_total", "?")
        self.logger.info(
            f"[OpenSet] 推理集覆盖: {n_classes_inf}/{n_classes_tot} 个已知类 "
            f"(Macro-F1 仅基于 {n_classes_inf} 个出现的类)"
        )

        n_known = sum(
            1 for (pc, gc, _, _) in all_matched
            if gc in set(known_class_ids)
        )
        n_unknown = sum(
            1 for (_, gc, _, _) in all_matched
            if gc == UNKNOWN_GT_ID
        )

        return {
            "Macro-F1": results["Macro-F1"],
            "FRR": results["FRR"],
            "URR": results["URR"],
            "PS": results.get("PS", 0.0),
            "RS": results.get("RS", 0.0),
            "PR": results.get("PR", 0.0),
            "RR": results.get("RR", 0.0),
            "Acc": results.get("Acc", 0.0),
            "summary": {
                "total_matched": n_matched,
                "n_known": n_known,
                "n_unknown": n_unknown,
                "n_unmatched_preds": n_unmatched_p,
                "n_unmatched_gts": n_unmatched_g,
                "known_class_ids": known_class_ids,
                "detailed_json": str(detailed_json),
                "gt_dir": str(gt_dir),
                "iou_threshold": IOU_THRESHOLD,
                "unknown_gt_id": UNKNOWN_GT_ID,
                "source": source or "auto-detected",
            }
        }

    # ---------------------------------------------------------------------
    # Embedding quality metrics
    def compute_embedding_quality(
        self,
        detailed_json: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """Compute Embedding Space Quality metrics from prototype geometry.

        使用 JSON 中保存的 prototype 矩阵和 query embedding，计算：
          - intra_compactness（类内紧凑度）：已知类样本到其类原型的平均/最大距离
          - inter_separation（类间分离度）：各类原型之间的最小成对距离
          - svg_photo_consistency（SVG-Photo 一致性）：推理图与对应 SVG 的相似度

        Args:
            detailed_json: predictions JSON path. Auto-searched if None.

        Returns:
            Dict with aggregated and per-class quality metrics.
        """
        paths = self.get_data_paths()
        UNKNOWN_GT_ID = 999  # 未知类 GT ID

        # ── 1. 加载详细结果 ──────────────────────────────────────────────
        if detailed_json is None:
            detailed_dir = paths['stage2_inference_results_dir'].parent / "detailed_results"
            json_files = sorted(detailed_dir.glob("predictions_*.json"))
            if not json_files:
                return {"error": "no detailed predictions found"}
            detailed_json = json_files[-1]

        self.logger.info(f"[EmbQuality] 读取: {detailed_json}")
        with open(detailed_json, encoding="utf-8") as f:
            pred_data = json.load(f)

        summary = pred_data.get("summary", {})
        predictions: List[Dict] = pred_data.get("predictions", [])
        if not predictions:
            return {"error": "predictions list is empty"}

        # ── 2. 加载 prototype 矩阵 ───────────────────────────────────────
        class_ids: List[int] = summary.get("class_ids", [])
        prototypes: List[List[float]] = summary.get("prototypes", [])

        if not class_ids or not prototypes:
            return {"error": "prototype matrix not found in JSON summary"}

        D = len(prototypes[0]) if prototypes else 0
        proto_matrix = np.array(prototypes, dtype=np.float32)  # [C, D]
        self.logger.info(
            f"[EmbQuality] Prototype matrix: {len(class_ids)} classes, "
            f"dim={D}, {len(predictions)} query samples"
        )

        # ── 3. 已知类 ID 集合 ────────────────────────────────────────────
        known_set = set(class_ids)

        # ── 4. 聚合同类样本的距离信息 ─────────────────────────────────────
        # 每个类的样本到其"目标原型"的距离分布
        # 目标原型 = true_class 的原型（如果知道）；否则用 predicted_class 的原型
        class_distances: Dict[int, List[float]] = {cid: [] for cid in class_ids}

        for sample in predictions:
            true_cls = int(sample.get("true_class", -1))
            pred_cls = int(sample.get("predicted_class", -1))
            target_cls = true_cls if true_cls in known_set else pred_cls

            if target_cls not in known_set:
                continue

            # 从 proto_distances 中取目标类的距离
            proto_distances: Dict[int, float] = sample.get("proto_distances", {})
            dist = proto_distances.get(target_cls)
            if dist is not None:
                class_distances[target_cls].append(float(dist))
            else:
                # fallback: 用 min_dist
                m = sample.get("min_dist")
                if m is not None:
                    class_distances[target_cls].append(float(m))

        # ── 5. 计算指标 ───────────────────────────────────────────────────
        results: Dict[str, Any] = {}
        n_empty = 0

        # --- 5a. 类内紧凑度（intra_compactness） ---
        per_cls_mean: Dict[int, float] = {}
        per_cls_max: Dict[int, float] = {}
        per_cls_std: Dict[int, float] = {}

        for cid, dists in class_distances.items():
            if len(dists) < 1:
                continue
            per_cls_mean[cid] = float(np.mean(dists))
            per_cls_max[cid] = float(np.max(dists))
            if len(dists) >= 2:
                per_cls_std[cid] = float(np.std(dists))
            else:
                per_cls_std[cid] = 0.0

        all_means = list(per_cls_mean.values())
        overall_intra = float(np.mean(all_means)) if all_means else 0.0
        results["intra_compactness"] = {
            "description": "平均距离（到类原型），越低越好",
            "overall_mean": overall_intra,
            "per_class_mean": per_cls_mean,
            "per_class_max": per_cls_max,
            "per_class_std": per_cls_std,
        }
        self.logger.info(
            f"[EmbQuality] intra_compactness (mean dist to proto) = {overall_intra:.4f}"
        )

        # --- 5b. 类间分离度（inter_separation）---
        if proto_matrix.shape[0] >= 2:
            # 成对欧氏距离矩阵
            pairwise_dists = np.linalg.norm(
                proto_matrix[:, np.newaxis, :] - proto_matrix[np.newaxis, :, :],
                axis=2
            )  # [C, C]
            np.fill_diagonal(pairwise_dists, np.inf)

            # 最小类间距离（各类到其最近邻的距离）
            nearest_neighbor_dists = pairwise_dists.min(axis=1)  # [C]
            inter_sep = float(np.mean(nearest_neighbor_dists))

            # 全部类间距离均值
            upper_tri = pairwise_dists[np.triu_indices_from(pairwise_dists, k=1)]
            inter_sep_all = float(np.mean(upper_tri))

            results["inter_separation"] = {
                "description": "类原型间距离，越高越好",
                "overall_mean_nn": inter_sep,        # 平均最近邻距离
                "overall_mean_all": inter_sep_all,    # 全部类对均值
                "per_class_nn": {
                    int(class_ids[i]): float(nearest_neighbor_dists[i])
                    for i in range(len(class_ids))
                },
            }
            self.logger.info(
                f"[EmbQuality] inter_separation: mean_nn={inter_sep:.4f}, "
                f"mean_all={inter_sep_all:.4f}"
            )
        else:
            results["inter_separation"] = {
                "description": "类原型间距离（不足2类，无法计算）",
                "overall_mean_nn": 0.0,
                "overall_mean_all": 0.0,
            }

        # --- 5c. SVG-Photo 一致性 ---
        svg_sims = []
        for sample in predictions:
            svg_sim = sample.get("svg_sim") or sample.get("svg_consistency")
            if svg_sim is not None:
                svg_sims.append(float(svg_sim))
        if svg_sims:
            results["svg_photo_consistency"] = {
                "description": "推理图与SVG的相似度，越高越好",
                "overall": float(np.mean(svg_sims)),
                "std": float(np.std(svg_sims)),
                "n": len(svg_sims),
            }
            self.logger.info(
                f"[EmbQuality] svg_photo_consistency = {np.mean(svg_sims):.4f} "
                f"(n={len(svg_sims)})"
            )

        return results

    # ---------------------------------------------------------------------
    # Saving helpers
    # ---------------------------------------------------------------------
    @staticmethod
    def _get_eval_dir() -> Path:
        """Return evaluation dir path, create if missing."""
        eval_dir = Path('evaluation')
        eval_dir.mkdir(parents=True, exist_ok=True)
        return eval_dir

    @staticmethod
    def _make_timestamp() -> str:
        return datetime.now().strftime('%Y%m%d_%H%M%S')

    def save_metrics(self, metrics: Dict[str, Any], tag: str,
                     timestamp: str | None = None) -> Path:
        """Save metrics dict to json file, return file path."""
        ts = timestamp or self._make_timestamp()
        file_path = self._get_eval_dir() / f"{ts}_{tag}.json"
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        self.logger.info(f"Metrics saved to {file_path}")
        return file_path

    # ---------------------------------------------------------------------
    # Pipeline entry
    # ---------------------------------------------------------------------
    def run(self) -> None:
        """Run full evaluation: Stage-1 → Stage-2 → Five metrics.

        各阶段相互独立，缺失数据时该阶段自动跳过并继续。
        """
        self.logger.info("=" * 60)
        self.logger.info("Starting evaluation pipeline (all stages)")
        self.logger.info("=" * 60)

        if not self.validate():
            self.logger.error("Validation failed")
            return

        stage1_metrics = self.compute_stage1_metrics()
        if stage1_metrics.get('skipped'):
            self.logger.info("[Stage-1] Skipped — no prediction files found")
        else:
            self.save_metrics(stage1_metrics, 'stage1')

        stage2_metrics = self.compute_stage2_metrics()
        if stage2_metrics.get('skipped'):
            self.logger.info("[Stage-2] Skipped — no prediction files found")
        else:
            self.save_metrics(stage2_metrics, 'stage2')

        eval_config = self.get_evaluation_metrics_config()
        openset_cfg = eval_config.get("openset", {})
        openset_metrics = {"error": "disabled"}
        if openset_cfg.get("enabled", True):
            self.logger.info("[Open-Set] 五个指标计算")
            openset_metrics = self.compute_openset_metrics()
            if "error" not in openset_metrics:
                self.save_metrics(openset_metrics, 'openset')

        # Embedding quality metrics
        embedding_metrics = {"error": "skipped"}
        if eval_config.get("embedding_quality", {}).get("enabled", True):
            self.logger.info("[Embedding] Embedding space quality metrics")
            embedding_metrics = self.compute_embedding_quality()
            if "error" not in embedding_metrics:
                self.save_metrics(embedding_metrics, 'embedding_quality')

        # Combined summary — 跳过缺失阶段
        combined: Dict[str, Any] = {}
        if not stage1_metrics.get('skipped'):
            combined['stage1'] = stage1_metrics
        if not stage2_metrics.get('skipped'):
            combined['stage2'] = stage2_metrics

        combined['overall'] = {}
        if not stage1_metrics.get('skipped'):
            combined['overall']['stage1'] = stage1_metrics.get('overall', {})
        if not stage2_metrics.get('skipped'):
            combined['overall']['stage2'] = stage2_metrics.get('overall', {})

        if openset_cfg.get("enabled", True) and "error" not in openset_metrics:
            combined['openset'] = openset_metrics
            combined['overall']['openset'] = {
                "Macro-F1": openset_metrics.get("Macro-F1"),
                "FRR": openset_metrics.get("FRR"),
                "URR": openset_metrics.get("URR"),
                "PS": openset_metrics.get("PS"),
                "RS": openset_metrics.get("RS"),
                "PR": openset_metrics.get("PR"),
                "RR": openset_metrics.get("RR"),
                "Acc": openset_metrics.get("Acc"),
            }

        embedding_cfg = eval_config.get("embedding_quality", {})
        if embedding_cfg.get("enabled", True) and "error" not in embedding_metrics:
            combined['embedding_quality'] = embedding_metrics
            combined['overall']['embedding_quality'] = {
                "intra_compactness": embedding_metrics.get("intra_compactness", {}).get("overall_mean"),
                "inter_separation": embedding_metrics.get("inter_separation", {}).get("overall_mean_nn"),
            }

        if combined.get('stage1') or combined.get('stage2') or 'openset' in combined or 'embedding_quality' in combined:
            self.save_metrics(combined, 'all')

        self.logger.info("Evaluation pipeline completed successfully")
        self.logger.info("=" * 60)
