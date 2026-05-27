"""
Structure Enhancement Network 推理脚本
=========================================

基于 Stage-1 YOLO class-agnostic 检测结果，对每个 bbox 做类别预测。

模型：SVGStructureEnhanceNet（三路共享网络）
- Photo和SVG共用同一个backbone，无模态差异
- 仅用Triplet Loss训练

输出格式与 baseline (stage2_inference.py) 完全一致：
  - results/stage_2/<run_name>/inference_results/*.txt     (YOLO 预测标注目录)
  - results/stage_2/<run_name>/detailed_results/predictions_*.json  (详细结果)

用法:
    python src/scripts/infer_svg_structure.py \
        --checkpoint outputs/svg_structure_v2/checkpoint_epoch_15.pth \
        --bbox_dir results/stage_1/class_agnostic \
        --image_dir data/inference/stage2/images \
        --train_dir data/processed/stage_2/train \
        --run_name svg_structure_v2_epoch15 \
        [--embed_dim 64] \
        [--rejection_threshold 0.25]
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import v2

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.svg_structure_enhance_net import SVGStructureEnhanceNet

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding="utf-8",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Transform（与训练阶段完全一致）
# ─────────────────────────────────────────────────────────────────────────────
def make_transform() -> v2.Compose:
    return v2.Compose([
        v2.ToTensor(),
        v2.Lambda(lambda t: _pad_to_square(t)),
        v2.Resize((224, 224), antialias=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def _pad_to_square(img: torch.Tensor) -> torch.Tensor:
    """[C, H, W] float tensor pad 为正方形，白色填充，居中"""
    _, h, w = img.shape
    if h == w:
        return img
    s = max(h, w)
    padded = torch.ones(img.shape[0], s, s, dtype=img.dtype, device=img.device)
    top  = (s - h) // 2
    left = (s - w) // 2
    padded[:, top:top+h, left:left+w] = img
    return padded


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────
def load_model(
    checkpoint_path: Path,
    embed_dim: int,
    device: torch.device,
) -> SVGStructureEnhanceNet:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        state_dict = ckpt["model_state"]
        logger.info(f"Checkpoint from epoch {ckpt.get('epoch', '?')}")
    elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
        logger.info(f"Checkpoint from epoch {ckpt.get('epoch', '?')}")
    else:
        state_dict = ckpt

    if any(k.startswith("embedding_net.") for k in state_dict):
        state_dict = {
            k.removeprefix("embedding_net."): v
            for k, v in state_dict.items()
            if k.startswith("embedding_net.")
        }
        logger.info("Loaded TripletNet checkpoint; stripped 'embedding_net.' prefix")

    model = SVGStructureEnhanceNet(
        embed_dim=embed_dim,
        use_pretrained=False,
    ).to(device)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        logger.warning(f"Missing {len(missing)} keys: {missing[:3]}")
    if unexpected:
        logger.warning(f"Unexpected {len(unexpected)} keys: {unexpected[:3]}")

    model.eval()
    logger.info(f"Model loaded: {checkpoint_path}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Prototype computation
# ─────────────────────────────────────────────────────────────────────────────
def get_image_files(directory: Path):
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
    files = []
    for p in directory.rglob("*"):
        if p.suffix.lower() in exts:
            files.append(p)
    return sorted(files)


def compute_prototypes(
    model: SVGStructureEnhanceNet,
    train_dir: Path,
    transform: v2.Compose,
    device: torch.device,
    batch_size: int = 64,
) -> tuple[dict[int, torch.Tensor], dict]:
    """
    从 train_dir 提取各类 embedding 并计算加权原型。

    策略（与 baseline 一致）：
        Step1: 1/sqrt(N) 先验
        Step2: 自适应温度 softmax 加权
        Step3: 加权平均 → L2 normalize

    Returns:
        prototypes: {class_id: [1, D] torch.Tensor}
        stats: 诊断统计
    """
    model.eval()
    class_dirs = sorted(
        [d for d in train_dir.iterdir() if d.is_dir()],
        key=lambda d: int(d.name) if d.name.isdigit() else 999,
    )

    prototypes: dict[int, torch.Tensor] = {}
    temperature_map: dict[int, float] = {}
    all_intra_dists: list[float] = []

    for class_dir in class_dirs:
        cls_id = int(class_dir.name) if class_dir.name.isdigit() else None
        if cls_id is None:
            continue

        image_paths = get_image_files(class_dir)
        if not image_paths:
            continue

        # 批量提取 embedding
        all_embs: list[torch.Tensor] = []
        for i in range(0, len(image_paths), batch_size):
            chunk = image_paths[i : i + batch_size]
            tensors = []
            for p in chunk:
                img = Image.open(p).convert("RGB")
                tensors.append(transform(img))
            batch = torch.stack(tensors).to(device)
            with torch.no_grad():
                out = model(batch)
            # out 可能是 dict{'photo_emb': ...} 或直接是 tensor
            emb = out['photo_emb'] if isinstance(out, dict) else out
            emb = F.normalize(emb, p=2, dim=1)
            all_embs.append(emb.cpu())

        embs = torch.cat(all_embs, dim=0)  # [N, D]
        embs_n = embs / (embs.norm(dim=1, keepdim=True) + 1e-8)
        n = embs.shape[0]

        # Step1: 1/sqrt(N) 先验
        prior = torch.ones(n) / n
        prior = prior / prior.sum() / np.sqrt(n)
        prior = prior / prior.sum()

        # Step2: 自适应温度
        if n < 200:
            T = 0.3
        elif n < 1000:
            T = 0.15
        else:
            T = 0.1
        temperature_map[cls_id] = T

        class_mean = embs_n.mean(dim=0, keepdim=True)
        class_mean = class_mean / (class_mean.norm(dim=1) + 1e-8)
        sims = (embs_n @ class_mean.T).squeeze(1)
        sim_weights = torch.softmax(sims / T, dim=0)

        combined = prior * sim_weights
        combined = combined / combined.sum()
        proto = (embs_n * combined.unsqueeze(1)).sum(dim=0, keepdim=True)
        proto = proto / (proto.norm(dim=1) + 1e-8)
        prototypes[cls_id] = proto

        # 类内距离
        intra = (embs_n - proto).norm(dim=1)
        all_intra_dists.extend(intra.tolist())

    all_arr = np.array(all_intra_dists) if all_intra_dists else np.array([0.0])
    stats = {
        "num_classes": len(prototypes),
        "p50_intra": float(np.percentile(all_arr, 50)),
        "p90_intra": float(np.percentile(all_arr, 90)),
        "p95_intra": float(np.percentile(all_arr, 95)),
        "p99_intra": float(np.percentile(all_arr, 99)),
    }
    logger.info(
        f"[Prototype] {len(prototypes)} classes built | "
        f"intra P50={stats['p50_intra']:.4f} P90={stats['p90_intra']:.4f}"
    )
    return prototypes, stats


# ─────────────────────────────────────────────────────────────────────────────
# Inference: crop + embed + classify
# ─────────────────────────────────────────────────────────────────────────────
def run_inference(
    model: SVGStructureEnhanceNet,
    prototypes: dict[int, torch.Tensor],
    bbox_dir: Path,
    image_dir: Path,
    output_dir: Path,
    transform: v2.Compose,
    device: torch.device,
    rejection_threshold: float = 0.25,
    rejection_strategy: str = "nearest_neighbor_distance",
) -> tuple[list, dict]:
    """
    遍历 stage-1 检测结果，对每个 bbox 做分类。

    输出（与 baseline 一致）：
        1. output_dir/*.txt        — YOLO 标注格式
        2. detailed_results/*.json — 逐 bbox 详细记录

    Returns:
        detailed_results (list of dicts), summary (dict)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 预计算 prototype 矩阵（加速最近邻）
    sorted_ids = sorted(prototypes.keys())
    proto_matrix = torch.cat([prototypes[c] for c in sorted_ids], dim=0).to(device)

    detailed_results: list[dict] = []
    total_bboxes = 0
    rejected_bboxes = 0
    diag_distances: list[float] = []

    # 遍历每个 bbox 标注文件
    bbox_files = sorted(bbox_dir.glob("*.txt"))
    if not bbox_files:
        logger.error(f"Bbox dir has no .txt files: {bbox_dir}")
        return [], {}

    for bbox_file in bbox_files:
        img_stem = bbox_file.stem

        # 找原图
        img_path = image_dir / f"{img_stem}.png"
        if not img_path.exists():
            img_path = image_dir / f"{img_stem}.jpg"
        if not img_path.exists():
            img_path = image_dir / f"{img_stem}.jpeg"

        if not img_path.exists():
            logger.warning(f"Image not found for {img_stem}, skipping")
            (output_dir / f"{img_stem}.txt").write_text("")
            continue

        image = cv2.imread(str(img_path))
        if image is None:
            continue
        img_h, img_w = image.shape[:2]

        try:
            lines = bbox_file.read_text().strip().split("\n")
        except Exception:
            continue

        out_labels: list[str] = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 4:
                continue

            total_bboxes += 1

            # 解析 bbox，兼容 YOLO 的 "cls xc yc w h" 和纯 "xc yc w h"
            if len(parts) >= 5:
                xc, yc, w, h = [float(x) for x in parts[1:5]]
            else:
                xc, yc, w, h = [float(x) for x in parts[:4]]

            # 裁剪 bbox
            x1 = int((xc - w / 2) * img_w)
            y1 = int((yc - h / 2) * img_h)
            x2 = int((xc + w / 2) * img_w)
            y2 = int((yc + h / 2) * img_h)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img_w, x2), min(img_h, y2)
            if x2 <= x1 or y2 <= y1:
                continue

            cropped = image[y1:y2, x1:x2]
            if cropped.size == 0:
                continue

            pil_img = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
            tensor = transform(pil_img).unsqueeze(0).to(device)

            with torch.no_grad():
                out = model(tensor)
            emb = out['photo_emb'] if isinstance(out, dict) else out
            emb = F.normalize(emb, p=2, dim=1)  # [1, D]

            dists = (emb - proto_matrix).norm(dim=1)
            sorted_idx = dists.argsort()
            min_dist = float(dists[sorted_idx[0]])
            second_min_dist = float(dists[sorted_idx[1]])
            gap = second_min_dist - min_dist
            pred_cls = int(sorted_ids[sorted_idx[0]])

            diag_distances.append(min_dist)
            reject = min_dist > rejection_threshold

            if reject:
                rejected_bboxes += 1
                out_labels.append(f"-1 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
                pred_cls_out = -1
            else:
                out_labels.append(f"{pred_cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
                pred_cls_out = pred_cls

            # Top-3 nearest prototypes. Keep the raw nearest class even when
            # the final open-set decision rejects the bbox.
            top3 = [
                (int(sorted_ids[sorted_idx[i]]), float(dists[sorted_idx[i]]))
                for i in range(min(3, len(sorted_ids)))
            ]

            detailed_results.append({
                "image_path": str(img_path),
                "image_name": img_path.name,
                "predicted_class": pred_cls_out,
                "nearest_class": pred_cls,
                "true_class": -1,
                "distance": min_dist,
                "gap": gap,
                "is_unknown": reject,
                "top3": top3,
                "bbox": {"xc": float(xc), "yc": float(yc), "w": float(w), "h": float(h)},
            })

        # 写入 YOLO 标注目录
        (output_dir / f"{img_stem}.txt").write_text("\n".join(out_labels) + "\n")

    # 距离统计
    if diag_distances:
        arr = np.array(diag_distances)
        dist_stats = {
            "min": float(arr.min()),
            "p10": float(np.percentile(arr, 10)),
            "p25": float(np.percentile(arr, 25)),
            "p50": float(np.percentile(arr, 50)),
            "p75": float(np.percentile(arr, 75)),
            "p90": float(np.percentile(arr, 90)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
            "max": float(arr.max()),
        }
    else:
        dist_stats = {}

    reject_rate = rejected_bboxes / total_bboxes if total_bboxes > 0 else 0.0
    logger.info(
        f"[Inference] total={total_bboxes}  rejected={rejected_bboxes} ({reject_rate:.1f}%)"
    )
    if dist_stats:
        logger.info(
            f"  distance: min={dist_stats['min']:.4f}  "
            f"P50={dist_stats['p50']:.4f}  "
            f"P90={dist_stats['p90']:.4f}  "
            f"max={dist_stats['max']:.4f}"
        )

    summary = {
        "total": total_bboxes,
        "rejected": rejected_bboxes,
        "rejected_rate": reject_rate,
        "rejection_strategy": rejection_strategy,
        "rejection_threshold": rejection_threshold,
        "distance_metric": "euclidean",
        "class_ids": sorted_ids,
        "distance_stats": dist_stats,
    }

    return detailed_results, summary


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Structure Enhancement 推理")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="checkpoint .pth 文件路径")
    parser.add_argument("--bbox_dir", type=str, required=True,
                        help="Stage-1 class-agnostic bbox 标注目录")
    parser.add_argument("--image_dir", type=str, required=True,
                        help="推理图像目录")
    parser.add_argument("--train_dir", type=str, required=True,
                        help="训练集目录（每类一个子文件夹，用于构建原型）")
    parser.add_argument("--run_name", type=str, default="structure_enhance_inference",
                        help="本次推理的名称（决定输出子目录）")
    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--rejection_threshold", type=float, default=0.25)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    output_root = PROJECT_ROOT / "results" / "stage_2" / args.run_name
    inference_dir = output_root / "inference_results"
    detailed_dir = output_root / "detailed_results"
    detailed_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    logger.info(f"Output: {output_root}")

    transform = make_transform()

    # ── 1. 加载模型 ───────────────────────────────────────────────────────────
    logger.info(f"[1/3] 加载模型: {args.checkpoint}")
    model = load_model(Path(args.checkpoint), args.embed_dim, device)

    # ── 2. 构建原型 ───────────────────────────────────────────────────────────
    logger.info(f"[2/3] 从训练集构建原型: {args.train_dir}")
    prototypes, proto_stats = compute_prototypes(
        model, Path(args.train_dir), transform, device, batch_size=args.batch_size,
    )

    # ── 3. 推理 ───────────────────────────────────────────────────────────────
    logger.info(f"[3/3] 在 Stage-1 bbox 上推理: {args.bbox_dir}")
    detailed_results, summary = run_inference(
        model, prototypes,
        Path(args.bbox_dir),
        Path(args.image_dir),
        inference_dir,
        transform, device,
        rejection_threshold=args.rejection_threshold,
    )

    # ── 4. 保存详细结果 ───────────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # JSON
    json_path = detailed_dir / f"predictions_{timestamp}.json"
    json_out = {"summary": summary, "predictions": detailed_results}
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_out, f, indent=2, ensure_ascii=False)
    logger.info(f"详细 JSON: {json_path}")

    # CSV
    csv_path = detailed_dir / f"predictions_{timestamp}.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("image_name,predicted_class,distance,gap,is_unknown,top3\n")
        for r in detailed_results:
            top3_str = ";".join(f"cls{c}:{d:.4f}" for c, d in r.get("top3", []))
            f.write(f"{r['image_name']},{r['predicted_class']},{r['distance']:.4f},"
                    f"{r.get('gap', 0):.4f},{r['is_unknown']},{top3_str}\n")
    logger.info(f"CSV: {csv_path}")

    # ── 5. 日志输出 ──────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info(f"Structure Enhancement 推理完成")
    logger.info(f"  Checkpoint: {args.checkpoint}")
    logger.info(f"  原型类别: {len(prototypes)} 类")
    logger.info(f"  总 bbox: {summary['total']}")
    logger.info(f"  拒识 bbox: {summary['rejected']} ({summary['rejected_rate']*100:.1f}%)")
    logger.info(f"  输出目录: {output_root}")
    logger.info("=" * 60)

    if summary.get("distance_stats"):
        ds = summary["distance_stats"]
        logger.info(
            f"距离分布: P50={ds.get('p50',0):.4f}  "
            f"P90={ds.get('p90',0):.4f}  "
            f"max={ds.get('max',0):.4f}"
        )

    logger.info(
        f"\n下一步：将 {inference_dir} 和 {detailed_dir} "
        "送入 evaluation.py 计算指标。"
    )


if __name__ == "__main__":
    main()
