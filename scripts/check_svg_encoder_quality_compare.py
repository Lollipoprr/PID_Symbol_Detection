"""
SVGEncoder 质量对比诊断脚本（支持多模型 × 多数据集）。

用法：
    cd /media/wit/SSD_5151/wxr/Long-CLIP/PID_Symbol_Detection
    python -m src.scripts.check_svg_encoder_quality_compare

默认配置：
    模型：models/stage2/baseline / models/stage2/svg_guided
          （相对于 _BASE = PID_Symbol_Detection）
    数据：data/processed/stage_2/train / data/processed/stage_2/svg_to_png
          （相对于 _BASE）
    输出：results/compare_encoder_qualityen
          （相对于 workspace = experiment/）
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from PIL import Image
from tqdm import tqdm
from sklearn.manifold import TSNE

warnings.filterwarnings("ignore")

# ── 确保 src/ 在路径中 ─────────────────────────────────────────────────────────
_SRC = Path(__file__).resolve().parent.parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Device] {DEVICE}")

# 全局基准路径（与原脚本保持一致）
_BASE = Path("/media/wit/SSD_5151/wxr/Long-CLIP/PID_Symbol_Detection")


# ══════════════════════════════════════════════════════════════════════════════════
# SVGEncoder（与训练代码完全一致）
# ══════════════════════════════════════════════════════════════════════════════════

class SVGEncoder(nn.Module):
    """与 SVGEmbeddingNet 训练代码完全一致的 backbone + FC。

    Args:
        pretrained_ckpt: 若传入 checkpoint 路径，则 backbone 从此路径加载，
                        跳过 ImageNet 预训练初始化。
    """

    def __init__(self, num_classes=63, embedding_size=64,
                 pretrained_ckpt: str | Path | None = None):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_size = embedding_size

        bb_name = "swin_tiny_patch4_window7_224"
        self.backbone = timm.create_model(
            bb_name,
            num_classes=0,
            global_pool="avg",
            pretrained=False,
        )

        bb_loaded = False

        # 若传入了训练好的 checkpoint，直接从那里加载 backbone，跳过预训练
        if pretrained_ckpt is not None:
            ckpt_path = _BASE / pretrained_ckpt
            if ckpt_path.exists():
                ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                if isinstance(ckpt, dict):
                    inner = None
                    for _key in ("model_state", "state_dict", "model"):
                        if _key in ckpt and isinstance(ckpt[_key], dict):
                            inner = ckpt[_key]
                            break
                    if inner is None:
                        inner = ckpt
                    sd = {}
                    for k, v in inner.items():
                        if k.startswith("embedding_net.backbone.") or k.startswith("backbone."):
                            prefix = ("embedding_net.backbone."
                                      if k.startswith("embedding_net.backbone.")
                                      else "backbone.")
                            sd[k[len(prefix):]] = v
                    if sd:
                        self.backbone.load_state_dict(sd, strict=False)
                        bb_loaded = True
                        print(f"  → Backbone 从训练 checkpoint 加载: {ckpt_path}")

        # 若还没加载过任何 backbone（没有传 ckpt，或 ckpt 不存在），才用 ImageNet 预训练
        if not bb_loaded:
            local_ckpt = _BASE / "swin_tiny_patch4_window7_224" / "pytorch_model.bin"
            if local_ckpt.exists():
                sd = torch.load(local_ckpt, map_location="cpu", weights_only=True)
                if isinstance(sd, dict) and "model" in sd:
                    sd = sd["model"]
                self.backbone.load_state_dict(sd, strict=False)
                print(f"  → Backbone 从本地 ImageNet 权重加载: {local_ckpt}")
            else:
                self.backbone = timm.create_model(
                    bb_name, num_classes=0, global_pool="avg", pretrained=True,
                )
                print(f"  → Backbone 从 timm 预训练权重加载")

        self.fc = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, embedding_size),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        emb = self.fc(feat)
        return F.normalize(emb, p=2, dim=1)


# ══════════════════════════════════════════════════════════════════════════════════
# 距离函数
# ══════════════════════════════════════════════════════════════════════════════════

def cosine_distance_matrix(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x_l2 = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-10)
    cos_sim = x_l2 @ x_l2.T
    return np.clip(1.0 - cos_sim, 0.0, 2.0)


# ══════════════════════════════════════════════════════════════════════════════════
# 指标计算
# ══════════════════════════════════════════════════════════════════════════════════

def davies_bouldin(embeddings: np.ndarray, labels: np.ndarray) -> float:
    unique = np.unique(labels)
    k = len(unique)
    if k <= 1:
        return 0.0

    centroids, intra_s = [], []
    for lbl in unique:
        mask = labels == lbl
        cls_emb = embeddings[mask]
        if len(cls_emb) < 2:
            intra_s.append(0.0)
            centroids.append(cls_emb.mean(axis=0))
            continue
        dists = cosine_distance_matrix(cls_emb)
        triu_idx = np.triu_indices_from(dists, k=1)
        intra_s.append(dists[triu_idx].mean())
        centroids.append(cls_emb.mean(axis=0))

    centroids = np.stack(centroids, axis=0)
    cent_dist_mat = cosine_distance_matrix(centroids)

    db_parts = []
    for i in range(k):
        si = intra_s[i]
        if si < 1e-10:
            continue
        max_r = max(
            (si + intra_s[j]) / cent_dist_mat[i, j]
            for j in range(k) if i != j and cent_dist_mat[i, j] > 1e-10
        ) or 0.0
        db_parts.append(max_r)

    return float(np.mean(db_parts)) if db_parts else float("inf")


def silhouette_score(embeddings: np.ndarray, labels: np.ndarray) -> float:
    n = len(labels)
    dists = cosine_distance_matrix(embeddings)
    a = np.zeros(n)
    b = np.zeros(n)
    unique = np.unique(labels)
    lbl_idx_map = {lbl: np.where(labels == lbl)[0] for lbl in unique}

    for i in range(n):
        cur = labels[i]
        same = lbl_idx_map[cur]
        same = same[same != i]
        a[i] = dists[i, same].mean() if len(same) > 0 else 0.0
        others = [
            dists[i, lbl_idx_map[olbl]].mean()
            for olbl in unique if olbl != cur and len(lbl_idx_map[olbl]) > 0
        ]
        b[i] = min(others) if others else 0.0

    with np.errstate(divide="ignore", invalid="ignore"):
        s = (b - a) / np.maximum(np.maximum(a, b), 1e-10)
    return float(np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0).mean())


def intra_inter_stats(embeddings: np.ndarray, labels: np.ndarray) -> dict:
    unique = np.unique(labels)
    ids, lbs = [], []
    for lbl in unique:
        idx = np.where(labels == lbl)[0]
        ids.extend(idx)
        lbs.extend([lbl] * len(idx))

    samp = embeddings[ids]
    lab = np.array(lbs)
    dist_mat = cosine_distance_matrix(samp)
    n = len(lab)

    intra_d, inter_d = [], []
    for i in range(n):
        for j in range(i + 1, n):
            d = dist_mat[i, j]
            (intra_d if lab[i] == lab[j] else inter_d).append(d)

    intra_to_center = []
    for lbl in unique:
        idx = np.where(lab == lbl)[0]
        cls_emb = samp[idx]
        center = cls_emb.mean(axis=0, keepdims=True)
        center_l2 = center / (np.linalg.norm(center, axis=1, keepdims=True) + 1e-10)
        emb_l2 = cls_emb / (np.linalg.norm(cls_emb, axis=1, keepdims=True) + 1e-10)
        cos_sim = (emb_l2 * center_l2).sum(axis=1)
        dists = 1.0 - cos_sim
        intra_to_center.extend(dists.tolist())

    centroids = np.stack([samp[np.where(lab == l)[0]].mean(axis=0) for l in unique], axis=0)
    centroids_l2 = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-10)
    inter_center_dists = []
    for i in range(len(unique)):
        for j in range(i + 1, len(unique)):
            cos_sim = (centroids_l2[i] * centroids_l2[j]).sum()
            inter_center_dists.append(float(1.0 - cos_sim))

    return {
        "intra_pair_mean": float(np.mean(intra_d)) if intra_d else 0.0,
        "intra_pair_std":  float(np.std(intra_d)) if intra_d else 0.0,
        "inter_pair_mean": float(np.mean(inter_d)) if inter_d else 0.0,
        "inter_pair_std":  float(np.std(inter_d)) if inter_d else 0.0,
        "pair_ratio":      float(np.mean(intra_d) / (np.mean(inter_d) + 1e-10)) if intra_d and inter_d else 0.0,
        "n_intra_pairs": len(intra_d),
        "n_inter_pairs": len(inter_d),
        "intra_to_center_mean": float(np.mean(intra_to_center)) if intra_to_center else 0.0,
        "intra_to_center_std":  float(np.std(intra_to_center)) if intra_to_center else 0.0,
        "inter_center_mean": float(np.mean(inter_center_dists)) if inter_center_dists else 0.0,
        "inter_center_std":  float(np.std(inter_center_dists)) if inter_center_dists else 0.0,
    }


# ══════════════════════════════════════════════════════════════════════════════════
# 可视化
# ══════════════════════════════════════════════════════════════════════════════════

def plot_tsne(embeddings: np.ndarray, labels: np.ndarray,
              out_path: Path, title: str, max_points: int = 3000):
    n = len(labels)
    if n > max_points:
        idx = np.random.choice(n, max_points, replace=False)
        idx.sort()
        emb_vis = embeddings[idx]
        lab_vis = labels[idx]
    else:
        emb_vis, lab_vis = embeddings, labels

    perp = min(30, len(emb_vis) - 1)
    tsne = TSNE(n_components=2, random_state=42, perplexity=perp, metric="cosine")
    emb_2d = tsne.fit_transform(emb_vis)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    unique = np.unique(lab_vis)
    n_cls = len(unique)
    hues = np.linspace(0.0, 1.0, n_cls, endpoint=False)
    colors = [plt.cm.hsv(h) for h in hues]

    fig, ax = plt.subplots(figsize=(16, 13))
    for i, lbl in enumerate(unique):
        mask = lab_vis == lbl
        ax.scatter(emb_2d[mask, 0], emb_2d[mask, 1],
                   c=[colors[i]], s=25, alpha=0.85,
                   label=f"C{lbl}", linewidths=0)

    ax.set_title(title, fontsize=12)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend(loc="best", fontsize=6, ncol=min(10, n_cls),
              markerscale=1.5, framealpha=0.9)
    ax.tick_params(labelsize=7)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → t-SNE 已保存: {out_path}")


def plot_distance_heatmap(dist_mat: np.ndarray, class_ids: np.ndarray,
                           out_path: Path, title: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(dist_mat, cmap="RdYlGn_r", aspect="auto",
                   vmin=0.0, vmax=2.0)
    n = len(class_ids)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_ids, fontsize=7, rotation=90)
    ax.set_yticklabels(class_ids, fontsize=7)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Class ID")
    ax.set_ylabel("Class ID")
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("Cosine Distance")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → 热力图已保存: {out_path}")


def plot_metric_summary(stats: dict, out_path: Path, model_name: str, data_name: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 6))
    metrics = {
        "DB Index\n(越小越好)": stats["db_score"],
        "Silhouette\n(越大越好)": stats["silhouette"],
        "Intra/Inter Ratio\n(越小越好)": stats["pair_ratio"],
        "Intra→Center\n(越小越好)": stats["intra_to_center_mean"],
    }
    names = list(metrics.keys())
    vals = list(metrics.values())
    colors = ["#e74c3c", "#2ecc71", "#e67e22", "#9b59b6"]

    bars = ax.bar(names, vals, color=colors, width=0.5, edgecolor="white")
    for bar, val in zip(bars, vals):
        label = f"{val:.4f}" if val == val and val != float("inf") else ("nan" if val != val else "inf")
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                label, ha="center", va="bottom", fontsize=11)

    finite = [v for v in vals if v == v and v != float("inf")]
    ax.set_ylim(0, max(finite) * 1.2 if finite else 1.0)
    ax.set_title(f"SVG Encoder Quality Metrics  [{model_name} / {data_name}]", fontsize=13)
    ax.set_ylabel("Score")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → 指标图已保存: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════════
# 单次评估（一个模型 × 一个数据集）
# ══════════════════════════════════════════════════════════════════════════════════

def evaluate(
    svg_encoder: SVGEncoder,
    svg_dir: Path,
    out_dir: Path,
    embedding_size: int = 64,
    num_classes: int = 63,
    max_per_class: int = 100,
    model_name: str = "model",
    data_name: str = "data",
) -> dict:
    """对指定数据和模型执行完整评估流程。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    svg_dir = Path(svg_dir)

    transform = timm.data.create_transform(
        input_size=224, is_training=False,
        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
    )

    # ── 收集 PNG 路径 ────────────────────────────────────────────────────────
    class_pngs: dict = {}
    for cls_folder in sorted(svg_dir.iterdir()):
        if not cls_folder.is_dir():
            continue
        try:
            cid = int(cls_folder.name)
        except ValueError:
            continue
        pngs = sorted(cls_folder.glob("*.png")) + sorted(cls_folder.glob("*.jpg"))
        if max_per_class > 0:
            pngs = pngs[:max_per_class]
        if pngs:
            class_pngs[cid] = pngs

    if not class_pngs:
        raise ValueError(f"No data found in {svg_dir}")

    # ── 批量提取 embeddings ──────────────────────────────────────────────────
    print(f"\n  提取 embeddings（batch=64）...")
    all_embs: list = []
    all_labels: list = []
    batch_imgs: list = []
    batch_labels: list = []

    for cid, paths in sorted(class_pngs.items()):
        for p in paths:
            img = Image.open(p).convert("RGB")
            tensor = transform(img)
            batch_imgs.append(tensor)
            batch_labels.append(cid)

            if len(batch_imgs) >= 64:
                batch = torch.stack(batch_imgs).to(DEVICE)
                with torch.no_grad():
                    embs = svg_encoder.encode(batch).cpu()
                all_embs.append(embs)
                all_labels.extend(batch_labels)
                batch_imgs.clear()
                batch_labels.clear()

        if batch_imgs:
            batch = torch.stack(batch_imgs).to(DEVICE)
            with torch.no_grad():
                embs = svg_encoder.encode(batch).cpu()
            all_embs.append(embs)
            all_labels.extend(batch_labels)
            batch_imgs.clear()
            batch_labels.clear()

    all_emb_np = torch.cat(all_embs, dim=0).numpy().astype(np.float64)
    all_labels_np = np.array(all_labels, dtype=np.int64)
    print(f"  → Embeddings shape: {all_emb_np.shape}")

    # ── 计算类中心 ────────────────────────────────────────────────────────────
    unique_ids = np.unique(all_labels_np)
    centroids_list: list = []
    for cid in unique_ids:
        mask = all_labels_np == cid
        center = all_emb_np[mask].mean(axis=0)
        centroids_list.append(center)
    centroids_np = np.stack(centroids_list, axis=0).astype(np.float64)

    # ── 计算指标 ─────────────────────────────────────────────────────────────
    db = davies_bouldin(all_emb_np, all_labels_np)
    sil = silhouette_score(all_emb_np, all_labels_np)
    stats = intra_inter_stats(all_emb_np, all_labels_np)
    dist_mat = cosine_distance_matrix(centroids_np)

    print(f"""
  ╔{'═' * 54}║
  ║  [{model_name} / {data_name}]                            ║
  ║  Davies-Bouldin Index:     {db:>8.4f}  (越小越好)    ║
  ║  Silhouette Score:         {sil:>8.4f}  (越大越好)    ║
  ╠{'─' * 54}╣
  ║  Intra  mean ± std:       {stats['intra_pair_mean']:.4f} ± {stats['intra_pair_std']:.4f}   ║
  ║  Inter  mean ± std:       {stats['inter_pair_mean']:.4f} ± {stats['inter_pair_std']:.4f}   ║
  ║  Intra/Inter Ratio:        {stats['pair_ratio']:>8.4f}  (越小越好)    ║
  ╠{'─' * 54}╣
  ║  Intra→Center mean±std:  {stats['intra_to_center_mean']:.4f} ± {stats['intra_to_center_std']:.4f}   ║
  ║  Inter-Center mean ± std: {stats['inter_center_mean']:.4f} ± {stats['inter_center_std']:.4f}   ║
  ╚{'═' * 54}╝
    """)

    # ── 可视化 ────────────────────────────────────────────────────────────────
    print("  生成可视化...")
    plot_tsne(
        all_emb_np, all_labels_np,
        out_dir / "tsne_all_samples.png",
        title=(
            f"t-SNE: {model_name} — All Samples [{data_name}]\n"
            f"DB={db:.4f}, Sil={sil:.4f}, "
            f"Intra/Center={stats['intra_to_center_mean']:.4f}, "
            f"{len(unique_ids)} classes, {len(all_labels_np)} samples"
        ),
    )

    plot_tsne(
        centroids_np, unique_ids,
        out_dir / "tsne_prototypes.png",
        title=(
            f"t-SNE: {model_name} — Class Prototypes [{data_name}]\n"
            f"DB={db:.4f}, Sil={sil:.4f}, "
            f"Inter-Center={stats['inter_center_mean']:.4f}, "
            f"{len(unique_ids)} classes"
        ),
    )

    plot_distance_heatmap(
        dist_mat, unique_ids,
        out_dir / "proto_distance_heatmap.png",
        title=(
            f"Prototype Cosine Distances: {model_name} [{data_name}]\n"
            f"DB={db:.4f}, Sil={sil:.4f}, Inter-Center={stats['inter_center_mean']:.4f}"
        ),
    )

    plot_metric_summary(
        {**stats, "db_score": db, "silhouette": sil},
        out_dir / "metric_summary.png",
        model_name=model_name,
        data_name=data_name,
    )

    # ── 保存 JSON ─────────────────────────────────────────────────────────────
    results = {
        "model_name": model_name,
        "data_name": data_name,
        "embedding_size": embedding_size,
        "num_classes": len(unique_ids),
        "total_samples": int(len(all_labels_np)),
        "samples_per_class": float(len(all_labels_np) / len(unique_ids)),
        "db_score": db,
        "silhouette": sil,
        **stats,
    }

    json_path = out_dir / "metrics.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  → JSON 已保存: {json_path}")

    return results


# ══════════════════════════════════════════════════════════════════════════════════
# 主脚本
# ══════════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="SVGEncoder 质量对比诊断（多模型 × 多数据集）"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=[
            "models/stage2/baseline/best_model.pth",
            "models/stage2/phase2b/best_backbone.pth",
        ],
        help="模型 checkpoint 路径列表（相对于项目根目录，与 --model_names 一一对应）",
    )
    parser.add_argument(
        "--model_names",
        nargs="+",
        default=["baseline", "svg_guided"],
        help="模型展示名称列表",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=[
            "data/processed/stage_2/train",
            "data/processed/stage_2/svg_to_png",
        ],
        help="数据目录列表（与 --data_names 一一对应）",
    )
    parser.add_argument(
        "--data_names",
        nargs="+",
        default=["train", "svg_to_png"],
        help="数据集展示名称列表",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/phase2a",
        help="输出根目录（相对于 workspace，即 experiment/；子目录按 {model_name}__{data_name} 命名）",
    )
    parser.add_argument("--embedding_size", type=int, default=64)
    parser.add_argument("--num_classes", type=int, default=63)
    parser.add_argument("--max_per_class", type=int, default=100,
                        help="每类最多取几张 PNG（0=不限）")
    args = parser.parse_args()

    assert len(args.models) == len(args.model_names), \
        "--models 与 --model_names 数量必须一致"
    assert len(args.datasets) == len(args.data_names), \
        "--datasets 与 --data_names 数量必须一致"

    base_out = _SRC / args.output
    base_out.mkdir(parents=True, exist_ok=True)

    # ── 全局汇总表 ──────────────────────────────────────────────────────────
    all_results: list = []

    # ── 构建 SVGEncoder ───────────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("构建 SVGEncoder")
    print("=" * 62)

    # ── 遍历模型 ─────────────────────────────────────────────────────────────
    for model_path_str, model_name in zip(args.models, args.model_names):
        model_path = _BASE / model_path_str
        print(f"\n{'=' * 62}")
        print(f"模型: {model_name}  →  {model_path}")
        print("=" * 62)

        # 直接通过 pretrained_ckpt 参数传入，SVGEncoder 内部会从训练 checkpoint
        # 加载 backbone + fc，跳过 ImageNet 预训练初始化
        svg_encoder = SVGEncoder(
            num_classes=args.num_classes,
            embedding_size=args.embedding_size,
            pretrained_ckpt=model_path_str if model_path.exists() else None,
        ).to(DEVICE)
        svg_encoder.eval()

        if not model_path.exists():
            print(f"  ⚠️  checkpoint 不存在: {model_path}，使用随机初始化！")

        # 若 checkpoint 存在但 SVGEncoder 内部没加载到 fc（罕见的 key 格式问题），
        # 这里再做一次兜底覆盖
        if model_path.exists():
            ckpt = torch.load(model_path, map_location=DEVICE, weights_only=False)
            if isinstance(ckpt, dict):
                inner = None
                for _key in ("model_state", "state_dict", "model"):
                    if _key in ckpt and isinstance(ckpt[_key], dict):
                        inner = ckpt[_key]
                        break
                if inner is None:
                    inner = ckpt
                sd_fc = {}
                for k, v in inner.items():
                    if k.startswith("embedding_net.fc.") or k.startswith("fc."):
                        prefix = ("embedding_net.fc."
                                  if k.startswith("embedding_net.fc.") else "fc.")
                        sd_fc[k[len(prefix):]] = v
                if sd_fc:
                    svg_encoder.fc.load_state_dict(sd_fc, strict=False)
                    print(f"  → 额外覆盖 embedding_net.fc ({len(sd_fc)} 层)")

        # ── 遍历数据集 ────────────────────────────────────────────────────────
        for data_path_str, data_name in zip(args.datasets, args.data_names):
            data_path = _BASE / data_path_str
            combo_name = f"{model_name}__{data_name}"
            out_dir = base_out / combo_name

            print(f"\n  ── {combo_name} ──")
            print(f"      数据目录: {data_path}")

            try:
                result = evaluate(
                    svg_encoder=svg_encoder,
                    svg_dir=_BASE / data_path_str,
                    out_dir=out_dir,
                    embedding_size=args.embedding_size,
                    num_classes=args.num_classes,
                    max_per_class=args.max_per_class,
                    model_name=model_name,
                    data_name=data_name,
                )
                all_results.append(result)
            except Exception as e:
                print(f"  ⚠️  评估失败: {e}")
                all_results.append({
                    "model_name": model_name,
                    "data_name": data_name,
                    "error": str(e),
                })

    # ── 汇总对比表 ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("汇总对比")
    print("=" * 70)

    # 保存汇总 CSV
    import csv
    csv_path = base_out / "summary.csv"
    fieldnames = [
        "model_name", "data_name", "db_score", "silhouette",
        "pair_ratio", "intra_to_center_mean", "inter_center_mean",
        "num_classes", "total_samples",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_results:
            if "error" not in r:
                writer.writerow({k: r.get(k) for k in fieldnames})
    print(f"\n汇总 CSV: {csv_path}")

    # 保存汇总 JSON
    summary_json_path = base_out / "summary.json"
    with open(summary_json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"汇总 JSON: {summary_json_path}")

    # 打印表格
    print(f"""
  {'模型':<12} {'数据集':<14} {'DB↓':>7} {'Sil↑':>7} {'IntraR↓':>9} {'Intra→C↓':>9} {'InterC↑':>9} {'样本数':>7}
  {'-' * 78}""")
    for r in all_results:
        if "error" not in r:
            print(
                f"  {r['model_name']:<12} {r['data_name']:<14} "
                f"{r['db_score']:>7.4f} {r['silhouette']:>7.4f} "
                f"{r['pair_ratio']:>9.4f} {r['intra_to_center_mean']:>9.4f} "
                f"{r['inter_center_mean']:>9.4f} {r['total_samples']:>7d}"
            )

    print("\n  参考标准：")
    print("    DB Score:        <2 优秀，<4 良好，>4 较差")
    print("    Silhouette:       >0.5 优秀，>0.25 良好，<0.25 较差")
    print("    Intra/Center:    <0.3 优秀，<0.5 良好")
    print("    Inter-Center:    >1.0 优秀，>0.7 良好")


if __name__ == "__main__":
    main()
