"""
Stage 2: Structure Enhancement Training Pipeline (基于 Baseline 改进)

核心设计：
- Photo和SVG共用同一个网络骨干，无模态差异
- 仅使用Triplet Loss训练
- StructureEnhancer放在Stage1(blocks)之后、PatchMerging之前
- PositionInvariantDescriptor放在最后

三层训练指标体系：
  Layer1 (每个step): d_ap_cross, d_ap_intra, d_an, active_triplet_ratio
  Layer2 (每个epoch): SE权重分布, margin_gap, 训练状态
  Layer3 (每N epoch): t-SNE可视化, Fisher Ratio

参考：
- Baseline: stage2_baseline_train.py
- 模型: svg_structure_enhance_net.py
- 数据集: svg_triplet_dataset.py
"""

import pickle
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from models.svg_structure_enhance_net import SVGStructureEnhanceNet, TripletNet, StructureEnhancerSENet
from pipeline.base import BasePipeline
from utils.bbox_utils import BBoxUtils
from utils.class_crops_builder import ClassCropsBuilder
from utils.svg_triplet_dataset import HybridPhotoSVGTripletDataset, PhotoSVGTripletDataset
from utils.stage2_Siamese_Network import EmbeddingNet as BaselineEmbeddingNet
from utils.stage2_Siamese_Network import TripletNet as BaselineTripletNet


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}


# ─────────────────────────────────────────────────────────────────────────────
# Transform
# ─────────────────────────────────────────────────────────────────────────────
def pad_to_square(img: torch.Tensor) -> torch.Tensor:
    _, h, w = img.shape
    if h == w:
        return img
    s = max(h, w)
    padded = torch.ones(img.shape[0], s, s, dtype=img.dtype, device=img.device)
    top  = (s - h) // 2
    left = (s - w) // 2
    padded[:, top:top+h, left:left+w] = img
    return padded


def _resolve_project_path(path_like: str | Path | None) -> Optional[Path]:
    if not path_like:
        return None
    path = Path(path_like)
    if path.is_absolute():
        return path
    return Path.cwd() / path


def _extract_state_dict(checkpoint_obj):
    if isinstance(checkpoint_obj, dict) and "model_state_dict" in checkpoint_obj:
        return checkpoint_obj["model_state_dict"]
    return checkpoint_obj


def load_baseline_teacher(
    checkpoint_path: str | Path,
    embedding_size: int,
    device: str,
) -> nn.Module:
    """Load the frozen baseline TripletNet used by preserve loss."""
    ckpt_path = _resolve_project_path(checkpoint_path)
    if ckpt_path is None or not ckpt_path.exists():
        raise FileNotFoundError(f"baseline teacher checkpoint not found: {ckpt_path}")

    embedding_net = BaselineEmbeddingNet(
        embedding_size=embedding_size,
        use_pretrained=False,
    )
    teacher = BaselineTripletNet(embedding_net)

    loaded = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = _extract_state_dict(loaded)
    missing, unexpected = teacher.load_state_dict(state_dict, strict=False)
    print(
        f"[BaselineTeacher] loaded: {ckpt_path} "
        f"(missing={len(missing)}, unexpected={len(unexpected)})"
    )

    teacher.to(device)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False
    return teacher


def load_student_backbone_from_baseline(
    model: nn.Module,
    checkpoint_path: str | Path,
    device: str,
) -> None:
    """Initialize compatible student backbone weights from baseline checkpoint."""
    ckpt_path = _resolve_project_path(checkpoint_path)
    if ckpt_path is None or not ckpt_path.exists():
        raise FileNotFoundError(f"baseline init checkpoint not found: {ckpt_path}")

    loaded = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = _extract_state_dict(loaded)
    current = model.state_dict()
    compatible = {
        key: value
        for key, value in state_dict.items()
        if key in current and current[key].shape == value.shape
    }
    current.update(compatible)
    model.load_state_dict(current, strict=True)
    print(
        f"[Stage2-SVG] initialized student from baseline: {ckpt_path} "
        f"({len(compatible)}/{len(current)} tensors matched)"
    )


def list_class_images(class_dir: Path) -> List[Path]:
    return sorted(
        path for path in class_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    )


# ─────────────────────────────────────────────────────────────────────────────
# Real validation evaluator
# ─────────────────────────────────────────────────────────────────────────────

class SVGRealValEvaluator:
    """
    Prototype-based validation matching the stage-2 inference setting.

    It builds class prototypes from train_dir with the current embedding model,
    then classifies val_dir crops by nearest prototype. This is closer to the
    final inference path than triplet validation loss, so it is suitable for
    selecting best.pth.
    """

    def __init__(
        self,
        train_dir: Path,
        val_dir: Path,
        transform,
        device: str,
        max_train_per_class: Optional[int] = None,
        max_val_per_class: Optional[int] = None,
        batch_size: int = 64,
    ):
        self.train_dir = Path(train_dir)
        self.val_dir = Path(val_dir)
        self.transform = transform
        self.device = device
        self.max_train_per_class = max_train_per_class
        self.max_val_per_class = max_val_per_class
        self.batch_size = int(batch_size)

    def evaluate(self, embedding_net: nn.Module) -> Dict[str, float]:
        embedding_net.eval()
        prototypes = self._build_prototypes(embedding_net)
        if not prototypes:
            return {
                'real_val_acc': 0.0,
                'real_val_macro_precision': 0.0,
                'real_val_macro_recall': 0.0,
                'real_val_macro_f1': 0.0,
                'real_val_known_p95': 0.0,
                'real_val_samples': 0,
                'real_val_classes': 0,
            }

        sorted_ids = sorted(prototypes.keys())
        proto_matrix = torch.stack([prototypes[cid] for cid in sorted_ids]).to(self.device)

        tp: Dict[int, int] = {cid: 0 for cid in sorted_ids}
        fp: Dict[int, int] = {cid: 0 for cid in sorted_ids}
        fn: Dict[int, int] = {cid: 0 for cid in sorted_ids}
        distances: List[float] = []
        total = 0
        correct = 0
        classes_seen = set()

        for class_dir in sorted(self.val_dir.iterdir()):
            if not class_dir.is_dir() or not class_dir.name.isdigit():
                continue
            gt_cls = int(class_dir.name)
            if gt_cls not in prototypes:
                continue
            image_paths = list_class_images(class_dir)
            if not image_paths:
                continue
            if self.max_val_per_class and len(image_paths) > self.max_val_per_class:
                image_paths = random.sample(image_paths, self.max_val_per_class)

            classes_seen.add(gt_cls)
            for emb in self._embed_images(embedding_net, image_paths):
                dists = torch.norm(emb - proto_matrix, dim=1)
                min_idx = int(torch.argmin(dists).item())
                pred_cls = int(sorted_ids[min_idx])
                min_dist = float(dists[min_idx].item())
                distances.append(min_dist)

                total += 1
                if pred_cls == gt_cls:
                    correct += 1
                    tp[gt_cls] = tp.get(gt_cls, 0) + 1
                else:
                    fp[pred_cls] = fp.get(pred_cls, 0) + 1
                    fn[gt_cls] = fn.get(gt_cls, 0) + 1

        f1_values = []
        precision_values = []
        recall_values = []
        for cls_id in sorted(classes_seen):
            cls_tp = tp.get(cls_id, 0)
            cls_fp = fp.get(cls_id, 0)
            cls_fn = fn.get(cls_id, 0)
            precision = cls_tp / (cls_tp + cls_fp) if (cls_tp + cls_fp) > 0 else 0.0
            recall = cls_tp / (cls_tp + cls_fn) if (cls_tp + cls_fn) > 0 else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )
            precision_values.append(precision)
            recall_values.append(recall)
            f1_values.append(f1)

        dist_arr = np.array(distances, dtype=np.float32)
        return {
            'real_val_acc': float(correct / total) if total else 0.0,
            'real_val_macro_precision': float(np.mean(precision_values)) if precision_values else 0.0,
            'real_val_macro_recall': float(np.mean(recall_values)) if recall_values else 0.0,
            'real_val_macro_f1': float(np.mean(f1_values)) if f1_values else 0.0,
            'real_val_known_p50': float(np.percentile(dist_arr, 50)) if len(dist_arr) else 0.0,
            'real_val_known_p90': float(np.percentile(dist_arr, 90)) if len(dist_arr) else 0.0,
            'real_val_known_p95': float(np.percentile(dist_arr, 95)) if len(dist_arr) else 0.0,
            'real_val_samples': int(total),
            'real_val_classes': int(len(classes_seen)),
        }

    def _build_prototypes(self, embedding_net: nn.Module) -> Dict[int, torch.Tensor]:
        prototypes: Dict[int, torch.Tensor] = {}
        for class_dir in sorted(self.train_dir.iterdir()):
            if not class_dir.is_dir() or not class_dir.name.isdigit():
                continue
            cls_id = int(class_dir.name)
            image_paths = list_class_images(class_dir)
            if not image_paths:
                continue
            if self.max_train_per_class and len(image_paths) > self.max_train_per_class:
                image_paths = random.sample(image_paths, self.max_train_per_class)

            embs = [
                emb.squeeze(0).cpu()
                for emb in self._embed_images(embedding_net, image_paths)
            ]
            if not embs:
                continue
            proto = torch.stack(embs).mean(dim=0, keepdim=True)
            proto = F.normalize(proto, p=2, dim=1).squeeze(0)
            prototypes[cls_id] = proto
        return prototypes

    def _embed_images(
        self,
        embedding_net: nn.Module,
        image_paths: List[Path],
    ) -> List[torch.Tensor]:
        outputs: List[torch.Tensor] = []
        batch_tensors: List[torch.Tensor] = []

        def flush_batch():
            if not batch_tensors:
                return
            batch = torch.stack(batch_tensors).to(self.device)
            with torch.no_grad():
                embs = F.normalize(embedding_net(batch), p=2, dim=1)
            outputs.extend(emb.unsqueeze(0) for emb in embs)
            batch_tensors.clear()

        for img_path in image_paths:
            try:
                image = Image.open(img_path).convert("RGB")
                arr = np.array(image)
                tensor = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
                tensor = self.transform(tensor)
                batch_tensors.append(tensor)
            except Exception as exc:
                print(f"[RealVal] 跳过损坏图片 {img_path}: {exc}")
                continue

            if len(batch_tensors) >= self.batch_size:
                flush_batch()

        flush_batch()
        return outputs


class ProxyUnknownPool:
    """Random sampler for proxy-unknown crops used by boundary loss."""

    def __init__(
        self,
        unknown_dir: str | Path,
        transform,
        max_samples: int = 300,
    ):
        self.unknown_dir = _resolve_project_path(unknown_dir)
        self.transform = transform
        self.max_samples = int(max_samples)
        self.image_paths: List[Path] = []
        self._build_index()

    def _build_index(self) -> None:
        if self.unknown_dir is None or not self.unknown_dir.exists():
            print(f"[ProxyUnknown] unknown_dir 不存在: {self.unknown_dir}")
            return

        all_images = sorted(
            path for path in self.unknown_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTS
        )
        if self.max_samples > 0 and len(all_images) > self.max_samples:
            self.image_paths = random.sample(all_images, self.max_samples)
        else:
            self.image_paths = all_images

        print(
            f"[ProxyUnknown] loaded {len(self.image_paths)}/{len(all_images)} "
            f"unknown crops from {self.unknown_dir}"
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def sample_batch(self, batch_size: int) -> Optional[torch.Tensor]:
        if not self.image_paths or batch_size <= 0:
            return None

        if len(self.image_paths) >= batch_size:
            paths = random.sample(self.image_paths, batch_size)
        else:
            paths = random.choices(self.image_paths, k=batch_size)

        tensors: List[torch.Tensor] = []
        for img_path in paths:
            try:
                image = Image.open(img_path).convert("RGB")
                arr = np.array(image)
                tensor = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
                tensor = self.transform(tensor)
                tensors.append(tensor)
            except Exception as exc:
                print(f"[ProxyUnknown] 跳过损坏图片 {img_path}: {exc}")

        if not tensors:
            return None
        return torch.stack(tensors)


# ─────────────────────────────────────────────────────────────────────────────
# 三层指标计算器
# ─────────────────────────────────────────────────────────────────────────────

class TrainingMetrics:
    """
    三层训练指标收集器。

    Layer1 (per-step): 核心距离
      - d_ap_cross: photo ↔ SVG 同类距离
      - d_ap_intra:  photo ↔ photo 同类距离（需要同类另一个photo样本）
      - d_an:        anchor ↔ negative 距离
      - active_ratio: loss > 0 的三元组占比

    Layer2 (per-epoch): 训练状态
      - se_weight_std: SE-Net 通道权重标准差
      - se_weight_max / min: 权重范围
      - margin_gap: d_an - d_ap_cross 均值

    Layer3 (periodic): 可视化与语义指标
      - t-SNE: photo/SVG 混合可视化
      - fisher_ratio: 类间距 / 类内距
    """

    def __init__(
        self,
        output_dir: Path,
        eval_interval: int = 5,
        tsne_samples: int = 500,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.eval_interval = eval_interval
        self.tsne_samples = tsne_samples

        # History
        self.history: List[Dict] = []

        # Per-epoch accumulators
        self._reset_epoch_accumulators()

    def _reset_epoch_accumulators(self):
        self._d_ap_cross_list: List[float] = []
        self._d_ap_intra_list: List[float] = []
        self._d_an_list: List[float] = []
        self._active_count = 0
        self._total_count = 0

    def step(
        self,
        anchor_emb,
        positive_emb,
        negative_emb,
        per_sample_loss: torch.Tensor,
        intra_pos_emb: Optional[torch.Tensor] = None,
    ):
        """
        Layer1: 每个 step 记录。
        anchor_emb: [B, D] photo
        positive_emb: [B, D] SVG
        negative_emb: [B, D] photo (异类)
        per_sample_loss: [B] TripletMarginLoss(reduction='none')，每个样本的 loss
        intra_pos_emb: [B, D] 与 anchor 同类的另一张 photo embedding
        """
        d_ap = F.pairwise_distance(anchor_emb, positive_emb)   # [B]
        d_an = F.pairwise_distance(anchor_emb, negative_emb)    # [B]

        self._d_ap_cross_list.extend(d_ap.detach().cpu().tolist())
        self._d_an_list.extend(d_an.detach().cpu().tolist())

        d_ap_intra = None
        if intra_pos_emb is not None:
            d_ap_intra = F.pairwise_distance(anchor_emb, intra_pos_emb)
            self._d_ap_intra_list.extend(d_ap_intra.detach().cpu().tolist())

        # active triplet: loss > 0（per_sample_loss 已是 [B]）
        mask = per_sample_loss.detach() > 0
        self._active_count += mask.sum().item()
        self._total_count += mask.numel()

        return {
            'd_ap_cross': d_ap.mean().item(),
            'd_ap_intra': d_ap_intra.mean().item() if d_ap_intra is not None else 0.0,
            'd_an': d_an.mean().item(),
            'active_ratio': mask.float().mean().item(),
        }

    def epoch_end(self, epoch: int, model: nn.Module, train_loader, device: torch.device):
        """
        Layer2: 每个 epoch 结束时调用。
        返回汇总指标。
        """
        d_ap_arr = np.array(self._d_ap_cross_list)
        d_ap_intra_arr = np.array(self._d_ap_intra_list)
        d_an_arr = np.array(self._d_an_list)

        sample_feats = None
        try:
            batch = next(iter(train_loader))
            anchor_photo = batch[0].to(device)
            sample_feats = model.embedding_net.extract_stage1_feature_map(anchor_photo)
        except Exception:
            sample_feats = None

        se_metrics = self._compute_se_metrics(model, sample_feats)

        metrics = {
            'epoch': epoch,
            # Layer1 汇总
            'd_ap_cross_mean': float(d_ap_arr.mean()),
            'd_ap_cross_std':  float(d_ap_arr.std()),
            'd_ap_cross_p50':  float(np.percentile(d_ap_arr, 50)),
            'd_ap_cross_p90':  float(np.percentile(d_ap_arr, 90)),
            'd_ap_intra_mean': float(d_ap_intra_arr.mean()) if len(d_ap_intra_arr) > 0 else 0.0,
            'd_ap_intra_std':  float(d_ap_intra_arr.std()) if len(d_ap_intra_arr) > 0 else 0.0,
            'd_an_mean':       float(d_an_arr.mean()),
            'd_an_std':        float(d_an_arr.std()),
            'active_triplet_ratio': self._active_count / max(self._total_count, 1),
            # Layer2
            **se_metrics,
        }

        # margin_gap
        if len(self._d_ap_cross_list) > 0 and len(self._d_an_list) > 0:
            metrics['margin_gap'] = metrics['d_an_mean'] - metrics['d_ap_cross_mean']
        else:
            metrics['margin_gap'] = 0.0

        if len(d_ap_intra_arr) > 0 and metrics['d_ap_intra_mean'] > 1e-8:
            metrics['cross_intra_ratio'] = metrics['d_ap_cross_mean'] / metrics['d_ap_intra_mean']
        else:
            metrics['cross_intra_ratio'] = 0.0

        self.history.append(metrics)

        # Layer3: periodic
        if (epoch + 1) % self.eval_interval == 0:
            self._eval_layer3(epoch, model, train_loader, device)

        self._reset_epoch_accumulators()
        return metrics

    def _compute_se_metrics(self, model: nn.Module, sample_feats: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """
        Layer2: SE-Net sigmoid 输出分布统计。

        监控的是实际的通道注意力权重（sigmoid 输出，值域 0~1），
        而不是 MLP 最后一层的线性权重参数。
        """
        if sample_feats is None:
            return {
                'se_weight_mean': 0.0,
                'se_weight_std':  0.0,
                'se_weight_max':  0.0,
                'se_weight_min':  0.0,
            }

        se_weights: List[float] = []
        for m in model.modules():
            if isinstance(m, StructureEnhancerSENet):
                w = m.get_sigmoid_weights(sample_feats)   # [B, C] sigmoid 输出
                se_weights.extend(w.detach().cpu().tolist())

        if se_weights:
            w = np.array(se_weights)
            return {
                'se_weight_mean': float(w.mean()),
                'se_weight_std':  float(w.std()),
                'se_weight_max':  float(w.max()),
                'se_weight_min':  float(w.min()),
            }
        return {
            'se_weight_mean': 0.0,
            'se_weight_std':  0.0,
            'se_weight_max':  0.0,
            'se_weight_min':  0.0,
        }

    def _eval_layer3(self, epoch: int, model: nn.Module, loader, device: torch.device):
        """Layer3: t-SNE 可视化 + Fisher Ratio"""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Collect sample embeddings (photo + SVG)
        photo_embs: List[np.ndarray] = []
        svg_embs: List[np.ndarray] = []
        labels: List[int] = []
        classes_seen: set = set()

        model.eval()
        batch_count = 0
        max_batches = max(1, self.tsne_samples // loader.batch_size)

        with torch.no_grad():
            for batch in loader:
                if batch_count >= max_batches:
                    break
                anchor_photo, positive_svg, label_A = self._unpack_eval_batch(batch)
                anchor_photo = anchor_photo.to(device)
                positive_svg = positive_svg.to(device)

                # embedding_net 是 TripletNet.embedding_net，即 SVGStructureEnhanceNet
                a_emb = model.embedding_net(anchor_photo)
                p_emb = model.embedding_net(positive_svg)

                photo_embs.append(a_emb.detach().cpu().numpy())
                svg_embs.append(p_emb.detach().cpu().numpy())
                labels.extend([int(label_A[i]) for i in range(len(label_A))])

                batch_count += 1
                classes_seen.update([int(label_A[i]) for i in range(len(label_A))])

        if not photo_embs:
            return

        photo_embs = np.concatenate(photo_embs, axis=0)   # [N, D]
        svg_embs   = np.concatenate(svg_embs,   axis=0)   # [N, D]
        labels     = np.array(labels)

        # Subsample if too many
        if len(photo_embs) > self.tsne_samples:
            idx = np.random.choice(len(photo_embs), self.tsne_samples, replace=False)
            photo_embs = photo_embs[idx]
            svg_embs   = svg_embs[idx]
            labels     = labels[idx]

        # Fisher Ratio: 类间距 / 类内距
        fisher = self._compute_fisher_ratio(photo_embs, labels)

        # t-SNE
        all_embs = np.concatenate([photo_embs, svg_embs], axis=0)
        modalities = np.array(['photo'] * len(photo_embs) + ['svg'] * len(svg_embs))
        combined_labels = np.concatenate([labels, labels])

        try:
            tsne = TSNE(n_components=2, perplexity=min(30, len(all_embs) - 1),
                        random_state=42, n_iter=1000)
            tsne_coords = tsne.fit_transform(all_embs)
        except Exception as e:
            print(f"  [Layer3] t-SNE failed: {e}")
            tsne_coords = None

        # Save
        vis_path = self.output_dir / f"tsne_epoch_{epoch+1:03d}.png"
        self._plot_tsne(tsne_coords, combined_labels, modalities, vis_path)

        # Save embedding data for later analysis
        np.savez(
            self.output_dir / f"embeddings_epoch_{epoch+1:03d}.npz",
            photo_embs=photo_embs,
            svg_embs=svg_embs,
            labels=labels,
        )

        # Print summary
        print(
            f"  [Layer3] epoch={epoch+1} | "
            f"Fisher Ratio={fisher:.4f} | "
            f"t-SNE saved to {vis_path.name}"
        )

    @staticmethod
    def _unpack_eval_batch(batch):
        if len(batch) == 6 and torch.is_tensor(batch[3]) and batch[3].ndim >= 4:
            anchor_photo, _positive_photo, positive_svg, _negative_photo, label_A, _label_N = batch
            return anchor_photo, positive_svg, label_A

        anchor_photo, positive_svg, _negative_photo, label_A, _label_P, _label_N = batch
        return anchor_photo, positive_svg, label_A

    def _compute_fisher_ratio(self, embeddings: np.ndarray, labels: np.ndarray) -> float:
        """Fisher Ratio = mean(类间距) / mean(类内距)"""
        unique_labels = np.unique(labels)
        if len(unique_labels) < 2:
            return 0.0

        class_means: List[np.ndarray] = []
        intra_dists: List[float] = []
        inter_dists: List[float] = []

        overall_mean = embeddings.mean(axis=0)

        for cls in unique_labels:
            mask = labels == cls
            cls_embs = embeddings[mask]
            if len(cls_embs) < 2:
                continue

            cls_mean = cls_embs.mean(axis=0)
            class_means.append(cls_mean)

            # 类内距
            centered = cls_embs - cls_mean
            intra = np.linalg.norm(centered, axis=1).mean()
            intra_dists.append(intra)

        # 类间距
        for i, mi in enumerate(class_means):
            for mj in class_means[i+1:]:
                inter = np.linalg.norm(mi - mj)
                inter_dists.append(inter)

        if not intra_dists or not inter_dists:
            return 0.0

        mean_intra = float(np.mean(intra_dists))
        mean_inter = float(np.mean(inter_dists))

        if mean_intra < 1e-8:
            return 999.0

        return mean_inter / mean_intra

    def _plot_tsne(
        self,
        coords: Optional[np.ndarray],
        labels: np.ndarray,
        modalities: np.ndarray,
        save_path: Path,
    ):
        """绘制 t-SNE 图（photo/SVG 双色调，类别着色）"""
        if coords is None:
            return

        unique_labels = np.unique(labels)
        n_classes = len(unique_labels)

        # Color map for classes
        cmap = cv2.COLORMAP_VIRIDIS if False else None
        fig_h, fig_w = 600, 900
        canvas = np.ones((fig_h, fig_w, 3), dtype=np.uint8) * 255

        # Normalize coords to canvas
        x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
        y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
        if x_max - x_min < 1e-8 or y_max - y_min < 1e-8:
            return

        pad = 40
        canvas_h, canvas_w = fig_h - 2 * pad, fig_w - 2 * pad

        def to_canvas(x, y):
            cx = int((x - x_min) / (x_max - x_min) * canvas_w) + pad
            cy = int((y - y_min) / (y_max - y_min) * canvas_h) + pad
            return cx, cy

        # Draw points
        for i in range(len(coords)):
            cx, cy = to_canvas(coords[i, 0], coords[i, 1])
            is_svg = modalities[i] == 'svg'
            color = (0, 120, 255) if is_svg else (255, 80, 80)  # BGR: blue=SVG, red=photo
            cv2.circle(canvas, (cx, cy), 3, color, -1)

        # Legend
        cv2.putText(canvas, "Red: Photo", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 80, 80), 2)
        cv2.putText(canvas, "Blue: SVG", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 120, 255), 2)
        cv2.putText(canvas, f"Classes: {n_classes}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        cv2.imwrite(str(save_path), canvas)

    def _save_history(self):
        """持久化训练历史"""
        with open(self.output_dir / "training_history.pkl", "wb") as f:
            pickle.dump(self.history, f)

    def print_epoch_summary(self, metrics: Dict, train_loss: float, val_loss: Optional[float] = None):
        """打印当前 epoch 汇总"""
        sep = "-" * 65
        print(sep)
        print(f"  Epoch {metrics['epoch']+1}  ── Train Loss: {train_loss:.4f}", end="")
        if val_loss is not None:
            print(f"  Val Loss: {val_loss:.4f}", end="")
        print()
        print(sep)

        print("  Layer1 (核心距离):")
        print(
            f"    d_ap_cross  : mean={metrics['d_ap_cross_mean']:.4f}  "
            f"std={metrics['d_ap_cross_std']:.4f}  "
            f"p50={metrics['d_ap_cross_p50']:.4f}  "
            f"p90={metrics['d_ap_cross_p90']:.4f}"
        )
        print(
            f"    d_ap_intra  : mean={metrics['d_ap_intra_mean']:.4f}  "
            f"std={metrics['d_ap_intra_std']:.4f}"
        )
        print(
            f"    cross/intra : {metrics['cross_intra_ratio']:.4f}  "
            f"(d_ap_cross / d_ap_intra, 趋近 1 更好)"
        )
        print(
            f"    d_an        : mean={metrics['d_an_mean']:.4f}  "
            f"std={metrics['d_an_std']:.4f}"
        )
        print(
            f"    margin_gap  : {metrics['margin_gap']:+.4f}  "
            f"(d_an - d_ap_cross, 应持续为正且增大)"
        )
        print(
            f"    active_tri  : {metrics['active_triplet_ratio']*100:.1f}%  "
            f"(健康区间 20%~70%)"
        )

        print("  Layer2 (训练状态):")
        print(
            f"    SE_weight   : mean={metrics['se_weight_mean']:.4f}  "
            f"std={metrics['se_weight_std']:.4f}  "
            f"range=[{metrics['se_weight_min']:.4f}, {metrics['se_weight_max']:.4f}]"
        )
        if 'real_val_macro_f1' in metrics:
            print("  RealVal (prototype nearest-neighbor):")
            print(
                f"    acc={metrics['real_val_acc']:.4f}  "
                f"macro_f1={metrics['real_val_macro_f1']:.4f}  "
                f"macro_p={metrics['real_val_macro_precision']:.4f}  "
                f"macro_r={metrics['real_val_macro_recall']:.4f}"
            )
            print(
                f"    known_dist: p50={metrics.get('real_val_known_p50', 0.0):.4f}  "
                f"p90={metrics.get('real_val_known_p90', 0.0):.4f}  "
                f"p95={metrics.get('real_val_known_p95', 0.0):.4f}  "
                f"samples={int(metrics.get('real_val_samples', 0))}"
            )
            print(
                f"    best_metric={metrics.get('best_metric_name')}  "
                f"current={metrics.get('best_score_current', 0.0):.4f}  "
                f"best={metrics.get('best_score', 0.0):.4f}"
            )
        print(sep)

    def plot_curves(self):
        """训练完成后绘制指标曲线"""
        if not self.history:
            return

        epochs   = [h['epoch'] + 1 for h in self.history]
        d_ap     = [h['d_ap_cross_mean'] for h in self.history]
        d_intra  = [h['d_ap_intra_mean'] for h in self.history]
        ratio    = [h['cross_intra_ratio'] for h in self.history]
        d_an     = [h['d_an_mean'] for h in self.history]
        gap      = [h['margin_gap'] for h in self.history]
        active   = [h['active_triplet_ratio'] for h in self.history]
        se_std   = [h['se_weight_std'] for h in self.history]
        loss_tr  = [h.get('train_loss', 0) for h in self.history]
        loss_val = [h.get('val_loss', None) for h in self.history]

        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle("Structure Enhancement Training Metrics", fontsize=14)

        def plot_epoch(ax, y, label, color, ylim=None):
            ax.plot(epochs, y, color=color, linewidth=1.5, marker='o', markersize=3)
            ax.set_xlabel("Epoch")
            ax.set_ylabel(label)
            ax.set_title(label)
            ax.grid(True, alpha=0.3)
            if ylim:
                ax.set_ylim(ylim)

        plot_epoch(axes[0, 0], d_ap, "d_ap_cross (同类距离)", "blue")
        plot_epoch(axes[0, 1], d_intra, "d_ap_intra (photo-photo)", "red")
        plot_epoch(axes[0, 2], ratio, "cross/intra ratio", "green")
        plot_epoch(axes[1, 0], active, "active_triplet_ratio (%)", "purple",
                   ylim=(0, 1))
        plot_epoch(axes[1, 1], se_std, "SE_weight_std", "orange")
        plot_epoch(axes[1, 2], gap, "margin_gap (d_an - d_ap)", "blue")

        plt.tight_layout()
        plt.savefig(self.output_dir / "training_metrics.png", dpi=150)
        plt.close()
        print(f"  指标曲线已保存: {self.output_dir / 'training_metrics.png'}")


# ─────────────────────────────────────────────────────────────────────────────
# 训练器
# ─────────────────────────────────────────────────────────────────────────────

class SVGStructureTrainer:
    """
    三路共享网络的Triplet训练器

    三路全部调用同一个 _to_embedding 函数：
        Anchor:    Photo → _to_embedding → emb_a
        Positive:  SVG   → _to_embedding → emb_p
        Negative:  Photo → _to_embedding → emb_n
        Loss:      TripletMarginLoss(emb_a, emb_p, emb_n)
    """

    def __init__(
        self,
        model: nn.Module,
        train_dataloader,
        val_dataloader,
        output_dir: Path,
        triplet_margin: float = 1.0,
        warmup_epochs: int = 2,
        eval_interval: int = 5,
        device: str = None,
        loss_mode: str = "svg_only",
        lambda_photo: float = 1.0,
        lambda_svg: float = 1.0,
        photo_triplet_margin: Optional[float] = None,
        svg_triplet_margin: Optional[float] = None,
        svg_align_type: str = "triplet",
        baseline_teacher: Optional[nn.Module] = None,
        beta_preserve: float = 0.0,
        preserve_type: str = "embedding",
        real_val_evaluator: Optional[SVGRealValEvaluator] = None,
        best_metric_name: str = "real_val_macro_f1",
        proxy_unknown_pool: Optional[ProxyUnknownPool] = None,
        gamma_unknown: float = 0.0,
        unknown_margin: float = 0.15,
        unknown_batch_size: Optional[int] = None,
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.output_dir = Path(output_dir)
        self.triplet_margin = triplet_margin
        self.loss_mode = loss_mode
        self.lambda_photo = float(lambda_photo)
        self.lambda_svg = float(lambda_svg)
        self.photo_triplet_margin = (
            float(photo_triplet_margin)
            if photo_triplet_margin is not None
            else float(triplet_margin)
        )
        self.svg_triplet_margin = (
            float(svg_triplet_margin)
            if svg_triplet_margin is not None
            else float(triplet_margin)
        )
        self.svg_align_type = svg_align_type
        self.baseline_teacher = baseline_teacher
        self.beta_preserve = float(beta_preserve)
        self.preserve_type = preserve_type
        self.real_val_evaluator = real_val_evaluator
        self.best_metric_name = best_metric_name
        self.proxy_unknown_pool = proxy_unknown_pool
        self.gamma_unknown = float(gamma_unknown)
        self.unknown_margin = float(unknown_margin)
        self.unknown_batch_size = unknown_batch_size
        self.warmup_epochs = warmup_epochs
        self.eval_interval = eval_interval
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._epoch_loss_details: Dict[str, float] = {}

        self.model.to(self.device)
        if self.baseline_teacher is not None:
            self.baseline_teacher.to(self.device)
            self.baseline_teacher.eval()
            for param in self.baseline_teacher.parameters():
                param.requires_grad = False
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.metrics = TrainingMetrics(output_dir, eval_interval=eval_interval)
        print(
            "[Trainer] loss_mode="
            f"{self.loss_mode}, lambda_photo={self.lambda_photo}, "
            f"lambda_svg={self.lambda_svg}, svg_align_type={self.svg_align_type}, "
            f"beta_preserve={self.beta_preserve}, preserve_type={self.preserve_type}, "
            f"gamma_unknown={self.gamma_unknown}, unknown_margin={self.unknown_margin}, "
            f"best_metric={self.best_metric_name}"
        )

    def train(self, epochs: int, optimizer, base_lr: float, resume: bool = False):
        """训练循环"""
        start_epoch = 0
        best_score = -float('inf')

        if resume:
            checkpoint_path = self.output_dir / "latest.pth"
            if checkpoint_path.exists():
                ckpt = torch.load(checkpoint_path, map_location=self.device,
                                  weights_only=False)
                start_epoch = ckpt.get('epoch', 0) + 1
                optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                print(f"[Trainer] 从epoch {start_epoch} 恢复训练")
                best_score = float(ckpt.get('best_score', best_score))

            hist_path = self.output_dir / "training_history.pkl"
            if hist_path.exists():
                import pickle
                with open(hist_path, "rb") as f:
                    self.metrics.history = pickle.load(f)
                print(f"[Trainer] 恢复历史记录 {len(self.metrics.history)} epochs")

        for epoch in range(start_epoch, epochs):
            if epoch < self.warmup_epochs:
                lr = base_lr * (epoch + 1) / self.warmup_epochs
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
                print(f"[Epoch {epoch+1}/{epochs}] Warmup LR: {lr:.6f}")

            # Semi-Hard Negative 更新
            if hasattr(self.train_dataloader.dataset, 'update_embeddings'):
                self.train_dataloader.dataset.update_embeddings(
                    self.model.embedding_net,
                    self.device,
                    self.train_dataloader.dataset.transform,
                )

            train_loss = self._train_epoch(epoch, optimizer)

            val_loss = None
            if self.val_dataloader is not None:
                val_loss = self._validate_epoch(epoch)

            # Layer1+Layer2+Layer3 指标收集
            metrics = self.metrics.epoch_end(
                epoch, self.model, self.train_dataloader, self.device
            )
            metrics['train_loss'] = train_loss
            metrics.update({
                'loss_mode': self.loss_mode,
                'lambda_photo': self.lambda_photo,
                'lambda_svg': self.lambda_svg,
                'beta_preserve': self.beta_preserve,
                'preserve_type': self.preserve_type,
                **self._epoch_loss_details,
            })
            if val_loss is not None:
                metrics['val_loss'] = val_loss

            real_val_metrics = {}
            if self.real_val_evaluator is not None:
                real_val_metrics = self.real_val_evaluator.evaluate(self.model.embedding_net)
                metrics.update(real_val_metrics)
                self.model.train()

            current_score = self._select_best_score(metrics, val_loss)
            metrics['best_metric_name'] = self.best_metric_name
            metrics['best_score_current'] = current_score
            if current_score > best_score:
                best_score = current_score
                metrics['best_score'] = best_score
                self._save_checkpoint(epoch, optimizer, 'best.pth', best_score=best_score)
                print(
                    f"[Trainer] best.pth 更新: {self.best_metric_name}={current_score:.4f}"
                )
            else:
                metrics['best_score'] = best_score
            self.metrics._save_history()

            self.metrics.print_epoch_summary(metrics, train_loss, val_loss)

            # 保存 checkpoint
            self._save_checkpoint(epoch, optimizer, 'latest.pth', best_score=best_score)
            if (epoch + 1) % 5 == 0:
                self._save_checkpoint(
                    epoch, optimizer, f'epoch_{epoch+1}.pth', best_score=best_score
                )

        # 训练完成后绘制曲线
        self.metrics.plot_curves()
        print("[Trainer] 训练完成")

    def _train_epoch(self, epoch: int, optimizer) -> float:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        total_photo_loss = 0.0
        total_svg_loss = 0.0
        total_preserve_loss = 0.0
        total_unknown_loss = 0.0
        total_unknown_min_dist = 0.0
        total_unknown_active = 0.0
        total_preserve_cos = 0.0
        total_preserve_l2 = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(self.train_dataloader):
            (
                anchor_photo,
                positive_photo,
                positive_svg,
                negative_photo,
                label_A,
                label_N,
            ) = self._unpack_batch(batch)

            anchor_photo   = anchor_photo.to(self.device)
            positive_photo = positive_photo.to(self.device)
            positive_svg   = positive_svg.to(self.device)
            negative_photo = negative_photo.to(self.device)

            optimizer.zero_grad()

            # 三路全部走同一个网络（Photo和SVG无模态差异）
            anchor_emb   = self.model.embedding_net(anchor_photo)
            photo_pos_emb = self.model.embedding_net(positive_photo)
            svg_pos_emb = self.model.embedding_net(positive_svg)
            negative_emb = self.model.embedding_net(negative_photo)

            (
                loss_total,
                loss_photo,
                loss_svg,
                loss_preserve,
                loss_unknown,
                metric_pos_emb,
                preserve_stats,
                unknown_stats,
            ) = self._compute_losses(
                anchor_emb,
                photo_pos_emb,
                svg_pos_emb,
                negative_emb,
                anchor_photo=anchor_photo,
                positive_photo=positive_photo,
                negative_photo=negative_photo,
            )
            per_sample_loss = F.triplet_margin_loss(
                anchor_emb, metric_pos_emb, negative_emb,
                margin=self._metric_margin(), p=2,
                reduction='none',
            )

            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()

            # Layer1 指标（每个 step）
            self.metrics.step(
                anchor_emb, svg_pos_emb, negative_emb, per_sample_loss, photo_pos_emb
            )

            total_loss += loss_total.item()
            total_photo_loss += loss_photo.item()
            total_svg_loss += loss_svg.item()
            total_preserve_loss += loss_preserve.item()
            total_unknown_loss += loss_unknown.item()
            total_unknown_min_dist += unknown_stats.get('unknown_min_dist', 0.0)
            total_unknown_active += unknown_stats.get('unknown_active_ratio', 0.0)
            total_preserve_cos += preserve_stats.get('preserve_cosine_sim', 0.0)
            total_preserve_l2 += preserve_stats.get('preserve_l2', 0.0)
            num_batches += 1

            if (batch_idx + 1) % 50 == 0:
                d_ap_photo = F.pairwise_distance(anchor_emb, photo_pos_emb).mean().item()
                d_ap_svg = F.pairwise_distance(anchor_emb, svg_pos_emb).mean().item()
                d_an = F.pairwise_distance(anchor_emb, negative_emb).mean().item()
                print(
                    f"  [Batch {batch_idx+1}/{len(self.train_dataloader)}] "
                    f"Total: {loss_total.item():.4f} | "
                    f"L_photo: {loss_photo.item():.4f} | "
                    f"L_svg: {loss_svg.item():.4f} | "
                    f"L_preserve: {loss_preserve.item():.4f} | "
                    f"L_unknown: {loss_unknown.item():.4f} | "
                    f"d_ap_photo: {d_ap_photo:.4f} | "
                    f"d_ap_svg: {d_ap_svg:.4f} | d_an: {d_an:.4f} | "
                    f"preserve_cos: {preserve_stats.get('preserve_cosine_sim', 0.0):.4f} | "
                    f"unk_min: {unknown_stats.get('unknown_min_dist', 0.0):.4f}"
                )

        if num_batches > 0:
            self._epoch_loss_details = {
                'train_loss_total': total_loss / num_batches,
                'train_loss_photo': total_photo_loss / num_batches,
                'train_loss_svg': total_svg_loss / num_batches,
                'train_loss_preserve': total_preserve_loss / num_batches,
                'train_loss_unknown': total_unknown_loss / num_batches,
                'unknown_min_dist': total_unknown_min_dist / num_batches,
                'unknown_active_ratio': total_unknown_active / num_batches,
                'preserve_cosine_sim': total_preserve_cos / num_batches,
                'preserve_l2': total_preserve_l2 / num_batches,
            }
            return total_loss / num_batches
        self._epoch_loss_details = {
            'train_loss_total': 0.0,
            'train_loss_photo': 0.0,
            'train_loss_svg': 0.0,
            'train_loss_preserve': 0.0,
            'train_loss_unknown': 0.0,
            'unknown_min_dist': 0.0,
            'unknown_active_ratio': 0.0,
            'preserve_cosine_sim': 0.0,
            'preserve_l2': 0.0,
        }
        return 0.0

    def _validate_epoch(self, epoch: int) -> float:
        """验证一个epoch"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in self.val_dataloader:
                (
                    anchor_photo,
                    positive_photo,
                    positive_svg,
                    negative_photo,
                    label_A,
                    label_N,
                ) = self._unpack_batch(batch)

                anchor_photo   = anchor_photo.to(self.device)
                positive_photo = positive_photo.to(self.device)
                positive_svg   = positive_svg.to(self.device)
                negative_photo = negative_photo.to(self.device)

                anchor_emb   = self.model.embedding_net(anchor_photo)
                photo_pos_emb = self.model.embedding_net(positive_photo)
                svg_pos_emb = self.model.embedding_net(positive_svg)
                negative_emb = self.model.embedding_net(negative_photo)

                loss_total, _, _, _, _, _, _, _ = self._compute_losses(
                    anchor_emb,
                    photo_pos_emb,
                    svg_pos_emb,
                    negative_emb,
                    anchor_photo=anchor_photo,
                    positive_photo=positive_photo,
                    negative_photo=negative_photo,
                )

                total_loss += loss_total.item()
                num_batches += 1

        return total_loss / num_batches if num_batches > 0 else 0.0

    def _unpack_batch(self, batch):
        """Support legacy 3-image batches and new hybrid 4-image batches."""
        if len(batch) == 6 and torch.is_tensor(batch[3]) and batch[3].ndim >= 4:
            anchor_photo, positive_photo, positive_svg, negative_photo, label_A, label_N = batch
            return anchor_photo, positive_photo, positive_svg, negative_photo, label_A, label_N

        anchor_photo, positive_svg, negative_photo, label_A, _label_P, label_N = batch
        return anchor_photo, anchor_photo, positive_svg, negative_photo, label_A, label_N

    def _compute_losses(
        self,
        anchor_emb,
        photo_pos_emb,
        svg_pos_emb,
        negative_emb,
        anchor_photo: Optional[torch.Tensor] = None,
        positive_photo: Optional[torch.Tensor] = None,
        negative_photo: Optional[torch.Tensor] = None,
    ):
        zero = anchor_emb.new_tensor(0.0)
        loss_photo = zero
        loss_svg = zero
        loss_preserve = zero
        loss_unknown = zero
        preserve_stats: Dict[str, float] = {
            'preserve_cosine_sim': 0.0,
            'preserve_l2': 0.0,
        }
        unknown_stats: Dict[str, float] = {
            'unknown_min_dist': 0.0,
            'unknown_active_ratio': 0.0,
        }

        photo_loss_modes = {
            "photo_only",
            "hybrid_triplet",
            "hybrid_cosine",
            "baseline_preserving_hybrid",
        }
        svg_triplet_modes = {
            "svg_only",
            "hybrid_triplet",
            "baseline_preserving_hybrid",
        }

        if self.loss_mode in photo_loss_modes:
            loss_photo = F.triplet_margin_loss(
                anchor_emb, photo_pos_emb, negative_emb,
                margin=self.photo_triplet_margin, p=2,
                reduction='mean',
            )

        if self.loss_mode in svg_triplet_modes and self.svg_align_type == "triplet":
            loss_svg = F.triplet_margin_loss(
                anchor_emb, svg_pos_emb, negative_emb,
                margin=self.svg_triplet_margin, p=2,
                reduction='mean',
            )
        elif self.loss_mode == "hybrid_cosine" or (
            self.loss_mode == "baseline_preserving_hybrid"
            and self.svg_align_type == "cosine"
        ):
            loss_svg = 1.0 - F.cosine_similarity(anchor_emb, svg_pos_emb, dim=1).mean()

        if self.loss_mode == "svg_only":
            loss_total = loss_svg
            metric_pos_emb = svg_pos_emb
        elif self.loss_mode == "photo_only":
            loss_total = self.lambda_photo * loss_photo
            metric_pos_emb = photo_pos_emb
        elif self.loss_mode in {"hybrid_triplet", "hybrid_cosine"}:
            loss_total = self.lambda_photo * loss_photo + self.lambda_svg * loss_svg
            metric_pos_emb = photo_pos_emb
        elif self.loss_mode == "baseline_preserving_hybrid":
            loss_preserve, preserve_stats = self._compute_preserve_loss(
                anchor_emb,
                photo_pos_emb,
                negative_emb,
                anchor_photo=anchor_photo,
                positive_photo=positive_photo,
                negative_photo=negative_photo,
            )
            loss_unknown, unknown_stats = self._compute_unknown_boundary_loss(
                anchor_emb,
                photo_pos_emb,
                svg_pos_emb,
                negative_emb,
            )
            loss_total = (
                self.lambda_photo * loss_photo
                + self.lambda_svg * loss_svg
                + self.beta_preserve * loss_preserve
                + self.gamma_unknown * loss_unknown
            )
            metric_pos_emb = photo_pos_emb
        else:
            raise ValueError(f"Unsupported loss_mode: {self.loss_mode}")

        return (
            loss_total,
            loss_photo,
            loss_svg,
            loss_preserve,
            loss_unknown,
            metric_pos_emb,
            preserve_stats,
            unknown_stats,
        )

    def _compute_unknown_boundary_loss(
        self,
        anchor_emb: torch.Tensor,
        photo_pos_emb: torch.Tensor,
        svg_pos_emb: torch.Tensor,
        negative_emb: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        zero = anchor_emb.new_tensor(0.0)
        if (
            self.gamma_unknown <= 0
            or self.proxy_unknown_pool is None
            or len(self.proxy_unknown_pool) == 0
        ):
            return zero, {'unknown_min_dist': 0.0, 'unknown_active_ratio': 0.0}

        batch_size = (
            int(self.unknown_batch_size)
            if self.unknown_batch_size is not None
            else int(anchor_emb.shape[0])
        )
        unknown_batch = self.proxy_unknown_pool.sample_batch(batch_size)
        if unknown_batch is None:
            return zero, {'unknown_min_dist': 0.0, 'unknown_active_ratio': 0.0}

        unknown_batch = unknown_batch.to(self.device)
        unknown_emb = self.model.embedding_net(unknown_batch)
        unknown_emb = F.normalize(unknown_emb, p=2, dim=1)

        known_refs = torch.cat(
            [
                F.normalize(anchor_emb.detach(), p=2, dim=1),
                F.normalize(photo_pos_emb.detach(), p=2, dim=1),
                F.normalize(svg_pos_emb.detach(), p=2, dim=1),
                F.normalize(negative_emb.detach(), p=2, dim=1),
            ],
            dim=0,
        )
        min_dists = torch.cdist(unknown_emb, known_refs, p=2).min(dim=1)[0]
        per_sample = F.relu(self.unknown_margin - min_dists)
        loss_unknown = per_sample.mean()

        with torch.no_grad():
            active_ratio = (per_sample > 0).float().mean().item()
            min_dist = min_dists.mean().item()

        return loss_unknown, {
            'unknown_min_dist': min_dist,
            'unknown_active_ratio': active_ratio,
        }

    def _compute_preserve_loss(
        self,
        anchor_emb: torch.Tensor,
        photo_pos_emb: torch.Tensor,
        negative_emb: torch.Tensor,
        anchor_photo: Optional[torch.Tensor],
        positive_photo: Optional[torch.Tensor],
        negative_photo: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        zero = anchor_emb.new_tensor(0.0)
        if self.baseline_teacher is None or self.beta_preserve <= 0:
            return zero, {'preserve_cosine_sim': 0.0, 'preserve_l2': 0.0}
        if anchor_photo is None or positive_photo is None:
            raise ValueError("baseline_preserving_hybrid requires photo tensors")

        new_embeddings = [
            F.normalize(anchor_emb, p=2, dim=1),
            F.normalize(photo_pos_emb, p=2, dim=1),
        ]
        photo_tensors = [anchor_photo, positive_photo]

        with torch.no_grad():
            base_embeddings = [
                F.normalize(self.baseline_teacher.embedding_net(photo), p=2, dim=1)
                for photo in photo_tensors
            ]

        if self.preserve_type == "embedding":
            losses = [
                F.mse_loss(new_emb, base_emb)
                for new_emb, base_emb in zip(new_embeddings, base_embeddings)
            ]
            loss_preserve = torch.stack(losses).mean()
        elif self.preserve_type == "cosine":
            losses = [
                1.0 - F.cosine_similarity(new_emb, base_emb, dim=1).mean()
                for new_emb, base_emb in zip(new_embeddings, base_embeddings)
            ]
            loss_preserve = torch.stack(losses).mean()
        elif self.preserve_type == "pairwise":
            new_all = torch.cat(new_embeddings, dim=0)
            base_all = torch.cat(base_embeddings, dim=0)
            loss_preserve = F.smooth_l1_loss(
                torch.cdist(new_all, new_all, p=2),
                torch.cdist(base_all, base_all, p=2),
            )
        else:
            raise ValueError(f"Unsupported preserve_type: {self.preserve_type}")

        new_all = torch.cat(new_embeddings, dim=0)
        base_all = torch.cat(base_embeddings, dim=0)
        with torch.no_grad():
            preserve_cos = F.cosine_similarity(new_all, base_all, dim=1).mean().item()
            preserve_l2 = F.pairwise_distance(new_all, base_all).mean().item()

        return loss_preserve, {
            'preserve_cosine_sim': preserve_cos,
            'preserve_l2': preserve_l2,
        }

    def _metric_margin(self) -> float:
        if self.loss_mode in {
            "photo_only",
            "hybrid_triplet",
            "hybrid_cosine",
            "baseline_preserving_hybrid",
        }:
            return self.photo_triplet_margin
        return self.svg_triplet_margin

    def _select_best_score(self, metrics: Dict, val_loss: Optional[float]) -> float:
        if self.best_metric_name == "negative_val_loss":
            return -float(val_loss) if val_loss is not None else -float('inf')
        if self.best_metric_name not in metrics:
            if val_loss is not None:
                return -float(val_loss)
            return -float('inf')
        return float(metrics[self.best_metric_name])

    def _save_checkpoint(
        self,
        epoch: int,
        optimizer,
        filename: str,
        best_score: Optional[float] = None,
    ):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_metric_name': self.best_metric_name,
        }
        if best_score is not None:
            checkpoint['best_score'] = best_score
        torch.save(checkpoint, self.output_dir / filename)


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────

class Stage2SVGStructurePipeline(BasePipeline):
    """Structure Enhancement训练Pipeline"""

    def __init__(self, config_path: str = "configs/config.yaml"):
        super().__init__(config_path)
        self.dataset_builder = None

    def validate(self) -> bool:
        paths = self.get_data_paths()
        if not paths['stage1_class_agnostic_weights_dir'].exists():
            raise FileNotFoundError("Stage 1 model directory not found")
        if not paths['raw_images_dir'].exists() or \
           not paths['raw_class_aware_labels_dir'].exists():
            raise FileNotFoundError("Raw data directories not found")
        return True

    def prepare_symbol_crops(self) -> None:
        paths = self.get_data_paths()
        BBoxUtils.extract_bbox_crops(
            image_dir=paths['raw_images_dir'],
            label_dir=paths['raw_class_aware_labels_dir'],
            output_dir=paths['symbol_crops_dir']
        )

    def prepare_train_data(self) -> None:
        config = self.get_stage2_baseline_config()
        paths  = self.get_data_paths()

        train_val_split = tuple(
            float(x) for x in config.get("train_val_split", [0.8, 0.2])
        )

        self.dataset_builder = ClassCropsBuilder(
            crops_root_dir=paths['symbol_crops_dir']
        )
        self.dataset_builder.split_train_val(
            train_dir=paths['train_dir'],
            val_dir=paths['val_dir'],
            train_val_split=train_val_split,
            seed=42,
        )

    def train_model(self, resume: bool = False) -> None:
        """训练Structure Enhancement模型"""
        paths = self.get_data_paths()
        model_paths = self.get_model_paths()
        config = self.get_stage2_svg_config()

        # ── 预训练权重路径 ─────────────────────────────────────────────────
        # 优先使用本地预训练权重，避免联网下载
        local_pretrained = (
            Path("/media/wit/HDD_16T/wxr/experiment/swin_tiny_patch4_window7_224/pytorch_model.bin")
        )
        pretrained_path = str(local_pretrained) if local_pretrained.exists() else ""
        if pretrained_path:
            print(f"[Stage2-SVG] 使用本地预训练权重: {pretrained_path}")
        else:
            print("[Stage2-SVG] 警告: 本地预训练权重不存在，将使用timm在线下载")

        # ── 模型 ──────────────────────────────────────────────────────────
        embedding_net = SVGStructureEnhanceNet(
            embed_dim=config['embed_dim'],
            dropout=config.get('dropout', 0.1),
            pretrained_path=pretrained_path,
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        embedding_net.to(device)

        for p in embedding_net.backbone.parameters():
            p.requires_grad = True

        model = TripletNet(embedding_net)
        model.to(device)

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters()
                               if p.requires_grad)
        print(f"[Stage2-SVG] 参数统计: trainable={trainable_params:,}/{total_params:,}")

        # ── 超参 ─────────────────────────────────────────────────────────
        triplet_margin = config.get('triplet_margin', 1.0)
        loss_mode = config.get('loss_mode', 'svg_only')
        lambda_photo = float(config.get('lambda_photo', 1.0))
        lambda_svg = float(config.get('lambda_svg', 1.0))
        beta_preserve = float(config.get('beta_preserve', 0.0))
        preserve_type = config.get('preserve_type', 'embedding')
        baseline_teacher_checkpoint = config.get('baseline_teacher_checkpoint')
        init_from_baseline = bool(config.get('init_from_baseline', False))
        gamma_unknown = float(config.get('gamma_unknown', 0.0))
        unknown_margin = float(config.get('unknown_margin', 0.15))
        unknown_dir = config.get('unknown_dir')
        unknown_max_samples = int(config.get('unknown_max_samples', 300))
        unknown_batch_size = config.get('unknown_batch_size')
        if unknown_batch_size is not None:
            unknown_batch_size = int(unknown_batch_size)
        photo_triplet_margin = float(config.get('photo_triplet_margin', triplet_margin))
        svg_triplet_margin = float(config.get('svg_triplet_margin', triplet_margin))
        svg_align_type = config.get('svg_align_type', 'triplet')
        learning_rate  = float(config.get('learning_rate', 3e-5))
        weight_decay   = float(config.get('weight_decay', 0.05))
        batch_size     = int(config.get('batch_size', 32))
        epochs         = int(config.get('epochs', 25))
        warmup_epochs  = int(config.get('warmup_epochs', 2))
        eval_interval  = int(config.get('eval_interval', 5))
        real_val_enabled = bool(config.get('real_val_enabled', True))
        real_val_metric = str(config.get('real_val_metric', 'real_val_macro_f1'))
        real_val_max_train_per_class = config.get('real_val_max_train_per_class', 200)
        real_val_max_val_per_class = config.get('real_val_max_val_per_class', None)
        real_val_batch_size = int(config.get('real_val_batch_size', 64))
        if real_val_max_train_per_class is not None:
            real_val_max_train_per_class = int(real_val_max_train_per_class)
        if real_val_max_val_per_class is not None:
            real_val_max_val_per_class = int(real_val_max_val_per_class)

        baseline_teacher = None
        if loss_mode == 'baseline_preserving_hybrid':
            if not baseline_teacher_checkpoint:
                raise ValueError(
                    "baseline_preserving_hybrid requires "
                    "stage2_svg_structure.baseline_teacher_checkpoint"
                )
            baseline_teacher = load_baseline_teacher(
                baseline_teacher_checkpoint,
                embedding_size=int(config['embed_dim']),
                device=device,
            )
            if init_from_baseline:
                load_student_backbone_from_baseline(
                    model,
                    baseline_teacher_checkpoint,
                    device=device,
                )

        # ── 优化器 ───────────────────────────────────────────────────────
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        # ── Transform ─────────────────────────────────────────────────────
        base_transform = v2.Compose([
            v2.Lambda(pad_to_square),
            v2.Resize(size=(224, 224), antialias=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # ── 数据路径 ──────────────────────────────────────────────────────
        svg_dir       = config.get('svg_dir', 'data/processed/stage_2/svg_to_png')
        svg_train_dir = config.get('svg_train_dir', svg_dir)
        svg_val_dir   = config.get('svg_val_dir', svg_dir)

        max_per_class        = int(config.get('max_samples_per_class', 400))
        semi_hard_candidates = int(config.get('semi_hard_candidates', 10))
        semi_hard_margin     = float(config.get('semi_hard_margin', 1.0))

        # ── 打印配置 ────────────────────────────────────────────────────
        print("=" * 70)
        print("【Structure Enhancement 训练配置】")
        print(f"  模型: SVGStructureEnhanceNet (三路共享网络)")
        print(f"  embed_dim: {config['embed_dim']}")
        print(f"  batch_size: {batch_size}")
        print(f"  learning_rate: {learning_rate}")
        print(f"  epochs: {epochs}")
        print(f"  warmup_epochs: {warmup_epochs}")
        print(f"  eval_interval (Layer3): {eval_interval}")
        print(f"  triplet_margin: {triplet_margin}")
        print(f"  loss_mode: {loss_mode}")
        print(f"  lambda_photo: {lambda_photo}")
        print(f"  lambda_svg: {lambda_svg}")
        print(f"  beta_preserve: {beta_preserve}")
        print(f"  preserve_type: {preserve_type}")
        print(f"  baseline_teacher_checkpoint: {baseline_teacher_checkpoint}")
        print(f"  init_from_baseline: {init_from_baseline}")
        print(f"  gamma_unknown: {gamma_unknown}")
        print(f"  unknown_margin: {unknown_margin}")
        print(f"  unknown_dir: {unknown_dir}")
        print(f"  unknown_max_samples: {unknown_max_samples}")
        print(f"  unknown_batch_size: {unknown_batch_size}")
        print(f"  real_val_enabled: {real_val_enabled}")
        print(f"  real_val_metric: {real_val_metric}")
        print(f"  real_val_max_train_per_class: {real_val_max_train_per_class}")
        print(f"  real_val_max_val_per_class: {real_val_max_val_per_class}")
        print(f"  real_val_batch_size: {real_val_batch_size}")
        print(f"  photo_triplet_margin: {photo_triplet_margin}")
        print(f"  svg_triplet_margin: {svg_triplet_margin}")
        print(f"  svg_align_type: {svg_align_type}")
        print(f"  photo_train_dir: {paths['train_dir']}")
        print(f"  svg_train_dir: {svg_train_dir}")
        print(f"  photo_val_dir: {paths['val_dir']}")
        print("=" * 70)

        # ── Dataset ───────────────────────────────────────────────────────
        dataset_cls = (
            HybridPhotoSVGTripletDataset
            if loss_mode in {
                'photo_only',
                'hybrid_triplet',
                'hybrid_cosine',
                'baseline_preserving_hybrid',
            }
            else PhotoSVGTripletDataset
        )
        train_dataset = dataset_cls(
            photo_root_dir=paths['train_dir'],
            svg_root_dir=svg_train_dir,
            transform=base_transform,
            max_per_class=max_per_class,
            semi_hard_candidates=semi_hard_candidates,
            semi_hard_margin=semi_hard_margin,
        )

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
        )

        val_dataloader = None
        if paths['val_dir'].exists():
            val_dataset = dataset_cls(
                photo_root_dir=paths['val_dir'],
                svg_root_dir=svg_val_dir,
                transform=base_transform,
                max_per_class=max_per_class,
                semi_hard_candidates=semi_hard_candidates,
                semi_hard_margin=semi_hard_margin,
            )
            val_dataloader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=False,
            )
        else:
            print("[Stage2-SVG] val_dir 不存在，val_dataloader 设为 None")

        # ── 训练器 ────────────────────────────────────────────────────
        real_val_evaluator = None
        if real_val_enabled and paths['val_dir'].exists():
            real_val_evaluator = SVGRealValEvaluator(
                train_dir=paths['train_dir'],
                val_dir=paths['val_dir'],
                transform=base_transform,
                device=device,
                max_train_per_class=real_val_max_train_per_class,
                max_val_per_class=real_val_max_val_per_class,
                batch_size=real_val_batch_size,
            )

        proxy_unknown_pool = None
        if gamma_unknown > 0:
            if not unknown_dir:
                print("[ProxyUnknown] gamma_unknown > 0 但未配置 unknown_dir，禁用 unknown loss")
            else:
                proxy_unknown_pool = ProxyUnknownPool(
                    unknown_dir=unknown_dir,
                    transform=base_transform,
                    max_samples=unknown_max_samples,
                )
                if len(proxy_unknown_pool) == 0:
                    print("[ProxyUnknown] 未加载到 unknown 样本，禁用 unknown loss")
                    proxy_unknown_pool = None

        output_suffix = config.get('output_suffix')
        if output_suffix:
            output_dir = model_paths['stage2_weights_dir'] / str(output_suffix)
        else:
            output_dir = model_paths.get(
                'stage2_svg_weights_dir',
                model_paths['stage2_weights_dir'] / 'svg_structure_v2'
            )
        output_dir.mkdir(parents=True, exist_ok=True)

        trainer = SVGStructureTrainer(
            model,
            train_dataloader,
            val_dataloader,
            output_dir,
            triplet_margin=triplet_margin,
            warmup_epochs=warmup_epochs,
            eval_interval=eval_interval,
            loss_mode=loss_mode,
            lambda_photo=lambda_photo,
            lambda_svg=lambda_svg,
            photo_triplet_margin=photo_triplet_margin,
            svg_triplet_margin=svg_triplet_margin,
            svg_align_type=svg_align_type,
            baseline_teacher=baseline_teacher,
            beta_preserve=beta_preserve,
            preserve_type=preserve_type,
            real_val_evaluator=real_val_evaluator,
            best_metric_name=real_val_metric if real_val_evaluator is not None else "negative_val_loss",
            proxy_unknown_pool=proxy_unknown_pool,
            gamma_unknown=gamma_unknown if proxy_unknown_pool is not None else 0.0,
            unknown_margin=unknown_margin,
            unknown_batch_size=unknown_batch_size,
        )

        print(
            f"[SVG-Structure] 开始训练: {epochs} epochs, "
            f"batch_size={batch_size}, max_per_class={max_per_class}"
        )
        trainer.train(epochs, optimizer, base_lr=learning_rate, resume=resume)

    def run(self) -> None:
        if not self.validate():
            raise ValueError("Pipeline validation failed")
        print("Extracting symbol crops...")
        self.prepare_symbol_crops()
        print("Preparing training data...")
        self.prepare_train_data()
        print("Training Structure Enhancement model...")
        self.train_model()
        print("Stage 2 Structure Enhancement pipeline completed successfully")
