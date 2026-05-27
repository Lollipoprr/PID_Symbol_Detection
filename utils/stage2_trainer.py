# 训练器：triplet loss
from pathlib import Path
from .stage2_Siamese_Network import TripletNet, EmbeddingNet
from .stage2_triplets_generator import EpisodicTripletDatasetFromDir
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LinearLR
from torchvision.transforms import v2
import pickle
import logging
import random
import numpy as np
from typing import List, Tuple, Dict, Optional, Callable
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    encoding='utf-8'
)


# ─────────────────────────────────────────────────────────────────────────────
# pad工具函数
# ─────────────────────────────────────────────────────────────────────────────
def pad_to_square_pil(img: Image.Image, fill: int = 255) -> Image.Image:
    """PIL图像pad到正方形（白色填充，居中）"""
    w, h = img.size
    if w == h:
        return img
    s = max(w, h)
    new_img = Image.new("RGB", (s, s), (fill, fill, fill))
    new_img.paste(img, ((s - w) // 2, (s - h) // 2))
    return new_img


def pad_to_square_tensor(img: torch.Tensor) -> torch.Tensor:
    """[C,H,W] float tensor pad到正方形，白色填充，居中"""
    _, h, w = img.shape
    if h == w:
        return img
    s      = max(h, w)
    padded = torch.ones(img.shape[0], s, s, dtype=img.dtype, device=img.device)
    top    = (s - h) // 2
    left   = (s - w) // 2
    padded[:, top:top+h, left:left+w] = img
    return padded


# 推理专用预处理（无数据增强，与原型构建一致）
_EVAL_TRANSFORM = v2.Compose([
    v2.Lambda(pad_to_square_tensor),
    v2.Resize(size=(224, 224), antialias=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ─────────────────────────────────────────────────────────────────────────────
# Representative-prototype utilities
# ─────────────────────────────────────────────────────────────────────────────
def adaptive_temperature(n_samples: int, base_temp: float = 0.1) -> float:
    """
    根据类样本量自适应温度（专为长尾设计）：

      数据量越大 → 温度越低（精确，置信加权效果明显）
      数据量越小 → 温度越高（保守，接近均匀平均，保护tail类不被极端权重压垮）

    阈值参考（基于长尾分布 53~6802 的实际情况）：
      < 200 样本 → ×3.0  保护极稀类
      < 1000样本 → ×1.5  平衡
      >= 1000样本 → ×1.0  默认精确
    """
    if n_samples < 200:
        return base_temp * 3.0
    elif n_samples < 1000:
        return base_temp * 1.5
    else:
        return base_temp


def compute_weighted_prototypes(
    embedding_net: nn.Module,
    train_dir: Path,
    device: str,
    base_temperature: float = 0.1,
        use_adaptive_temp: bool = True,
) -> Dict[int, torch.Tensor]:
    """
    长尾对抗加权原型构建（双层权重，专为高度不平衡数据集设计）：

    Step 1 ─ 类间均衡：样本量逆频率权重 1/sqrt(N) 抵消大类的embedding空间主导效应
    Step 2 ─ 类内去噪：余弦相似度 softmax 对离群点施以低权重（但温度固定为0.3，
                不再对head类用低温度，避免少数高置信样本主导整个类）
    Step 3 ─ 两层权重相乘后归一化，得到抗长尾的鲁棒原型

    对 head 类（数千样本）：
      - 逆频率权重降至 ~0.1，有效对抗其对空间的压迫
      - 高温度保证类内所有变体都参与原型构建
    对 tail 类（数十样本）：
      - 逆频率权重接近1.0，充分利用每个样本
      - 高温度防止极少量样本的原型被极端权重压垮

    Args:
        embedding_net:      已加载的embedding模型（.eval()后使用）
        train_dir:      每类一个子目录，子目录名=class_id
        device:            cuda/cpu
        base_temperature:  基准温度（现固定为0.3，不再区分大小类）
        use_adaptive_temp: 保留参数，向后兼容（实际不再使用adaptive温度）

    Returns:
        Dict[class_id -> prototype_tensor [1, D]]
    """
    embedding_net.eval()
    transform = _EVAL_TRANSFORM

    class_embeddings: Dict[int, List[torch.Tensor]] = {}

    logging.info("[原型计算] 开始计算每类原型...")

    for cls_folder in sorted(train_dir.iterdir()):
        if not cls_folder.is_dir():
            continue
        try:
            cls_id = int(cls_folder.name)
        except ValueError:
            continue

        imgs = (list(cls_folder.glob("*.png")) +
                list(cls_folder.glob("*.jpg")) +
                list(cls_folder.glob("*.jpeg")))
        if not imgs:
            continue

        embs = []
        for img_path in imgs:
            img = Image.open(img_path).convert("RGB")
            img = pad_to_square_pil(img)
            t = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
            t = transform(t).unsqueeze(0).to(device)
            with torch.no_grad():
                emb = embedding_net(t)
            embs.append(emb.squeeze(0).cpu())

        if not embs:
            continue
        class_embeddings[cls_id] = embs

    if not class_embeddings:
        logging.warning("[原型计算] 未找到任何类别，跳过")
        return {}

    logging.info(f"[原型计算] 完成，共{len(class_embeddings)}类")

    prototypes = {}
    for cls_id, embs in class_embeddings.items():
        n_samples = len(embs)

        stacked = torch.stack(embs)                          # [N, D]
        class_mean = stacked.mean(dim=0, keepdim=True)  # [1, D]
        class_mean = F.normalize(class_mean, p=2, dim=1)

        # ── Step 1: 类内均匀权重 ────────────────────────────────────
        # 注：类间均衡在 triplet 采样层面处理（tail 类过采样），
        # 单类原型内部无法实现类间均衡（归一化后效果被抵消）
        prior_weights = torch.ones(n_samples) / n_samples  # 均匀权重，直接 1/N

        # ── Step 2: 类内去噪（次权重：自适应温度的cosine相似度）────
        # 【修复】使用自适应温度替代固定T=0.3：
        #   tail类(N<200): T=0.3，保护少数样本不被极端权重压垮
        #   mid类(N<1000): T=0.2，平衡精确度和稳健性
        #   head类(N>=1000): T=0.1，高精确度区分高置信样本
        # 原固定T=0.3过于aggressive，softmax后少数高置信样本主导，
        # 导致tail类原型质量严重下降
        temperature = adaptive_temperature(n_samples, base_temperature)
        similarities = (stacked @ class_mean.T).squeeze(1)  # [N]
        sim_weights = torch.softmax(similarities / temperature, dim=0)  # [N]

        # ── Step 3: 双层权重相乘（prior权重主导，sim权重辅助）─────
        # prior 权重范围大（tail类~1.0, head类~0.1），决定相对贡献
        # sim 权重决定类内哪些样本更重要
        combined_weights = prior_weights * sim_weights          # [N]
        combined_weights = combined_weights / combined_weights.sum()

        weighted_mean = (stacked * combined_weights.unsqueeze(1)).sum(dim=0, keepdim=True)  # [1, D]
        weighted_mean = F.normalize(weighted_mean, p=2, dim=1)

        prototypes[cls_id] = weighted_mean.to(device)

    logging.info(
        f"[原型计算] 双层加权原型完成：{len(prototypes)}类，"
        f"prior=1/sqrt(N)，sim_temp=0.3"
    )
    return prototypes


# ─────────────────────────────────────────────────────────────────────────────
# RealValEvaluator
# ─────────────────────────────────────────────────────────────────────────────
class RealValEvaluator:
    """
    真实验证集评估器（基于分割后的 val_dir）。

    流程：
      1. 从 train_dir 的图计算每类原型
      2. 遍历 val_dir 的裁剪图（每类一个文件夹）
      3. 对每张图计算 embedding，找最近邻原型，得到预测 class id
      4. 与文件名所属的真实 class id 比对，计算 macro precision

    Args:
        train_dir:    原型来源目录（每类N张图）
        val_dir:      验证目录（每类一个子目录）
        device:       cuda/cpu
        max_per_class: 每类最多评估多少张图，None=全部
    """

    def __init__(
        self,
        train_dir:           Path,
        val_dir:             Path,
        device:              str,
        max_per_class:       Optional[int] = None,
        prototype_temperature: float        = 0.1,
        use_adaptive_temp:   bool          = True,
    ):
        self.train_dir   = Path(train_dir)
        self.val_dir     = Path(val_dir)
        self.device      = device
        self.max_per_class = max_per_class
        self.prototype_temperature = prototype_temperature
        self.use_adaptive_temp     = use_adaptive_temp

        self.val_class_dirs = sorted([
            d for d in self.val_dir.iterdir() if d.is_dir()
        ])

        logging.info(
            f"[RealValEvaluator] val_dir={self.val_dir}, "
            f"train_dir={self.train_dir}, "
            f"classes={len(self.val_class_dirs)}"
        )

    def _compute_prototypes(
        self, embedding_net: nn.Module
    ) -> Dict[int, torch.Tensor]:
        """从 train_dir 计算每类双层加权原型"""
        return compute_weighted_prototypes(
            embedding_net,
            self.train_dir,
            self.device,
            base_temperature=0.1,
            use_adaptive_temp=True,
        )

    def evaluate(self, embedding_net: nn.Module) -> float:
        """
        计算当前模型在 val 集上的 macro precision

        Returns:
            precision (float): 0~1
        """
        embedding_net.eval()

        prototypes = self._compute_prototypes(embedding_net)
        if not prototypes:
            logging.warning("[RealValEvaluator] 未找到任何原型，跳过评估")
            return 0.0

        tp_per_class:   Dict[int, int] = {}
        pred_per_class: Dict[int, int] = {}
        gt_per_class:   Dict[int, int] = {}

        for class_dir in self.val_class_dirs:
            try:
                gt_cls = int(class_dir.name)
            except ValueError:
                continue

            image_paths = sorted(
                list(class_dir.glob("*.png")) +
                list(class_dir.glob("*.jpg")) +
                list(class_dir.glob("*.jpeg"))
            )

            if self.max_per_class and len(image_paths) > self.max_per_class:
                image_paths = random.sample(image_paths, self.max_per_class)

            for img_path in image_paths:
                image = Image.open(img_path).convert("RGB")
                image = pad_to_square_pil(image)  # 与原型构建一致
                img_tensor = v2.Compose([
                    v2.Resize((224, 224), antialias=True),
                    v2.ToTensor(),
                    v2.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]),
                ])(image).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    emb = embedding_net(img_tensor)
                emb = F.normalize(emb, p=2, dim=1)

                min_dist = float('inf')
                pred_cls = -1
                for cls_id, proto in prototypes.items():
                    dist = torch.cdist(emb, proto.view(1, -1)).item()
                    if dist < min_dist:
                        min_dist = dist
                        pred_cls = cls_id

                gt_per_class[gt_cls] = gt_per_class.get(gt_cls, 0) + 1
                pred_per_class[pred_cls] = pred_per_class.get(pred_cls, 0) + 1
                if pred_cls == gt_cls:
                    tp_per_class[gt_cls] = tp_per_class.get(gt_cls, 0) + 1

        if not gt_per_class:
            logging.warning("[RealValEvaluator] val 集中没有任何有效样本")
            return 0.0

        precisions = []
        for cls_id in gt_per_class:
            tp   = tp_per_class.get(cls_id, 0)
            pred = pred_per_class.get(cls_id, 0)
            precisions.append(tp / pred if pred > 0 else 0.0)

        macro_precision = sum(precisions) / len(precisions)
        return macro_precision


# ─────────────────────────────────────────────────────────────────────────────
# FewShotTrainer
# ─────────────────────────────────────────────────────────────────────────────
class FewShotTrainer:
    """
    联合训练：
        total_loss = loss_triplet

    Args:
        model:               TripletNet
        train_dataloader:    triplet DataLoader
        val_dataloader:      triplet DataLoader（记录loss用，不再决定best model）
        output_dir:          模型保存目录
        warmup_epochs:       线性warmup轮数
        real_val_evaluator:  RealValEvaluator实例，None=禁用真实val（fallback到triplet acc）
    """

    def __init__(
        self,
        model:               nn.Module,
        train_dataloader:    DataLoader,
        val_dataloader:      DataLoader,
        output_dir:          Path,
        warmup_epochs:       int                        = 0,
        real_val_evaluator:  Optional[RealValEvaluator] = None,
    ):
        self.model             = model
        self.train_dataloader  = train_dataloader
        self.val_dataloader    = val_dataloader
        self.output_dir        = output_dir
        self.warmup_epochs     = warmup_epochs
        self.real_val_evaluator = real_val_evaluator
        self.device            = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if real_val_evaluator is not None:
            logging.info("[FewShotTrainer] 真实val评估已启用，按真实precision保存best model")
        else:
            logging.info("[FewShotTrainer] 真实val评估未启用，按triplet val acc保存best model")

    def _update_train_embeddings(self) -> None:
        """用当前模型更新 train dataset 的 embedding 缓存"""
        if hasattr(self.train_dataloader.dataset, 'update_embeddings'):
            self.train_dataloader.dataset.update_embeddings(
                self._get_base_embedding_net(),
                self.device,
                _EVAL_TRANSFORM,  # 必须用固定 transform，不能用训练 transform
            )

    def _update_val_embeddings(self) -> None:
        """用当前模型更新 val dataset 的 embedding 缓存"""
        if self.val_dataloader is not None and hasattr(self.val_dataloader.dataset, 'update_embeddings'):
            self.val_dataloader.dataset.update_embeddings(
                self._get_base_embedding_net(),
                self.device,
                _EVAL_TRANSFORM,  # 必须用固定 transform
            )

    def _get_base_embedding_net(self) -> nn.Module:
        """获取基础 EmbeddingNet（兼容不同模型结构）"""
        if hasattr(self.model, 'embedding_net'):
            return self.model.embedding_net
        return self.model

    def train(
        self,
        epochs:    int,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        base_lr:   float = 3e-5,
        resume:    bool  = False,
        resume_ckpt: str = "checkpoint_latest.pth",
    ) -> None:
        start_epoch = 0
        history = {
            "train_losses":        [],
            "train_accuracies":    [],
            "val_losses":          [],
            "val_accuracies":      [],
            "train_triplet_losses":[],
            "real_val_precisions": [],
            "train_dist_AP":       [],
            "train_dist_AN":       [],
            "val_dist_AP":         [],
            "val_dist_AN":         [],
        }
        best_metric = 0.0

        if resume:
            loaded_epoch, loaded_history = self.load_checkpoint(optimizer, resume_ckpt)
            if loaded_history is not None:
                history = loaded_history
                start_epoch = loaded_epoch + 1
                # best_metric 从 checkpoint 存的历史中取
                best_metric = history.get("best_metric", 0.0)
                logging.info(f"从 epoch {start_epoch}/{epochs} 继续训练")
            else:
                logging.info("未找到有效 checkpoint，从头开始训练")

        warmup_scheduler = LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0,
            total_iters=self.warmup_epochs
        )

        # warmup: 跳过已完成的 epoch 数
        warmup_done = min(start_epoch, self.warmup_epochs)
        if warmup_done > 0:
            for _ in range(warmup_done):
                warmup_scheduler.step()
            logging.info(f"Warmup 已跳过 {warmup_done} 步（从 checkpoint 恢复）")
        logging.info(
            f"Linear Warmup：{self.warmup_epochs} epochs，"
            f"lr {base_lr*0.1:.2e} → {base_lr:.2e}"
        )
        logging.info(f"开始训练，共 {epochs} epochs，设备：{self.device}")

        for epoch in range(start_epoch, epochs):
            # ── Self-Guided Semi-Hard Negative：更新 embedding 缓存 ──────────
            # Epoch 0：模型随机初始化，跳过（退化为随机采样）
            # Epoch 1+：模型已有初步学习，开始真正的 semi-hard mining
            if epoch > 0:
                self._update_train_embeddings()
                self._update_val_embeddings()

            # 每个 epoch 开始前重新采样 anchor 截断池
            if hasattr(self.train_dataloader.dataset, 'on_epoch_start'):
                self.train_dataloader.dataset.on_epoch_start()
            if self.val_dataloader is not None and hasattr(self.val_dataloader.dataset, 'on_epoch_start'):
                self.val_dataloader.dataset.on_epoch_start()

            current_lr = optimizer.param_groups[0]['lr']
            logging.info(f"Epoch {epoch+1}/{epochs}，LR：{current_lr:.2e}")

            train_loss, train_acc, t_loss, dist_AP, dist_AN = self._train_one_epoch(
                criterion, optimizer
            )
            history["train_losses"].append(train_loss)
            history["train_accuracies"].append(train_acc)
            history["train_triplet_losses"].append(t_loss)
            history["train_dist_AP"].append(dist_AP)
            history["train_dist_AN"].append(dist_AN)
            logging.info(
                f"Train  loss={train_loss:.4f} "
                f"(triplet={t_loss:.4f})  "
                f"acc={train_acc:.4f}  "
                f"dist_AP={dist_AP:.4f}  dist_AN={dist_AN:.4f}"
            )

            # triplet val（仅记录，不决定best；距离打印在 _test_one_epoch 内部）
            val_loss, val_acc, val_dist_AP, val_dist_AN = self._test_one_epoch(criterion)
            history["val_losses"].append(val_loss)
            history["val_accuracies"].append(val_acc)
            history["val_dist_AP"].append(val_dist_AP)
            history["val_dist_AN"].append(val_dist_AN)

            # 真实val precision
            real_precision = 0.0
            if self.real_val_evaluator is not None:
                embedding_net = (
                    self.model.embedding_net
                    if hasattr(self.model, "embedding_net")
                    else self.model
                )
                real_precision = self.real_val_evaluator.evaluate(embedding_net)
                history["real_val_precisions"].append(real_precision)
                logging.info(f"Val(real)  precision={real_precision:.4f}  [决定best model]")
                self.model.train()  # 评估后切回训练模式

            self._save_model(epoch, optimizer)

            current_metric = real_precision if self.real_val_evaluator else val_acc
            if current_metric > best_metric:
                best_metric = current_metric
                history["best_metric"] = best_metric
                self._save_best_model(optimizer)
                metric_name = "real_precision" if self.real_val_evaluator else "val_acc"
                logging.info(f"  ✓ best model更新：{metric_name}={current_metric:.4f}")

            if epoch < self.warmup_epochs:
                warmup_scheduler.step()

            # 每个 epoch 保存断点（覆盖式，只保留最新）
            self._save_checkpoint(epoch, optimizer, current_metric, history)

        self._save_training_history(history)
        self._plot_and_save_curves(history)
        logging.info("训练完成。")

    def _train_one_epoch(
        self,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
    ) -> Tuple[float, float, float, float, float]:
        self.model.train()
        epoch_loss     = 0.0
        epoch_triplet = 0.0
        epoch_correct = 0
        total_samples = 0
        total_dist_AP = 0.0
        total_dist_AN = 0.0
        total_batches = 0

        for batch in self.train_dataloader:
            if len(batch) == 6:
                A, P, N, _, _, _ = batch
            else:
                A, P, N = batch[:3]
            A, P, N = A.to(self.device), P.to(self.device), N.to(self.device)

            dist_AP, dist_AN, emb_A, emb_P, emb_N = self.model(A, P, N)
            loss = criterion(emb_A, emb_P, emb_N)
            epoch_loss    += loss.item()
            epoch_triplet += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_correct += (dist_AP < dist_AN).sum().item()
            total_samples += A.size(0)
            total_dist_AP += dist_AP.mean().item()
            total_dist_AN += dist_AN.mean().item()
            total_batches += 1

        n = len(self.train_dataloader)
        return (
            epoch_loss    / n,
            epoch_correct / total_samples if total_samples > 0 else 0.0,
            epoch_triplet / n,
            total_dist_AP / total_batches,
            total_dist_AN / total_batches,
        )

    def _test_one_epoch(self, criterion: nn.Module) -> Tuple[float, float, float, float]:
        self.model.eval()
        epoch_loss     = 0.0
        epoch_correct = 0
        total_dist_AP  = 0.0
        total_dist_AN  = 0.0
        total_batches  = 0

        if self.val_dataloader is None:
            return 0.0, 0.0, 0.0, 0.0

        with torch.no_grad():
            for batch in self.val_dataloader:
                if len(batch) == 6:
                    A, P, N, _, _, _ = batch
                else:
                    A, P, N = batch[:3]
                A, P, N = A.to(self.device), P.to(self.device), N.to(self.device)
                dist_AP, dist_AN, emb_A, emb_P, emb_N = self.model(A, P, N)
                loss = criterion(emb_A, emb_P, emb_N)
                epoch_loss    += loss.item()
                epoch_correct += (dist_AP < dist_AN).sum().item()
                total_dist_AP += dist_AP.mean().item()
                total_dist_AN += dist_AN.mean().item()
                total_batches += 1

        n = len(self.val_dataloader)
        avg_loss  = epoch_loss / n
        avg_acc   = epoch_correct / len(self.val_dataloader.dataset)
        avg_AP    = total_dist_AP / total_batches
        avg_AN    = total_dist_AN / total_batches
        logging.info(
            f"Val(triplet) loss={avg_loss:.4f}  acc={avg_acc:.4f}  "
            f"dist_AP={avg_AP:.4f}  dist_AN={avg_AN:.4f}  [仅供参考]"
        )
        return avg_loss, avg_acc, avg_AP, avg_AN

    def _save_best_model(self, optimizer: optim.Optimizer) -> None:
        torch.save(self.model.state_dict(), self.output_dir / "best_model.pth")
        torch.save(optimizer.state_dict(), self.output_dir / "best_optimizer.pth")
        logging.info("Best model saved.")

    def _save_model(self, epoch: int, optimizer: optim.Optimizer) -> None:
        torch.save(
            self.model.state_dict(),
            self.output_dir / f"fewshot_model_state_dict_epoch_{epoch}.pth"
        )
        torch.save(
            optimizer.state_dict(),
            self.output_dir / f"fewshot_optimizer_state_dict_epoch_{epoch}.pth"
        )
        torch.save(
            self.model,
            self.output_dir / f"fewshot_model_epoch_{epoch}.pth"
        )

    def _save_training_history(self, history: dict) -> None:
        with open(self.output_dir / "training_history.pkl", "wb") as f:
            pickle.dump(history, f)
        logging.info("Training history saved.")

    # ── 断点保存 ──────────────────────────────────────────────────────────────
    def _save_checkpoint(
        self,
        epoch: int,
        optimizer: optim.Optimizer,
        metric: float,
        history: dict,
    ) -> None:
        ckpt = {
            "epoch":             epoch,
            "model_state_dict":  self.model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metric":            metric,
            "best_metric":       history.get("best_metric", 0.0),
            "history":           history,
        }
        torch.save(ckpt, self.output_dir / "checkpoint_latest.pth")
        logging.debug(f"Checkpoint saved at epoch {epoch}")

    def load_checkpoint(
        self,
        optimizer: optim.Optimizer,
        ckpt_path: str = "checkpoint_latest.pth",
    ) -> Tuple[int, dict]:
        """
        加载断点，恢复训练状态。

        Returns:
            (start_epoch, history): 从 start_epoch+1 继续训练
        """
        path = self.output_dir / ckpt_path
        if not path.exists():
            logging.warning(f"Checkpoint not found: {path}，从头开始训练")
            return 0, None

        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        loaded_epoch   = ckpt["epoch"]
        history        = ckpt.get("history")
        logging.info(f"Checkpoint loaded: epoch={loaded_epoch}, metric={ckpt.get('metric'):.4f}")
        return loaded_epoch, history

    # ── 训练可视化 ────────────────────────────────────────────────────────────
    def _plot_and_save_curves(self, history: dict) -> None:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            logging.warning("matplotlib not available, skip plotting")
            return

        epochs = list(range(1, len(history["train_losses"]) + 1))

        # ── 图 1: Loss 曲线 ─────────────────────────────────────────────────
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].plot(epochs, history["train_losses"], label="Train Loss", marker="o", ms=3)
        if history["val_losses"]:
            axes[0].plot(epochs, history["val_losses"], label="Val Loss", marker="s", ms=3)
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Loss vs Epoch")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(epochs, history["train_triplet_losses"], label="Triplet Loss", marker="o", ms=3)
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Loss")
        axes[1].set_title("Component Losses vs Epoch")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        fig.suptitle("Training Loss Curves", fontsize=14)
        fig.tight_layout()
        fig.savefig(self.output_dir / "loss_curves.png", dpi=150)
        plt.close(fig)
        logging.info(f"Loss curves saved: {self.output_dir / 'loss_curves.png'}")

        # ── 图 2: Accuracy / Precision 曲线 ───────────────────────────────
        fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))

        axes2[0].plot(epochs, history["train_accuracies"], label="Train Triplet Acc", marker="o", ms=3)
        if history["val_accuracies"]:
            axes2[0].plot(epochs, history["val_accuracies"], label="Val Triplet Acc", marker="s", ms=3)
        axes2[0].set_xlabel("Epoch")
        axes2[0].set_ylabel("Accuracy")
        axes2[0].set_title("Triplet Ordering Accuracy vs Epoch")
        axes2[0].legend()
        axes2[0].grid(True, alpha=0.3)

        if history["real_val_precisions"] and any(v > 0 for v in history["real_val_precisions"]):
            axes2[1].plot(epochs, history["real_val_precisions"], label="Val Macro Precision", marker="D", ms=3, color="green")
            axes2[1].set_xlabel("Epoch")
            axes2[1].set_ylabel("Macro Precision")
            axes2[1].set_title("Real Val Macro Precision vs Epoch")
            axes2[1].legend()
            axes2[1].grid(True, alpha=0.3)
        else:
            axes2[1].text(0.5, 0.5, "No real val precision recorded", ha="center", va="center")

        fig2.suptitle("Training Accuracy Curves", fontsize=14)
        fig2.tight_layout()
        fig2.savefig(self.output_dir / "accuracy_curves.png", dpi=150)
        plt.close(fig2)
        logging.info(f"Accuracy curves saved: {self.output_dir / 'accuracy_curves.png'}")

        # ── 图 3: 距离曲线 (dist_AP, dist_AN) ──────────────────────────────
        if history["train_dist_AP"] and any(v > 0 for v in history["train_dist_AP"]):
            fig3, axes3 = plt.subplots(1, 2, figsize=(14, 5))

            axes3[0].plot(epochs, history["train_dist_AP"], label="Train dist(Anchor, Positive)", marker="o", ms=3, color="blue")
            axes3[0].plot(epochs, history["train_dist_AN"], label="Train dist(Anchor, Negative)", marker="o", ms=3, color="red")
            axes3[0].set_xlabel("Epoch")
            axes3[0].set_ylabel("Distance")
            axes3[0].set_title("Training Pair Distances vs Epoch")
            axes3[0].legend()
            axes3[0].grid(True, alpha=0.3)

            if history["val_dist_AP"] and any(v > 0 for v in history["val_dist_AP"]):
                axes3[1].plot(epochs, history["val_dist_AP"], label="Val dist(Anchor, Positive)", marker="s", ms=3, color="blue")
                axes3[1].plot(epochs, history["val_dist_AN"], label="Val dist(Anchor, Negative)", marker="s", ms=3, color="red")
                axes3[1].set_xlabel("Epoch")
                axes3[1].set_ylabel("Distance")
                axes3[1].set_title("Validation Pair Distances vs Epoch")
                axes3[1].legend()
                axes3[1].grid(True, alpha=0.3)
            else:
                axes3[1].text(0.5, 0.5, "No val distance recorded", ha="center", va="center")

            fig3.suptitle("Embedding Distance Curves", fontsize=14)
            fig3.tight_layout()
            fig3.savefig(self.output_dir / "distance_curves.png", dpi=150)
            plt.close(fig3)
            logging.info(f"Distance curves saved: {self.output_dir / 'distance_curves.png'}")
