import os
import random
from pathlib import Path
from typing import Union, Optional, Callable, Tuple, List, Dict
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.io import decode_image
from torchvision.transforms import v2


def decode_image_from_path(image_path):
    """Helper function to decode image from path."""
    with open(image_path, 'rb') as f:
        img_bytes = f.read()
    img_tensor = torch.tensor(list(img_bytes), dtype=torch.uint8)
    image = decode_image(img_tensor)
    return image


class EpisodicTripletDatasetFromDir(Dataset):
    """PyTorch Dataset for generating triplets (anchor, positive, negative).

    目录结构要求：每个子文件夹是一个类别，里面是该类别的图片。

    核心机制（Self-Guided Semi-Hard Negative Mining）：
    - 每个 epoch 开始前，Trainer 调用 update_embeddings() 用当前模型计算所有样本的 embedding
    - Epoch 0：模型随机初始化，缓存无效，退化为随机采样
    - Epoch 1+：缓存有效，开始真正的 semi-hard negative mining
    - Semi-Hard 定义：d(a,p) < d(a,n) < d(a,p) + margin
      （比 positive 远，但没远出 margin，有训练价值但不会太难）

    Semi-Hard Negative 采样流程：
      1. 已知 anchor 的 cached embedding
      2. 从同类中随机选 positive，计算 pos_dist
      3. 从异类中随机抽取 n_candidates 个候选
      4. 计算每个候选到 anchor 的距离
      5. 筛选出 pos_dist < neg_dist < pos_dist + margin 的候选
      6. 若有多个，选择 neg_dist 最接近 pos_dist 的那个
      7. 若没有，回退到随机采样

    Args:
        root_dir (Union[str, Path]): 兼容旧接口：作为 anchor_root_dir 使用。
        transform (Optional[Callable]): 图像 transform。
        max_per_class (int): 每类每 epoch 最多采样张数。
        semi_hard_candidates (int): Semi-Hard Negative 候选数量，从异类中抽取。
        pn_root_dir (Union[str, Path], optional): Positive/Negative 的根目录。

    Returns:
        Tuple: (anchor, positive, negative, label_A, label_P, label_N)
               label_A == label_P（同类），label_N 不同类
    """

    def __init__(
        self,
        root_dir: Union[str, Path],
        transform: Optional[Callable] = None,
        max_per_class: int = 400,
        semi_hard_candidates: int = 10,
        pn_root_dir: Optional[Union[str, Path]] = None,
        semi_hard_margin: float = 1.0,
    ) -> None:
        self.anchor_root_dir = Path(root_dir) if not isinstance(root_dir, Path) else root_dir
        self.pn_root_dir = (
            Path(pn_root_dir) if (pn_root_dir is not None and not isinstance(pn_root_dir, Path)) else pn_root_dir
        )
        if self.pn_root_dir is None:
            self.pn_root_dir = self.anchor_root_dir

        self.transform = transform
        self.max_per_class = max_per_class
        self.semi_hard_candidates = semi_hard_candidates
        self.semi_hard_margin = semi_hard_margin

        # Anchor 池
        self.anchor_class_to_images: Dict[str, List[Path]] = {}
        self.anchor_image_list: List[Tuple[str, Path]] = []

        # Positive/Negative 池
        self.pn_class_to_images: Dict[str, List[Path]] = {}

        self._build_index(self.anchor_root_dir, self.anchor_class_to_images, self.anchor_image_list)
        self._build_index(self.pn_root_dir, self.pn_class_to_images, None)

        if not self.anchor_image_list:
            raise ValueError(f"No images found in any subfolder of {self.anchor_root_dir}")
        if not self.pn_class_to_images:
            raise ValueError(f"No images found in any subfolder of {self.pn_root_dir}")

        # 类平衡采样用的类别列表（固定顺序，保证可复现）
        self.anchor_classes: List[str] = sorted(self.anchor_class_to_images.keys())

        # 建立类别名 -> 整数id 的映射
        self.anchor_class_to_id: Dict[str, int] = {
            cls: i for i, cls in enumerate(sorted(self.anchor_class_to_images.keys()))
        }
        pn_only_classes = sorted(
            set(self.pn_class_to_images.keys()) - set(self.anchor_class_to_id.keys())
        )
        offset = len(self.anchor_class_to_id)
        self.pn_class_to_id: Dict[str, int] = dict(self.anchor_class_to_id)
        for i, cls in enumerate(pn_only_classes):
            self.pn_class_to_id[cls] = offset + i

        # 总类别数
        self.num_classes: int = len(self.pn_class_to_id)

        # 当 epoch 的截断池：扁平列表 [(path, label), ...]
        self.current_epoch_images: List[Tuple[Path, str]] = []

        # ── Self-Guided Embedding 缓存（由 Trainer 在每个 epoch 开始前更新）───
        self._cached_embeddings: Dict[Path, torch.Tensor] = {}
        self._cache_ready: bool = False  # 标记缓存是否可用

        # 初始化时生成一次截断池，供第一个 epoch 使用
        self.on_epoch_start(self.max_per_class)

        print(f"[EpisodicTripletDatasetFromDir] "
              f"anchor类别数={len(self.anchor_classes)}, "
              f"max_per_class={max_per_class}, "
              f"semi_hard_candidates={semi_hard_candidates}, "
              f"采样策略=类平衡 + Self-Guided Semi-Hard Negative")

    # ── Self-Guided Embedding 更新（由 Trainer 调用）─────────────────────────

    def update_embeddings(
        self,
        embedding_net: torch.nn.Module,
        device: str,
        transform,
    ) -> None:
        """用当前训练模型重新计算所有样本的 embedding 并缓存。

        Args:
            embedding_net: 当前训练的 EmbeddingNet 模型
            device: 计算设备
            transform: 与原型构建一致的 transform（pad + resize + normalize）

        注意：
            - Epoch 0 调用时模型随机初始化，此时缓存无意义
            - Epoch 1+ 调用时模型已有初步学习，缓存开始有效
        """
        embedding_net.eval()
        new_cache: Dict[Path, torch.Tensor] = {}

        # 收集所有需要计算 embedding 的路径
        # 同时缓存 anchor 池和 pn 池（negative 可能来自 pn 池）
        all_paths = set(
            path
            for paths in self.anchor_class_to_images.values()
            for path in paths
        ) | set(
            path
            for paths in self.pn_class_to_images.values()
            for path in paths
        )

        print(f"[SemiHard] 更新 embedding 缓存，共 {len(all_paths)} 张图...")

        # 批量处理，避免逐张推理太慢
        batch_size = 64
        tensors = []
        path_batch = []

        with torch.no_grad():
            for i, path in enumerate(all_paths):
                try:
                    img = decode_image_from_path(str(path))
                    img = img.float() / 255.0
                    img = transform(img)  # pad + resize + normalize
                    tensors.append(img)
                    path_batch.append(path)
                except Exception as e:
                    print(f"[SemiHard] 跳过损坏图片 {path}: {e}")
                    continue

                # 攒够一批或到最后
                if len(tensors) == batch_size or i == len(all_paths) - 1:
                    if tensors:
                        batch = torch.stack(tensors).to(device)
                        embs = embedding_net(batch)  # [B, D]
                        embs = F.normalize(embs, p=2, dim=1)  # 归一化

                        for path_item, emb in zip(path_batch, embs):
                            new_cache[path_item] = emb.cpu()

                    tensors = []
                    path_batch = []

        self._cached_embeddings = new_cache
        self._cache_ready = True
        embedding_net.train()
        print(f"[SemiHard] 缓存更新完成 ({len(new_cache)} 张图)")

    # ── Semi-Hard Negative Mining ───────────────────────────────────────────

    def _mine_semi_hard_negative(
        self,
        anchor_path: Path,
        positive_path: Path,
        negative_classes: list,
    ) -> Tuple[Path, str]:
        """
        Semi-Hard 定义：
            d(a, p) < d(a, n) < d(a, p) + margin

        即 negative 比 positive 远，但没远出 margin，
        这样 triplet loss > 0，有梯度，但又不是极难样本。

        Returns:
            (negative_path, negative_class)

        找不到满足条件的候选时，退化为最近的异类（Hard Negative）。
        再找不到退化为随机。
        """
        anchor_emb = self._cached_embeddings.get(anchor_path)
        pos_emb = self._cached_embeddings.get(positive_path)

        if anchor_emb is None or pos_emb is None:
            # 缓存中没有这条路径（边缘情况），随机退化
            neg_class = random.choice(negative_classes)
            neg_path = random.choice(self.pn_class_to_images[neg_class])
            return neg_path, neg_class

        d_ap = F.pairwise_distance(
            anchor_emb.unsqueeze(0),
            pos_emb.unsqueeze(0)
        ).item()

        # 随机抽候选：先选类，再从类里随机选样本，保证总数达到 semi_hard_candidates
        candidates: List[Tuple[Path, float]] = []
        sampled_classes = random.choices(negative_classes, k=self.semi_hard_candidates)

        for neg_cls in sampled_classes:
            pool = self.pn_class_to_images[neg_cls]
            if not pool:
                continue
            p = random.choice(pool)
            emb = self._cached_embeddings.get(p)
            if emb is None:
                continue
            d_an = F.pairwise_distance(
                anchor_emb.unsqueeze(0),
                emb.unsqueeze(0)
            ).item()
            candidates.append((p, d_an))

        if not candidates:
            neg_class = random.choice(negative_classes)
            neg_path = random.choice(self.pn_class_to_images[neg_class])
            return neg_path, neg_class

        # ── 优先选 Semi-Hard ─────────────────────────────────────────
        semi_hard = [
            (p, d) for p, d in candidates
            if d_ap < d < d_ap + self.semi_hard_margin
        ]
        if semi_hard:
            # 选距离最接近 d_ap 的（最有训练价值）
            semi_hard.sort(key=lambda x: abs(x[1] - d_ap))
            neg_path, _ = semi_hard[0]
            return neg_path, self._get_class_for_path(neg_path)

        # ── fallback 1：Hard Negative（d_an <= d_ap，最近异类）────────
        hard = [(p, d) for p, d in candidates if d <= d_ap]
        if hard:
            hard.sort(key=lambda x: x[1], reverse=True)  # 选最不难的
            neg_path, _ = hard[0]
            return neg_path, self._get_class_for_path(neg_path)

        # ── fallback 2：随机 ──────────────────────────────────────────
        neg_path = random.choice(candidates)[0]
        return neg_path, self._get_class_for_path(neg_path)

    # ── Index Building ───────────────────────────────────────────────────────

    IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}

    @staticmethod
    def _build_index(
        root: Path,
        class_to_images: dict,
        image_list: Optional[List[Tuple[str, Path]]],
    ) -> None:
        for class_folder in root.iterdir():
            if class_folder.is_dir():
                class_label = class_folder.name
                images = [
                    p for p in class_folder.iterdir()
                    if p.is_file() and p.suffix.lower() in EpisodicTripletDatasetFromDir.IMAGE_EXTS
                ]
                if images:
                    class_to_images[class_label] = images
                    if image_list is not None:
                        for img_path in images:
                            image_list.append((class_label, img_path))
                else:
                    print(f"Warning: No images found in class folder {class_folder}")

    # ── Epoch Management ────────────────────────────────────────────────────

    def on_epoch_start(self, max_per_class: Optional[int] = None) -> None:
        """每个 epoch 开始前调用：对每个类从全量中重新随机采样，更新截断池。"""
        if max_per_class is None:
            max_per_class = self.max_per_class
        self.current_epoch_images = []
        for label, all_imgs in self.anchor_class_to_images.items():
            if len(all_imgs) <= max_per_class:
                sampled = all_imgs[:]
            else:
                sampled = random.sample(all_imgs, max_per_class)
            self.current_epoch_images.extend([(path, label) for path in sampled])
        random.shuffle(self.current_epoch_images)

    def __len__(self) -> int:
        return len(self.current_epoch_images)

    # ── Triplet Sampling ─────────────────────────────────────────────────────

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int, int]:
        """
        Returns:
            anchor_img   (Tensor): [C, H, W]
            positive_img (Tensor): [C, H, W]
            negative_img (Tensor): [C, H, W]
            label_A      (int):    anchor/positive 的类别id（两者相同）
            label_P      (int):    与 label_A 相同
            label_N      (int):    negative 的类别id（与 label_A 不同）
        """
        # ── Anchor：从当 epoch 截断池中取 ────────────────────────────────
        anchor_path, anchor_label = self.current_epoch_images[idx]
        anchor_img = decode_image_from_path(str(anchor_path))
        anchor_img = anchor_img.float() / 255.0

        # ── Positive：从同类中随机取（排除 anchor）──────────────────────
        if anchor_label not in self.pn_class_to_images:
            raise ValueError(
                f"Class '{anchor_label}' not found under pn_root_dir={self.pn_root_dir}. "
                "请确保 pn_root_dir 目录下包含与 anchor 相同的类别子目录。"
            )
        positive_candidates = [p for p in self.pn_class_to_images[anchor_label] if p != anchor_path]
        if positive_candidates:
            positive_path = random.choice(positive_candidates)
        else:
            positive_path = self.pn_class_to_images[anchor_label][0]
            print(f"Warning: Only one PN image in class '{anchor_label}'. Using it as positive.")
        positive_img = decode_image_from_path(str(positive_path))
        positive_img = positive_img.float() / 255.0

        # ── Negative：有缓存就挖掘，没有就随机 ─────────────────────────
        negative_classes = [cls for cls in self.pn_class_to_images.keys() if cls != anchor_label]
        if not negative_classes:
            raise ValueError("Only one class available in PN dataset; cannot sample negative.")

        if self._cache_ready:
            negative_path, negative_class = self._mine_semi_hard_negative(
                anchor_path, positive_path, negative_classes
            )
        else:
            # Epoch 0：纯随机，安全退化
            negative_class = random.choice(negative_classes)
            negative_path = random.choice(self.pn_class_to_images[negative_class])

        negative_img = decode_image_from_path(str(negative_path))
        negative_img = negative_img.float() / 255.0

        if self.transform:
            anchor_img = self.transform(anchor_img)
            positive_img = self.transform(positive_img)
            negative_img = self.transform(negative_img)

        # 取整数标签
        label_A = self.anchor_class_to_id[anchor_label]
        label_P = label_A
        label_N = self.pn_class_to_id.get(negative_class, -1)

        return anchor_img, positive_img, negative_img, label_A, label_P, label_N

    def _get_class_for_path(self, path: Path) -> str:
        """Get the class name for a given image path by checking pn_class_to_images."""
        for cls, paths in self.pn_class_to_images.items():
            if path in paths:
                return cls
        # Fallback: use parent folder name
        return path.parent.name

    def get_total_images(self) -> int:
        """返回全量图片总数（不做截断）。"""
        return len(self.anchor_image_list)

