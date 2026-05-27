"""
Photo-SVG Triplet Dataset

用于SVG-Guided Structure Enhancement的训练。

每个样本返回：
- anchor: Photo图像
- positive: 同类SVG图像
- negative: 异类Photo图像

参考Baseline的 EpisodicTripletDatasetFromDir 实现Semi-Hard Negative Mining。
"""

import os
import random
from pathlib import Path
from typing import Union, Optional, Callable, Tuple, List, Dict
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.io import decode_image


def decode_image_from_path(image_path):
    """Helper function to decode image from path."""
    with open(image_path, 'rb') as f:
        img_bytes = f.read()
    img_tensor = torch.tensor(list(img_bytes), dtype=torch.uint8)
    image = decode_image(img_tensor)
    return image


class PhotoSVGTripletDataset(Dataset):
    """
    Photo-SVG Triplet Dataset
    
    同时加载Photo和SVG，用于SVG引导的结构增强训练。
    
    目录结构：
    photo_dir/
    ├── class_001/
    │   ├── img_001.png
    │   └── img_002.png
    └── class_002/
    
    svg_dir/
    ├── class_001/
    │   ├── svg_001.png
    │   └── svg_002.png
    └── class_002/
    
    核心机制：
    - Anchor: Photo（同类）
    - Positive: 同类SVG（随机选择）
    - Negative: 异类Photo（Semi-Hard Negative Mining）
    
    Args:
        photo_root_dir: Photo图像根目录
        svg_root_dir: SVG图像根目录
        transform: 图像transform
        max_per_class: 每类每epoch最多采样数
        semi_hard_candidates: Semi-Hard Negative候选数量
        semi_hard_margin: Semi-Hard Margin
    """

    def __init__(
        self,
        photo_root_dir: Union[str, Path],
        svg_root_dir: Union[str, Path],
        transform: Optional[Callable] = None,
        max_per_class: int = 400,
        semi_hard_candidates: int = 10,
        semi_hard_margin: float = 1.0,
    ) -> None:
        self.photo_root_dir = Path(photo_root_dir) if not isinstance(photo_root_dir, Path) else photo_root_dir
        self.svg_root_dir = Path(svg_root_dir) if not isinstance(svg_root_dir, Path) else svg_root_dir
        self.transform = transform
        self.max_per_class = max_per_class
        self.semi_hard_candidates = semi_hard_candidates
        self.semi_hard_margin = semi_hard_margin

        # Photo池（用于anchor和negative）
        self.photo_class_to_images: Dict[str, List[Path]] = {}
        self.photo_image_list: List[Tuple[str, Path]] = []

        # SVG池（用于positive）
        self.svg_class_to_images: Dict[str, List[Path]] = {}

        # 建立索引
        self._build_index(self.photo_root_dir, self.photo_class_to_images, self.photo_image_list)
        self._build_index(self.svg_root_dir, self.svg_class_to_images, None)

        # 检查类别是否匹配
        photo_classes = set(self.photo_class_to_images.keys())
        svg_classes = set(self.svg_class_to_images.keys())
        
        if photo_classes != svg_classes:
            missing_in_svg = photo_classes - svg_classes
            missing_in_photo = svg_classes - photo_classes
            if missing_in_svg:
                print(f"[PhotoSVGTripletDataset] 警告：Photo中有但SVG中没有的类别: {missing_in_svg}")
            if missing_in_photo:
                print(f"[PhotoSVGTripletDataset] 警告：SVG中有但Photo中没有的类别: {missing_in_photo}")
        
        # 类平衡采样
        self.photo_classes: List[str] = sorted(self.photo_class_to_images.keys())
        
        # 类别名 -> 整数id
        self.class_to_id: Dict[str, int] = {
            cls: i for i, cls in enumerate(self.photo_classes)
        }
        self.id_to_class: Dict[int, str] = {
            i: cls for cls, i in self.class_to_id.items()
        }
        
        self.num_classes: int = len(self.photo_classes)

        # 当epoch的截断池
        self.current_epoch_images: List[Tuple[Path, str]] = []

        # Self-Guided Embedding缓存
        self._cached_embeddings: Dict[Path, torch.Tensor] = {}
        self._cache_ready: bool = False

        # 初始化
        self.on_epoch_start(self.max_per_class)

        print(f"[PhotoSVGTripletDataset] "
              f"类别数={self.num_classes}, "
              f"max_per_class={max_per_class}, "
              f"semi_hard_candidates={semi_hard_candidates}, "
              f"采样策略=类平衡 + Semi-Hard Negative")

    def _build_index(
        self,
        root: Path,
        class_to_images: dict,
        image_list: Optional[List[Tuple[str, Path]]],
    ) -> None:
        """建立类别到图像的索引"""
        IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
        
        if not root.exists():
            print(f"[PhotoSVGTripletDataset] 警告：目录不存在 {root}")
            return
        
        for class_folder in root.iterdir():
            if class_folder.is_dir():
                class_label = class_folder.name
                images = [
                    p for p in class_folder.iterdir()
                    if p.is_file() and p.suffix.lower() in IMAGE_EXTS
                ]
                if images:
                    class_to_images[class_label] = images
                    if image_list is not None:
                        for img_path in images:
                            image_list.append((class_label, img_path))
                else:
                    print(f"Warning: No images found in class folder {class_folder}")

    def update_embeddings(
        self,
        embedding_net: torch.nn.Module,
        device: str,
        transform,
    ) -> None:
        """
        用当前训练模型重新计算所有Photo样本的embedding并缓存。
        
        Args:
            embedding_net: 当前训练的EmbeddingNet模型
            device: 计算设备
            transform: 与原型构建一致的transform
        """
        embedding_net.eval()
        new_cache: Dict[Path, torch.Tensor] = {}

        # 只缓存Photo（negative来自Photo池）
        all_paths = set(
            path
            for paths in self.photo_class_to_images.values()
            for path in paths
        )

        print(f"[SemiHard] 更新Photo embedding缓存，共 {len(all_paths)} 张图...")

        batch_size = 64
        tensors = []
        path_batch = []

        with torch.no_grad():
            for i, path in enumerate(all_paths):
                try:
                    img = decode_image_from_path(str(path))
                    img = img.float() / 255.0
                    img = transform(img)
                    tensors.append(img)
                    path_batch.append(path)
                except Exception as e:
                    print(f"[SemiHard] 跳过损坏图片 {path}: {e}")
                    continue

                if len(tensors) == batch_size or i == len(all_paths) - 1:
                    if tensors:
                        batch = torch.stack(tensors).to(device)
                        embs = embedding_net(batch)

                        for path_item, emb in zip(path_batch, embs):
                            new_cache[path_item] = emb.detach().cpu()

                    tensors = []
                    path_batch = []

        self._cached_embeddings = new_cache
        self._cache_ready = True
        embedding_net.train()
        print(f"[SemiHard] 缓存更新完成 ({len(new_cache)} 张图)")

    def sample_photo_from_class(
        self,
        class_label: str,
        exclude_path: Optional[Path] = None,
    ) -> Tuple[torch.Tensor, Path]:
        """
        从指定 photo 类别里随机采样一张图。

        这个接口用于训练阶段计算 d_ap_intra（photo ↔ photo 同类距离）。
        """
        candidates = self.photo_class_to_images.get(class_label, [])
        if not candidates:
            raise ValueError(f"Class '{class_label}' not found in photo pool.")

        if exclude_path is not None and len(candidates) > 1:
            filtered = [p for p in candidates if p != exclude_path]
            if filtered:
                candidates = filtered

        photo_path = random.choice(candidates)
        photo_img = decode_image_from_path(str(photo_path))
        photo_img = photo_img.float() / 255.0
        if self.transform:
            photo_img = self.transform(photo_img)
        return photo_img, photo_path

    def sample_photo_from_class_id(
        self,
        class_id: int,
        exclude_path: Optional[Path] = None,
    ) -> Tuple[torch.Tensor, Path]:
        class_label = self.id_to_class.get(int(class_id))
        if class_label is None:
            raise ValueError(f"Class id '{class_id}' not found in photo pool.")
        return self.sample_photo_from_class(class_label, exclude_path=exclude_path)

    def _mine_semi_hard_negative(
        self,
        anchor_path: Path,
        positive_path: Path,
        negative_classes: list,
    ) -> Tuple[Path, str]:
        """
        Semi-Hard Negative Mining
        
        定义：d(a, p) < d(a, n) < d(a, p) + margin
        """
        anchor_emb = self._cached_embeddings.get(anchor_path)
        pos_emb = self._cached_embeddings.get(positive_path)

        if anchor_emb is None or pos_emb is None:
            neg_class = random.choice(negative_classes)
            neg_path = random.choice(self.photo_class_to_images[neg_class])
            return neg_path, neg_class

        d_ap = F.pairwise_distance(
            anchor_emb.unsqueeze(0),
            pos_emb.unsqueeze(0)
        ).item()

        # 随机抽候选
        candidates: List[Tuple[Path, float]] = []
        sampled_classes = random.choices(negative_classes, k=self.semi_hard_candidates)

        for neg_cls in sampled_classes:
            pool = self.photo_class_to_images[neg_cls]
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
            neg_path = random.choice(self.photo_class_to_images[neg_class])
            return neg_path, neg_class

        # 优先选Semi-Hard
        semi_hard = [
            (p, d) for p, d in candidates
            if d_ap < d < d_ap + self.semi_hard_margin
        ]
        if semi_hard:
            semi_hard.sort(key=lambda x: abs(x[1] - d_ap))
            neg_path, _ = semi_hard[0]
            return neg_path, self._get_class_for_path(neg_path)

        # fallback 1: Hard Negative
        hard = [(p, d) for p, d in candidates if d <= d_ap]
        if hard:
            hard.sort(key=lambda x: x[1], reverse=True)
            neg_path, _ = hard[0]
            return neg_path, self._get_class_for_path(neg_path)

        # fallback 2: 随机
        neg_path = random.choice(candidates)[0]
        return neg_path, self._get_class_for_path(neg_path)

    def _get_class_for_path(self, path: Path) -> str:
        """获取路径对应的类别"""
        for cls, paths in self.photo_class_to_images.items():
            if path in paths:
                return cls
        return path.parent.name

    def on_epoch_start(self, max_per_class: Optional[int] = None) -> None:
        """每个epoch开始前调用"""
        if max_per_class is None:
            max_per_class = self.max_per_class
        self.current_epoch_images = []
        for label, all_imgs in self.photo_class_to_images.items():
            if len(all_imgs) <= max_per_class:
                sampled = all_imgs[:]
            else:
                sampled = random.sample(all_imgs, max_per_class)
            self.current_epoch_images.extend([(path, label) for path in sampled])
        random.shuffle(self.current_epoch_images)

    def __len__(self) -> int:
        return len(self.current_epoch_images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int, int]:
        """
        Returns:
            anchor_photo: [C, H, W] Photo图像
            positive_svg: [C, H, W] 同类SVG图像
            negative_photo: [C, H, W] 异类Photo图像
            label_A: int, anchor/positive的类别id
            label_P: int, 与label_A相同
            label_N: int, negative的类别id
        """
        # Anchor (Photo)
        anchor_path, anchor_label = self.current_epoch_images[idx]
        anchor_img = decode_image_from_path(str(anchor_path))
        anchor_img = anchor_img.float() / 255.0

        # Positive (同类SVG)
        if anchor_label not in self.svg_class_to_images:
            raise ValueError(
                f"Class '{anchor_label}' not found in SVG directory. "
                f"Please ensure SVG directory has the same structure as Photo."
            )
        
        svg_candidates = [p for p in self.svg_class_to_images[anchor_label]]
        positive_svg_path = random.choice(svg_candidates)
        positive_svg_img = decode_image_from_path(str(positive_svg_path))
        positive_svg_img = positive_svg_img.float() / 255.0

        # Negative (异类Photo, Semi-Hard Mining)
        negative_classes = [cls for cls in self.photo_class_to_images.keys() if cls != anchor_label]
        if not negative_classes:
            raise ValueError("Only one class available; cannot sample negative.")

        if self._cache_ready:
            negative_path, negative_class = self._mine_semi_hard_negative(
                anchor_path, positive_svg_path, negative_classes
            )
        else:
            negative_class = random.choice(negative_classes)
            negative_path = random.choice(self.photo_class_to_images[negative_class])

        negative_img = decode_image_from_path(str(negative_path))
        negative_img = negative_img.float() / 255.0

        if self.transform:
            anchor_img = self.transform(anchor_img)
            positive_svg_img = self.transform(positive_svg_img)
            negative_img = self.transform(negative_img)

        label_A = self.class_to_id[anchor_label]
        label_P = label_A
        label_N = self.class_to_id.get(negative_class, -1)

        return anchor_img, positive_svg_img, negative_img, label_A, label_P, label_N

    def get_total_images(self) -> int:
        """返回全量图片总数"""
        return len(self.photo_image_list)


class HybridPhotoSVGTripletDataset(PhotoSVGTripletDataset):
    """
    Hybrid Photo-SVG Triplet Dataset.

    Compared with PhotoSVGTripletDataset, this dataset returns both a same-class
    photo positive and a same-class SVG positive:

        anchor_photo, positive_photo, positive_svg, negative_photo, label_A, label_N

    This supports hybrid loss:
        L_photo_triplet(anchor_photo, positive_photo, negative_photo)
      + lambda_svg * L_svg_triplet(anchor_photo, positive_svg, negative_photo)
    """

    def __getitem__(
        self,
        idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
        anchor_path, anchor_label = self.current_epoch_images[idx]
        anchor_img = decode_image_from_path(str(anchor_path))
        anchor_img = anchor_img.float() / 255.0

        photo_candidates = [
            p for p in self.photo_class_to_images.get(anchor_label, [])
            if p != anchor_path
        ]
        if photo_candidates:
            positive_photo_path = random.choice(photo_candidates)
        else:
            positive_photo_path = anchor_path
            print(
                f"Warning: Only one Photo image in class '{anchor_label}'. "
                "Using anchor as positive_photo."
            )
        positive_photo_img = decode_image_from_path(str(positive_photo_path))
        positive_photo_img = positive_photo_img.float() / 255.0

        if anchor_label not in self.svg_class_to_images:
            raise ValueError(
                f"Class '{anchor_label}' not found in SVG directory. "
                f"Please ensure SVG directory has the same structure as Photo."
            )
        positive_svg_path = random.choice(self.svg_class_to_images[anchor_label])
        positive_svg_img = decode_image_from_path(str(positive_svg_path))
        positive_svg_img = positive_svg_img.float() / 255.0

        negative_classes = [
            cls for cls in self.photo_class_to_images.keys()
            if cls != anchor_label
        ]
        if not negative_classes:
            raise ValueError("Only one class available; cannot sample negative.")

        if self._cache_ready:
            negative_path, negative_class = self._mine_semi_hard_negative(
                anchor_path, positive_photo_path, negative_classes
            )
        else:
            negative_class = random.choice(negative_classes)
            negative_path = random.choice(self.photo_class_to_images[negative_class])

        negative_img = decode_image_from_path(str(negative_path))
        negative_img = negative_img.float() / 255.0

        if self.transform:
            anchor_img = self.transform(anchor_img)
            positive_photo_img = self.transform(positive_photo_img)
            positive_svg_img = self.transform(positive_svg_img)
            negative_img = self.transform(negative_img)

        label_A = self.class_to_id[anchor_label]
        label_N = self.class_to_id.get(negative_class, -1)

        return (
            anchor_img,
            positive_photo_img,
            positive_svg_img,
            negative_img,
            label_A,
            label_N,
        )
