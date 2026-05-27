# stage2组织数据集文件
from pathlib import Path
import random
import shutil
from typing import List, Dict, Tuple, Optional, Sequence
from utils.helpers import get_files
import logging


class ClassCropsBuilder:
    def __init__(self, crops_root_dir: Path):
        """
        Initialize the ClassCropsBuilder.

        Args:
            crops_root_dir (Path): Path to the root directory containing symbol crops.
                                   Structure: crops_root_dir/{class_id}/*.jpg
        """
        self.crops_root_dir = crops_root_dir
        self.logger = logging.getLogger(__name__)

    def _validate_data(self) -> bool:
        """
        Validate the data in the crops directory.

        Returns:
            bool: True if validation is successful, False otherwise.
        """
        if len(get_files(self.crops_root_dir, extensions=['.jpg', '.png', '.jpeg'])) == 0:
            raise FileNotFoundError(f"Directory {self.crops_root_dir} is empty.")

        class_dirs = [d for d in self.crops_root_dir.iterdir() if d.is_dir()]
        if not class_dirs:
            raise ValueError("No class directories found in the symbol crops directory.")

        return True

    def split_train_val(
        self,
        train_dir: Path,
        val_dir: Path,
        train_val_split: Sequence[float] = (0.8, 0.2),
        seed: int = 42,
    ) -> Dict[str, Dict]:
        """
        Split symbol crops into train and val directories per class.

        Args:
            train_dir (Path): Output directory for training data.
            val_dir (Path): Output directory for validation data.
            train_val_split (Sequence[float]): Split ratios, e.g. [0.8, 0.2].
                                              Must sum to 1.0.
            seed (int): Random seed for reproducibility.

        Returns:
            Dict with statistics: {'train': {class_id: count}, 'val': {class_id: count}}
        """
        if not self._validate_data():
            self.logger.error("Data validation failed.")
            return {}

        if abs(sum(train_val_split) - 1.0) > 1e-6:
            raise ValueError(f"train_val_split must sum to 1.0, got {train_val_split}")

        train_ratio, val_ratio = train_val_split[0], train_val_split[1]
        random.seed(seed)

        train_dir.mkdir(parents=True, exist_ok=True)
        val_dir.mkdir(parents=True, exist_ok=True)

        stats = {
            'train': {},
            'val': {},
            'total_train': 0,
            'total_val': 0,
        }

        class_dirs = sorted([
            d for d in self.crops_root_dir.iterdir() if d.is_dir()
        ])

        for class_dir in class_dirs:
            class_id = class_dir.name
            image_paths = get_files(class_dir, extensions=['.jpg', '.png', '.jpeg'])

            if len(image_paths) == 0:
                print(f"⚠️ Class '{class_id}' has no images — skipping.")
                continue

            random.shuffle(image_paths)

            n_val = max(1, int(len(image_paths) * val_ratio))
            # Ensure at least 1 sample in train if possible
            n_val = min(n_val, len(image_paths) - 1)

            val_paths = image_paths[:n_val]
            train_paths = image_paths[n_val:]

            train_class_dir = train_dir / class_id
            val_class_dir = val_dir / class_id
            train_class_dir.mkdir(parents=True, exist_ok=True)
            val_class_dir.mkdir(parents=True, exist_ok=True)

            for img_path in train_paths:
                shutil.copy2(img_path, train_class_dir / img_path.name)

            for img_path in val_paths:
                shutil.copy2(img_path, val_class_dir / img_path.name)

            stats['train'][class_id] = len(train_paths)
            stats['val'][class_id] = len(val_paths)
            stats['total_train'] += len(train_paths)
            stats['total_val'] += len(val_paths)

            print(
                f"Class '{class_id}': {len(train_paths)} train, "
                f"{len(val_paths)} val (from {len(image_paths)} total)"
            )

        print(
            f"\n✅ Split complete:\n"
            f"   Train: {stats['total_train']} images in {train_dir}\n"
            f"   Val:   {stats['total_val']} images in {val_dir}"
        )
        return stats

    def sample_train_data(self, k: int, output_dir: Path) -> None:
        """
        DEPRECATED: Use split_train_val() instead.

        Sample k images per class from symbol crops to create training data.

        Args:
            k (int): Maximum number of samples per class.
            output_dir (Path): Directory to save the sampled data.
        """
        self.logger.warning(
            "sample_train_data() is deprecated. Use split_train_val() instead."
        )

        if not self._validate_data():
            self.logger.error("Data validation failed.")
            return

        output_dir.mkdir(parents=True, exist_ok=True)
        class_dirs = [d for d in self.crops_root_dir.iterdir() if d.is_dir()]

        for class_dir in class_dirs:
            class_id = class_dir.name
            image_paths = get_files(class_dir, extensions=['.jpg', '.png', '.jpeg'])
            if len(image_paths) == 0:
                print(f"⚠️ No images found in class '{class_id}' — skipping.")
                continue
            elif len(image_paths) < k:
                print(f"⚠️ Class '{class_id}' has only {len(image_paths)} images — using all.")
                selected = image_paths
            else:
                selected = random.sample(image_paths, k)

            saved_dir = output_dir / class_id
            saved_dir.mkdir(parents=True, exist_ok=True)
            for image_path in selected:
                shutil.copy2(image_path, saved_dir / image_path.name)

        print(f"✅ Training data sampled at: {output_dir}")
