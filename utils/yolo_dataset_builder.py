import yaml
import shutil
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, Callable, Literal
import logging
import random
from sklearn.model_selection import train_test_split
from utils.helpers import get_files

class YOLODatasetBuilder:
    def __init__(self, images_dir: Path, labels_dir: Path, output_dir: Path):
        """
        Initialize the YOLO dataset builder.
        
        Args:
            images_dir: Directory containing images
            labels_dir: Directory containing corresponding labels
            output_dir: Directory to save the YOLO dataset
        """
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.output_dir = output_dir
        self.logger = logging.getLogger(__name__)
        
    def _validate_data(self) -> bool:
        """Validate that images and labels match."""
        image_files = set(f.stem for f in get_files(self.images_dir, extensions=['.png']))
        label_files = set(f.stem for f in get_files(self.labels_dir, extensions=['.txt']))
        
        if not image_files:
            self.logger.error(f"No images found in {self.images_dir}")
            return False
            
        if not label_files:
            self.logger.error("No labels found in labels directory")
            return False
            
        missing_labels = image_files - label_files
        if missing_labels:
            self.logger.warning(f"Missing labels for images: {missing_labels}")
            
        return True
    
    def _create_yolo_structure(self) -> None:
        """Create the YOLO dataset directory structure."""
        # Create directories
        (self.output_dir / "images/train").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "images/val").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "images/test").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "labels/train").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "labels/val").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "labels/test").mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Created YOLO dataset structure in {self.output_dir}")
        
    def _create_dataset_yaml(self, class_names: List[str]) -> None:
        """Create the dataset.yaml file."""
        data = {
            'train': str((self.output_dir / "images" / "train").resolve()),
            'val': str((self.output_dir / "images" / "val").resolve()),
            'test': str((self.output_dir / "images" / "test").resolve()),
            'nc': len(class_names),
            'names': class_names
        }
        yaml_path = self.output_dir / "dataset.yaml"
        try:
            with open(yaml_path, 'w') as f:
                yaml.dump(data, f, sort_keys=False)
            self.logger.info(f"Created dataset.yaml at: {yaml_path}")
        except Exception as e:
            self.logger.error(f"Error creating dataset.yaml: {e}")
            
    def create_dataset(self, 
                       train_val_test_split: Tuple[float, float, float], 
                       class_names: List[str], 
                       grouping_strategy: Literal['underscore', 'custom'] = 'underscore', 
                       custom_grouping_fn: Optional[Callable[[Path], str]] = None, 
                       random_seed: int = 42) -> None:
        """
        Create a YOLO dataset with train/val/test splits.

        Args:
            train_val_test_split: Tuple of (train_ratio, val_ratio, test_ratio).
            class_names: List of class names corresponding to the labels.
            grouping_strategy: Strategy to group images for splitting ('underscore' or 'custom'). Defaults to 'underscore'.
            custom_grouping_fn: Custom function to group images if grouping_strategy is 'custom'. Defaults to None.
            random_seed: Random seed for shuffling and splitting. Defaults to 42.
        """
        if not abs(sum(train_val_test_split) - 1) < 1e-6:
            raise ValueError(f"Train, validation, and test ratios must sum to 1. Got: {train_val_test_split}")

        if not self._validate_data():
            self.logger.error("Data validation failed. Please check your images and labels.")
            return

        self._create_yolo_structure()
        all_images = get_files(self.images_dir, ['.jpg', '.png', '.jpeg'])

        if grouping_strategy == 'underscore':
            groups = group_images_by_strategy(self.images_dir, 'underscore')
        elif grouping_strategy == 'custom' and custom_grouping_fn:
            groups = group_images_by_strategy(self.images_dir, 'custom', custom_grouping_fn=custom_grouping_fn)
        else:
            self.logger.error(f"Invalid grouping strategy: {grouping_strategy} or custom_grouping_fn not provided.")
            return

        group_keys = list(groups.keys())
        random.seed(random_seed)
        train_keys, val_test_keys = train_test_split(group_keys, test_size=train_val_test_split[1] + train_val_test_split[2], random_state=random_seed)
        val_keys, test_keys = train_test_split(val_test_keys, test_size=train_val_test_split[2] / (train_val_test_split[1] + train_val_test_split[2]), random_state=random_seed)

        train_images = [img for key in train_keys for img in groups[key]]
        val_images = [img for key in val_keys for img in groups[key]]
        test_images = [img for key in test_keys for img in groups[key]]

        self._organize_yolo_folders(train_images, val_images, test_images)
        self._create_dataset_yaml(class_names)
        self.logger.info("YOLO dataset creation complete.")

    def _organize_yolo_folders(self, train_images: List[Path], val_images: List[Path], test_images: List[Path]) -> None:
        """Moves images and labels into the YOLO-specific structure."""
        splits = {
            "train": train_images,
            "val": val_images,
            "test": test_images
        }

        for split_name, images_list in splits.items():
            for img_path in images_list:
                label_path = self.labels_dir / f"{img_path.stem}.txt"

                dst_img = self.output_dir / "images" / split_name / img_path.name
                dst_label = self.output_dir / "labels" / split_name / f"{img_path.stem}.txt"

                dst_img.parent.mkdir(parents=True, exist_ok=True)
                dst_label.parent.mkdir(parents=True, exist_ok=True)

                try:
                    shutil.copy2(img_path, dst_img)
                    if label_path.exists():
                        shutil.copy2(label_path, dst_label)
                    else:
                        self.logger.warning(f"No label found for {img_path.name}.")
                except Exception as e:
                    self.logger.error(f"Error copying files: {e}")


def default_grouping_function(image_path: Path) -> str:
    """ Default grouping function for images based on the first part of the filename before the first underscore."""
    try:
        if "_" in image_path.stem:
            return image_path.stem.split('_')[0]
        else:
            return image_path.stem
    except Exception as e:
        print(f"⚠️ Error processing filename {image_path.name}: {e}")
        return image_path.stem

def group_images_by_strategy(
    image_dir: Path,
    strategy: Literal['underscore', "custom"],
    custom_grouping_fn: Optional[Callable[[Path], str]] = None,
) -> dict[str, List[Path]]:
    """Group images by filename strategy."""
    groups = {}
    all_images = get_files(image_dir, extensions=['.jpg', '.png', '.jpeg'])
    for img_path in all_images:
        try:
            if strategy == 'underscore' :
                key = default_grouping_function(img_path)
            elif strategy == "custom" and custom_grouping_fn is not None:
                key = custom_grouping_fn(img_path)
            else:
                raise ValueError("⚠️ Invalid strategy or custom function not provided.")

            groups.setdefault(key, []).append(img_path)

        except Exception as e:
            print(f"⚠️ Error processing image {img_path.name}: {e}")
    return groups