from pathlib import Path
import cv2
import albumentations as A
from typing import List, Tuple, Union, Dict
import logging
import numpy as np
from utils.helpers import get_files
from utils.bbox_utils import BBoxUtils


class PatchGenerator:
    def __init__(self, patch_size: Tuple[int, int], overlap: Tuple[int, int], min_visibility: float = 0.5):
        """
        Initialize the patch generator.
        
        Args:
            patch_size: Tuple of (height, width) for the patch size
            overlap: Overlap in pixels
        """
        self.patch_size = patch_size
        self.overlap = overlap
        self.min_visibility = min_visibility
        self.logger = logging.getLogger(__name__)
        
    def generate_patches(self, images_dir: Path, labels_dir: Path, output_dir: Path) -> None:
        """
        Split images and labels into patches.
        
        Args:
            images_dir: Directory containing input images
            labels_dir: Directory containing corresponding labels
            output_dir: Directory to save generated patches
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get all image files in images_dir
        image_files = get_files(images_dir, extensions=['.png'])
        if not image_files:
            self.logger.error(f"No images found in {images_dir}. Check the extensions and the directory.")
            return
        
        for image_path in image_files:
            image_name = image_path.stem
            try:
                image = cv2.imread(str(image_path))
                if image is None:
                    print(f"❌ Error: Failed to read image file: {image_path}")
                    return
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                height, width = image.shape[:2]
                patch_w, patch_h = self.patch_size
                patch_overlap_x, patch_overlap_y = self.overlap

                # Step size for sliding window
                step_x = patch_w - patch_overlap_x
                step_y = patch_h - patch_overlap_y

                # Read bounding boxes and class labels
                label_path = labels_dir / (image_path.stem + ".txt")
                cxywh_yolo = BBoxUtils().get_bboxes_array_from_file(filepath=label_path)
                if not cxywh_yolo:
                    print(f"⚠️ Warning: No bounding boxes found in {label_path}. Skipping.")
                    return

                class_labels, bboxes = [line[0] for line in cxywh_yolo], [
                    line[1:] for line in cxywh_yolo
                ]

                # Crop using sliding window
                for r, y in enumerate(range(0, height - patch_h + 1, step_y)):
                    for c, x in enumerate(range(0, width - patch_w + 1, step_x)):
                        try:
                            crop_transform = A.Compose(
                                [
                                    A.Crop(
                                        x_min=x,
                                        y_min=y,
                                        x_max=int(min(x + patch_w, width)),
                                        y_max=int(min(y + patch_h, height)),
                                    )
                                ],
                                bbox_params=A.BboxParams(
                                    format="yolo",
                                    min_visibility=self.min_visibility,
                                    label_fields=["class_labels"],
                                ),
                            )
                            transformed = crop_transform(
                                image=image, bboxes=bboxes, class_labels=class_labels
                            )

                            cropped_img = transformed["image"]
                            cropped_bboxes = transformed["bboxes"]
                            cropped_class_labels = transformed["class_labels"]

                            # Save image patch
                            patch_image_name = f"{image_name}_{r}_{c}.png"
                            patch_image_path = output_dir / patch_image_name
                            cv2.imwrite(
                                str(patch_image_path), cv2.cvtColor(cropped_img, cv2.COLOR_RGB2BGR)
                            )

                            # Save label file
                            patch_label_name = f"{image_name}_{r}_{c}.txt"
                            patch_label_path = output_dir / patch_label_name
                            with patch_label_path.open("w") as label_file:
                                for bbox, cls in zip(cropped_bboxes, cropped_class_labels):
                                    xmin, ymin, w, h = bbox
                                    label_file.write(f"{int(cls)} {xmin} {ymin} {w} {h}\n")

                        except Exception as patch_error:
                            print(
                                f"⚠️ Warning: Failed to process patch ({r}, {c}) for image {image_name}: {patch_error}"
                            )

                print(f"✅ Successfully extracted patches for image: {image_path.name}")
            
            except Exception as e:
                print(f"❌ Error in split_image_to_patches: {e}")



        

        
        