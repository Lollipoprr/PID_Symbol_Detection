from pipeline.base import BasePipeline
from typing import Union, Dict, Any, List, Optional, Tuple
from pathlib import Path
import logging
import numpy as np
from utils.embedding_classifier import EmbeddingClassifier
from utils.helpers import get_files, resolve_image_paths
import torch
import torchvision.transforms as v2
import cv2

class Stage2InferencePipeline(BasePipeline):
    def __init__(self, config_path: str = "configs/config.yaml"):
        """Initialize stage 2 inference pipeline.

        Args:
            config_path: Path to configuration file
        """
        super().__init__(config_path)
        self.classifier = None
        self.logger = logging.getLogger(__name__)

    def validate(self) -> bool:
        """Validate stage 2 inference pipeline inputs.

        Returns:
            bool: True if validation passes, False otherwise
        """
        paths = self.get_data_paths()
        model_paths = self.get_model_paths()

        # Check if stage 1 model dir exists
        if not model_paths['stage1_class_agnostic_weights_dir'].exists():
            raise FileNotFoundError("Stage 1 model directory not found")

        # Check if inference image directory exists
        if not paths['stage2_inference_images_dir'].exists():
            raise FileNotFoundError(
                f"Inference images directory not found: {paths['stage2_inference_images_dir']}"
            )
        if not paths['raw_class_aware_labels_dir'].exists():
            raise FileNotFoundError(
                f"Labels directory not found: {paths['raw_class_aware_labels_dir']}"
            )
        
        # Check if model directory exists (.pth for baseline, .pt for newer variants)
        if not len(get_files(model_paths['stage2_weights_dir'], [".pt", ".pth"])) > 0:
            raise FileNotFoundError("Model directory not found or empty")
        return True
    
    def run(self, save_detailed: bool = True,
             rejection_mode: Optional[str] = None,
             distance_threshold_override: Optional[float] = None) -> None:
        """Run stage 2 inference pipeline.

        Args:
            save_detailed: 是否保存详细结果（JSON/CSV格式）
        """
        self.validate()
        self.config = self.get_stage2_inference_config()
        model_paths = self.get_model_paths()
        paths = self.get_data_paths()

        # ── 模型路径选择 ───────────────────────────────────────────────
        # 优先使用新架构 best_disentangle_model.pt，回退到 best_acc_model.pt 或 best_model.pt
        candidate_paths = [
            model_paths['stage2_weights_dir'] / 'best_disentangle_model.pt',
            model_paths['stage2_weights_dir'] / 'best_disentangle_model.pth',
            model_paths['stage2_weights_dir'] / 'best_acc_model.pt',
            model_paths['stage2_weights_dir'] / 'best_acc_model.pth',
            model_paths['stage2_weights_dir'] / 'best_model.pt',
            model_paths['stage2_weights_dir'] / 'best_model.pth',
        ]

        self.model_path = next((p for p in candidate_paths if p.exists()), None)
        if self.model_path is not None:
            self.logger.info(f"[推理] 使用模型: {self.model_path}")
        else:
            raise FileNotFoundError(
                f"[推理] 未找到任何模型，候选路径: {', '.join(str(p) for p in candidate_paths)}"
            )

        apn_transform = v2.Compose([
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # 初始化 EmbeddingClassifier
        self.classifier = EmbeddingClassifier(
            self.model_path,
            transform=apn_transform,
        )

        # ── 原型构建模式 ────────────────────────────────────────────────
        train_dir = paths['train_dir']
        self.logger.info(f"[原型] 训练数据目录: {train_dir}")
        use_weighted_prototypes = self.config.get('prototype_weighting', True)

        # 原型构建模式（weighted / simple）
        self.logger.info(
            f"[原型模式] prototype_weighting={use_weighted_prototypes} "
            f"({'加权原型（1/sqrt(N)+自适应温度）' if use_weighted_prototypes else '简单平均原型（无加权）'})"
        )
        if use_weighted_prototypes:
            self.logger.info("Computing weighted class prototypes (1/sqrt(N) + adaptive temperature)")
            self.class_prototypes = self.classifier.compute_weighted_class_prototypes(
                train_dir,
                base_temperature=0.1,
                use_adaptive_temp=True,
            )
            self.logger.info("Weighted class prototypes computed successfully")
        else:
            self.logger.info("Computing simple class prototypes (arithmetic mean, no weighting)")
            self.class_prototypes = self.classifier.compute_simple_class_prototypes(
                train_dir
            )
            self.logger.info("Simple class prototypes computed successfully")

        # ── 拒识配置 ──────────────────────────────────────────────────────
        rejection_mode = rejection_mode or self.config.get('rejection_mode', 'distance_only')
        rejection_threshold = self.config.get('distance_threshold', 0.25)
        if distance_threshold_override is not None:
            rejection_threshold = distance_threshold_override
        self.logger.info(f"拒识策略: mode={rejection_mode}, threshold={rejection_threshold}")

        # Perform inference
        self.logger.info("Performing inference")
        self.classifier.predict_class_from_prototypes(
            img_src=paths['stage2_inference_images_dir'],
            labels_dir=paths['class_agnostic_results_dir'],
            class_prototypes_dict=self.class_prototypes,
            output_dir=paths['stage2_inference_results_dir'],
            distance_metric=self.config.get('distance_metric', 'euclidean'),
            rejection_mode=rejection_mode,
            rejection_threshold=rejection_threshold,
            save_detailed_results=save_detailed,
        )
        
        self.logger.info("Stage 2 inference completed successfully.")
        self.logger.info(f"Results saved to {paths['stage2_inference_results_dir']}")
        
        # Visualize results
        self.logger.info("Generating visualization...")
        self._visualize_results(
            img_dir=Path(paths['stage2_inference_images_dir']),
            label_dir=Path(paths['stage2_inference_results_dir']),
            output_dir=Path('results/stage_2/view')
        )
        self.logger.info(f"Visualization saved to results/stage_2/view")
        
    def _build_class_color_map(self, label_files: List[Path]) -> Dict[int, Tuple[int, int, int]]:
        """Build a consistent color map for all class IDs seen across label files.

        Colors are generated using HSV hue wheel spacing so that adjacent
        class IDs remain visually distinct.

        Args:
            label_files: List of label .txt file paths

        Returns:
            Dict mapping class_id -> (B, G, R) tuple for cv2
        """
        seen_ids = set()
        for lf in label_files:
            with open(lf) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        seen_ids.add(int(parts[0]))

        # Sort for deterministic ordering
        sorted_ids = sorted(seen_ids)
        n = max(len(sorted_ids), 1)

        # Hue evenly spaced on [0, 180) – HSV hue range used by OpenCV
        hues = np.linspace(0, 180, n, endpoint=False).astype(np.uint8)
        colors = {}
        for class_id, hue in zip(sorted_ids, hues):
            # Saturation and Value set high for vivid colors
            sat, val = 220, 255
            bgr = cv2.cvtColor(
                np.uint8([[[hue, sat, val]]]), cv2.COLOR_HSV2BGR
            )[0, 0]
            colors[class_id] = tuple(int(v) for v in bgr)

        self.logger.info(f"Built color map for {len(colors)} classes")
        return colors

    def _add_legend(
        self,
        image: np.ndarray,
        color_map: Dict[int, Tuple[int, int, int]],
        font: int = cv2.FONT_HERSHEY_SIMPLEX,
        font_scale: float = 0.55,
        thickness: int = 1,
        margin: int = 10,
        sample_height: int = 18,
        line_gap: int = 4,
    ) -> None:
        """Draw a color legend in the top-left corner of the image.

        Args:
            image: Image array (modified in place)
            color_map: Dict class_id -> (B, G, R)
            font, font_scale, thickness: text rendering parameters
            margin: distance from image border
            sample_height: height of the colored square beside each label
            line_gap: vertical spacing between legend rows
        """
        if not color_map:
            return

        sorted_items = sorted(color_map.items())
        num_items = len(sorted_items)

        # Compute legend dimensions
        dummy_img = np.zeros((sample_height, 1, 3), dtype=np.uint8)
        (text_w, text_h), _ = cv2.getTextSize(
            "0", font, font_scale, thickness
        )
        row_h = sample_height
        box_w = sample_height
        legend_w = margin + box_w + 8 + text_w + margin
        legend_h = margin + num_items * (row_h + line_gap) + margin

        H, W = image.shape[:2]
        legend_h = min(legend_h, H - margin)
        legend_w = min(legend_w, W - margin)

        # Semi-transparent overlay background
        overlay = image.copy()
        cv2.rectangle(overlay, (margin, margin),
                      (margin + legend_w, margin + legend_h),
                      (40, 40, 40), -1)
        cv2.addWeighted(overlay, 0.75, image, 0.25, 0, image)

        # Draw border
        cv2.rectangle(image, (margin, margin),
                      (margin + legend_w, margin + legend_h),
                      (180, 180, 180), 1)

        y = margin + margin
        for class_id, color in sorted_items:
            label = f"Class {class_id}"
            cx = margin + margin
            cy = y + sample_height // 2 + text_h // 2 - 2
            # Colored square
            cv2.rectangle(image,
                          (cx, y),
                          (cx + box_w, y + sample_height),
                          color, -1)
            # White border on square
            cv2.rectangle(image,
                          (cx, y),
                          (cx + box_w, y + sample_height),
                          (255, 255, 255), 1)
            # Text label
            cv2.putText(image, label,
                        (cx + box_w + 8, cy),
                        font, font_scale, (255, 255, 255), thickness)
            y += row_h + line_gap

    def _visualize_results(self, img_dir: Path, label_dir: Path, output_dir: Path) -> None:
        """Visualize inference results by drawing bounding boxes on images.

        Each class gets a unique color drawn from the HSV hue wheel.
        A legend overlay is added to the top-left of each image.

        Args:
            img_dir: Directory containing inference images
            label_dir: Directory containing predicted labels
            output_dir: Directory to save visualized images
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get all label files
        label_files = sorted(label_dir.glob('*.txt'))
        if not label_files:
            self.logger.warning(f"No label files found in {label_dir}")
            return

        # Build a single global color map so the same class always gets
        # the same color across all images in this run.
        color_map = self._build_class_color_map(label_files)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 2

        for label_file in label_files:
            img_name = label_file.stem + '.png'
            img_path = img_dir / img_name

            # Try .png first, then .jpg
            if not img_path.exists():
                img_path = img_dir / (label_file.stem + '.jpg')

            if not img_path.exists():
                self.logger.warning(f"Image not found: {img_path}")
                continue

            image = cv2.imread(str(img_path))
            if image is None:
                continue

            height, width, _ = image.shape

            # Read labels
            with open(label_file, 'r') as f:
                lines = f.readlines()

            # Draw each bounding box with class-specific color
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue

                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                box_width = float(parts[3])
                box_height = float(parts[4])

                # Convert normalized coords to pixel coords
                x1 = int((x_center - box_width / 2) * width)
                y1 = int((y_center - box_height / 2) * height)
                x2 = int((x_center + box_width / 2) * width)
                y2 = int((y_center + box_height / 2) * height)

                box_color = color_map.get(class_id, (0, 255, 0))

                # Draw rectangle with class color
                cv2.rectangle(image, (x1, y1), (x2, y2), box_color, 2)

                # Label text background
                label_text = str(class_id)
                (text_w, text_h), _ = cv2.getTextSize(
                    label_text, font, font_scale, font_thickness
                )
                cv2.rectangle(
                    image,
                    (x1, y1 - text_h - 10),
                    (x1 + text_w + 10, y1),
                    (0, 0, 0), -1
                )

                # Draw class label text in class color
                cv2.putText(
                    image, label_text,
                    (x1 + 5, y1 - 5),
                    font, font_scale, box_color, font_thickness
                )

            # Overlay color legend
            self._add_legend(image, color_map)

            output_path = output_dir / img_name
            cv2.imwrite(str(output_path), image)

        self.logger.info(f"Visualized {len(label_files)} images with per-class colors")
