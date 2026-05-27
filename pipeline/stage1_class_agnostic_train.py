from pipeline.base import BasePipeline
from pathlib import Path
from ultralytics import YOLO
from typing import Dict, Any
from utils.patch_generator import PatchGenerator
from utils.yolo_dataset_builder import YOLODatasetBuilder
from utils.yolo_train_utils import YOLOTrainer
from utils.bbox_utils import BBoxUtils
from utils.helpers import get_text_files, get_image_files, get_files

class Stage1ClassAgnosticPipeline(BasePipeline):
    def __init__(self, config_path: str = "configs/config.yaml"):
        super().__init__(config_path)
        self.patch_generator = None
        self.dataset_builder = None
        self.bbox_utils = BBoxUtils()
        
    def validate(self) -> bool:
        """Validate stage 1 pipeline inputs.

        Returns:
            bool: True if validation passes, False otherwise
        """
        paths = self.get_data_paths()
        
        # Check if raw data directories exist
        if not paths['raw_images_dir'].exists():
            raise FileNotFoundError("Raw image directory not found")
        
        if not paths['raw_class_aware_labels_dir'].exists():
            raise FileNotFoundError("Raw class-aware labels directory not found")
            
        # Check if base YOLO model exists
        if not self.get_model_paths()['base_yolo_path'].exists():
            raise FileNotFoundError("Base YOLO model not found")
            
        return True
    
    def prepare_data(self) -> None:
        """Prepare data for class-agnostic detection."""
        config = self.get_symbol_detection_config()
        paths = self.get_data_paths()

        # Create & Save Class-Agnostic Labels from Raw Class-Aware Labels
        for label_filepath in get_text_files(paths['raw_class_aware_labels_dir']):
            self.bbox_utils.convert_bboxes_file_to_class_agnostic(label_filepath, paths['raw_class_agnostic_labels_dir'])
        
        # Initialize patch generator (使用统一配置)
        self.patch_generator = PatchGenerator(
            patch_size=config['slice_size'],  # 统一使用 slice_size
            overlap=int(config['slice_size'] * config['slice_overlap_ratio']),  # 转换为像素值
            min_visibility=config['min_visibility']
        )
        
        # Generate patches
        self.patch_generator.generate_patches(
            images_dir=paths['raw_images_dir'],
            labels_dir=paths['raw_class_agnostic_labels_dir'],
            output_dir=paths['class_agnostic_patches_dir']
        )
        
        # # Initialize dataset builder
        self.dataset_builder = YOLODatasetBuilder(
            images_dir=paths['class_agnostic_patches_dir'],
            labels_dir=paths['class_agnostic_patches_dir'],
            output_dir=paths['class_agnostic_yolo_train_dir']
        )
        
        # Split data and create YOLO dataset
        self.dataset_builder.create_dataset(
            train_val_test_split=self.get_training_config()['train_val_test_split'],
            class_names = config["class_agnostic_class_names"],
            grouping_strategy="underscore",
        )
    
    def train_model(self) -> None:
        """Train the class-agnostic detection model."""
        config = self.get_training_config()
        model_paths = self.get_model_paths()

        import os
        from pathlib import Path
        import torch.hub
        import traceback

        # 调试：检查模型路径
        w = Path(model_paths["base_yolo_path"])
        print(f"[DEBUG] cwd={os.getcwd()}")
        print(f"[DEBUG] base_yolo_path(str)={model_paths['base_yolo_path']}")
        print(f"[DEBUG] base_yolo_path(resolve)={w.resolve()}")
        print(f"[DEBUG] exists={w.exists()} size={(w.stat().st_size if w.exists() else None)}")
        
        # 调试：捕获下载行为
        _old_download = torch.hub.download_url_to_file

        def _debug_download(url, dst, hash_prefix=None, progress=True):
            print("\n[DEBUG] torch.hub.download_url_to_file 被调用")
            print(f"[DEBUG] url={url}")
            print(f"[DEBUG] dst={dst}")
            print("[DEBUG] 调用栈(最关键)：")
            traceback.print_stack(limit=30)
            return _old_download(url, dst, hash_prefix=hash_prefix, progress=progress)

        torch.hub.download_url_to_file = _debug_download
        
        # Load base model
        model = YOLO(model_paths['base_yolo_path'])
        
        # Train model
        trainer = YOLOTrainer(
            model=model,
            data_yaml=Path(self.get_data_paths()['class_agnostic_yolo_train_dir']) / 'dataset.yaml',
            output_dir=model_paths['stage1_class_agnostic_weights_dir']
        )

        trainer.train(
            epochs=config['epochs'],
            batch_size=config['batch_size'],
            img_size=config['image_size'],
        )


    def run(self) -> None:
        """Run the complete stage 1 pipeline."""
        if not self.validate():
            raise ValueError("Pipeline validation failed")
            
        print("Preparing data for class-agnostic detection...")
        self.prepare_data()
        
        print("Training class-agnostic detection model...")
        self.train_model()
        
        print("Stage 1 pipeline completed successfully") 