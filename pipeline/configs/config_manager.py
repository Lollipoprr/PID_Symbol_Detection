import yaml
from pathlib import Path
from typing import Dict, Any, Union, List, Optional
import logging

class ConfigManager:
    def __init__(self, config_path: Union[str, Path]):
        """
        Initializes the ConfigManager with a YAML configuration file.

        Args:
            config_path (Union[str, Path]): Path to the YAML configuration file.
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._create_directories()

    def _load_config(self) -> Dict[str, Any]:
        """
        Loads the YAML configuration file.

        Returns:
            Dict[str, Any]: The loaded configuration as a dictionary.
        """
        try: 
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)
            return config
        except FileNotFoundError:
            logging.error(f"❌ Configuration file not found: {self.config_path}")
            raise
        except yaml.YAMLError as e:
            logging.error(f"❌ Error parsing YAML file: {self.config_path}\n{e}")
            raise
    
    def _create_directories(self) -> None:
        """
        Creates necessary directories specified in the configuration.
        """
        paths = self.get_data_paths()
        model_paths = self.get_model_paths()

        # create data directories
        for path in paths.values():
            if not Path(path).exists():
                Path(path).mkdir(parents=True, exist_ok=True)
        
        # create model directories
        for path in model_paths.values():
            # check if the path is a directory
            if not Path(path).is_dir() and not Path(path).exists():
                Path(path).mkdir(parents=True, exist_ok=True)
    
    def _get_project_root(self) -> Path:
        """根据配置文件位置计算项目根目录。

        向上查找包含 src/ 子目录的路径作为项目根。
        这确保即使 src/pipeline/data 存在，也能正确找到真正的项目根目录。
        """
        cfg = self.config_path.resolve()
        for parent in [cfg] + list(cfg.parents):
            if (parent / "src").exists():
                return parent
        return cfg.parent

    def get_data_paths(self) -> Dict[str, Path]:
        """
        Retrieves data paths from the configuration.

        Returns:
            Dict[str, Path]: A dictionary of data paths.
        """
        root_dir = self._get_project_root() / self.config['data']['root_dir']
        return {
            'raw_images_dir': root_dir / self.config['data']['raw']['raw_images_dir'],
            'raw_class_aware_labels_dir': root_dir / self.config['data']['raw']['raw_class_aware_labels_dir'],
            'raw_class_agnostic_labels_dir': root_dir / self.config['data']['raw']['raw_class_agnostic_labels_dir'],

            # inference (separated from training/val data)
            'stage1_inference_images_dir': root_dir / self.config['data']['inference']['stage_1_images_dir'],
            'stage1_inference_labels_aware_dir': root_dir / self.config['data']['inference']['stage_1_labels_aware_dir'],
            'stage1_inference_labels_agnostic_dir': root_dir / self.config['data']['inference']['stage_1_labels_agnostic_dir'],
            'stage2_inference_images_dir': root_dir / self.config['data']['inference']['stage_2_images_dir'],
            'stage2_inference_labels_aware_dir': root_dir / self.config['data']['inference']['stage_2_labels_aware_dir'],

            'class_agnostic_patches_dir': root_dir / self.config['data']['processed']['stage_1']['class_agnostic_patches_dir'],
            'class_aware_patches_dir': root_dir / self.config['data']['processed']['stage_1']['class_aware_patches_dir'],
            'class_agnostic_yolo_train_dir': root_dir / self.config['data']['processed']['stage_1']['class_agnostic_yolo_train_dir'],
            'class_aware_yolo_train_dir': root_dir / self.config['data']['processed']['stage_1']['class_aware_yolo_train_dir'],
            'symbol_crops_dir': root_dir / self.config['data']['processed']['stage_2']['symbol_crops_dir'],
            'train_dir': root_dir / self.config['data']['processed']['stage_2']['train_dir'],
            'val_dir': root_dir / self.config['data']['processed']['stage_2']['val_dir'],
            'class_agnostic_results_dir': self._get_project_root() / self.config['data']['results']['stage_1']['class_agnostic_results_dir'],
            'class_aware_results_dir': self._get_project_root() / self.config['data']['results']['stage_1']['class_aware_results_dir'],
            'stage2_inference_results_dir': self._get_project_root() / self.config['data']['results']['stage_2']['inference_results_dir'],

        }
    
    def get_model_paths(self) -> Dict[str, Path]:
        """
        Retrieves model paths from the configuration.

        Returns:
            Dict[str, Path]: A dictionary of model paths.
        """
        root_dir = self._get_project_root() / self.config['models']['root_dir']
        return {
            'base_yolo_path': root_dir / self.config['models']['base_yolo_path'],
            'stage1_class_agnostic_weights_dir': root_dir / self.config['models']['stage1_class_agnostic_weights_dir'],
            'stage1_class_aware_weights_dir': root_dir / self.config['models']['stage1_class_aware_weights_dir'],
            'stage2_weights_dir': root_dir / self.config['models']['stage2_weights_dir']
        }

    def get_training_config(self) -> Dict[str, Any]:
        """获取训练配置。
        
        合并通用配置和训练特有配置。
        """
        common = self.get_common_config()
        training_specific = self.config.get('training', {})
        return {**common, **training_specific}
    
    def get_common_config(self) -> Dict[str, Any]:
        """获取通用配置（训练和推理共用）。
        
        Returns:
            Dict 包含: slice_size, slice_overlap_ratio, slice_edge_margin_ratio,
                      nms_iou_threshold, confidence_threshold, 
                      num_classes_agnostic, num_classes_aware, 
                      class_agnostic_class_names, class_aware_class_names
        """
        return {
            'slice_size': self.config.get('slice_size', 1024),
            'slice_overlap_ratio': self.config.get('slice_overlap_ratio', 0.5),
            'slice_edge_margin_ratio': self.config.get('slice_edge_margin_ratio', 0.075),
            'nms_iou_threshold': self.config.get('nms_iou_threshold', 0.5),
            'confidence_threshold': self.config.get('confidence_threshold', 0.25),
            'num_classes_agnostic': self.config.get('num_classes_agnostic', 1),
            'num_classes_aware': self.config.get('num_classes_aware', 63),
            'class_agnostic_class_names': self.config.get('class_agnostic_class_names', ["Symbol"]),
            'class_aware_class_names': self.config.get('class_aware_class_names', []),
            # 数据预处理配置
            'min_visibility': self.config.get('data_preprocessing', {}).get('min_visibility', 0.3),
            'train_val_test_split': self.config.get('data_preprocessing', {}).get('train_val_test_split', [0.7, 0.2, 0.1]),
        }
    
    def get_symbol_detection_config(self) -> Dict[str, Any]:
        """获取 symbol detection 配置（向后兼容）。
        
        .. deprecated::
            请使用 get_common_config() 替代。
        """
        return self.get_common_config()
    
    def get_few_shot_config(self) -> Dict[str, Any]:
        """.. deprecated:: Use get_stage2_baseline_config() instead."""
        return self.config.get("stage2_baseline", {})
    
    def create_directories(self) -> None:
        """Create all necessary directories."""
        paths = self.get_data_paths()
        for path in paths.values():
            path.mkdir(parents=True, exist_ok=True) 
        
        model_paths = self.get_model_paths()
        for path in model_paths.values():
            path.mkdir(parents=True, exist_ok=True)
        logging.info("✅ All necessary directories have been created.")

    def get_stage1_inference_config(self) -> Dict[str, Any]:
        """获取 Stage 1 推理配置。
        
        合并通用配置和推理特有配置。
        """
        common = self.get_common_config()
        inference_specific = self.config.get('stage1_inference', {})
        return {**common, **inference_specific}
    
    def get_stage2_baseline_config(self) -> Dict[str, Any]:
        """Get stage 2 baseline training configuration (prototype-based)."""
        return self.config.get("stage2_baseline", {})

    def get_stage2_svg_config(self) -> Dict[str, Any]:
        """Get SVG-Guided Structure Enhancement training configuration."""
        return self.config.get("stage2_svg_structure", {})

    def get_stage2_expA_config(self) -> Dict[str, Any]:
        """Get Stage 2 Exp-A configuration (ProtoBank + DomainAdapt + L_domain)."""
        return self.config.get("stage2_expA", {})

    def get_stage2_inference_config(self) -> Dict[str, Any]:
        """Get stage 2 inference configuration (prototype-based)."""
        return self.config['stage2_inference']

    def get_evaluation_metrics_config(self) -> Dict[str, Any]:
        """Get evaluation metrics configuration (open-set + embedding quality)."""
        return self.config.get("evaluation_metrics", {})
