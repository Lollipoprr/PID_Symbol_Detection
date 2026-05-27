from pathlib import Path
from typing import Dict, Any
import logging
from ultralytics import YOLO
import torch

class YOLOTrainer:
    def __init__(self, model: YOLO, data_yaml: Path, output_dir: Path):
        """
        Initialize the YOLO trainer.
        
        Args:
            model: YOLO model instance
            data_yaml: Path to dataset.yaml file
            output_dir: Directory to save training outputs
        """
        self.model = model
        self.data_yaml = data_yaml
        self.output_dir = output_dir
        self.logger = logging.getLogger(__name__)
        
    def train(self, epochs: int, batch_size: int, img_size: int, **kwargs) -> None:
        """
        Train the YOLO model.
        
        Args:
            epochs: Number of training epochs
            batch_size: Batch size for training
            img_size: Image size for training
            **kwargs: Additional training parameters
        """
        try:
            device = "3" if torch.cuda.is_available() else "cpu"
            # Train the model
            results = self.model.train(
                data=str(self.data_yaml),
                epochs=epochs,
                batch=batch_size,
                imgsz=(img_size[0], img_size[1]),
                device=device,
                multi_scale=False,
                project=str(self.output_dir),
                **kwargs
            )
            
            self.logger.info(f"Training completed successfully. Results saved to {self.output_dir}")
            return results
            
        except Exception as e:
            self.logger.error(f"Error during training: {str(e)}")
            raise
            
    def validate(self) -> Dict[str, Any]:
        """Validate the trained model."""
        try:
            results = self.model.val()
            self.logger.info("Validation completed successfully")
            return results
            
        except Exception as e:
            self.logger.error(f"Error during validation: {str(e)}")
            raise
            
    def export(self, format: str = "onnx") -> Path:
        """
        Export the trained model.
        
        Args:
            format: Export format (e.g., 'onnx', 'torchscript')
            
        Returns:
            Path to the exported model
        """
        try:
            export_path = self.model.export(format=format)
            self.logger.info(f"Model exported successfully to {export_path}")
            return Path(export_path)
            
        except Exception as e:
            self.logger.error(f"Error during model export: {str(e)}")
            raise 