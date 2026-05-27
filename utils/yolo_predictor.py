from ultralytics import YOLO
from pathlib import Path
import logging
from typing import List, Optional, Union, Tuple
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction, get_prediction, predict
from sahi.prediction import ObjectPrediction
from torchvision.ops import nms
import torch
import cv2
import numpy as np
from utils.helpers import resolve_image_paths
from utils.bbox_utils import coco_array_to_yolo_file

class BoundingBox:
    """封装边界框的核心属性和常用方法"""
    def __init__(self, xmin, ymin, xmax, ymax, score=None, category_id=None):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.score = score
        self.category_id = category_id

    def to_xyxy(self) -> List[float]:
        return [self.xmin, self.ymin, self.xmax, self.ymax]

    def get_area(self) -> float:
        return (self.xmax - self.xmin) * (self.ymax - self.ymin)

    def get_intersection_area(self, other_box: 'BoundingBox') -> float:
        xA = max(self.xmin, other_box.xmin)
        yA = max(self.ymin, other_box.ymin)
        xB = min(self.xmax, other_box.xmax)
        yB = min(self.ymax, other_box.ymax)
        if xA >= xB or yA >= yB:
            return 0
        return (xB - xA) * (yB - yA)

    @classmethod
    def from_sahi_prediction(cls, pred: ObjectPrediction) -> 'BoundingBox':
        bbox = pred.bbox
        return cls(
            xmin=bbox.minx,
            ymin=bbox.miny,
            xmax=bbox.maxx,
            ymax=bbox.maxy,
            score=pred.score.value,
            category_id=pred.category.id
        )


def calculate_steps(full_size: int, tile_size: int, overlap_ratio: float) -> List[int]:
    """计算切片在图像上的起始坐标列表"""
    step = int(tile_size * (1 - overlap_ratio))
    starts = []
    current = 0
    while current + tile_size <= full_size:
        starts.append(current)
        current += step
    if starts and (starts[-1] + tile_size < full_size):
        final_start = max(0, full_size - tile_size)
        if final_start not in starts:
            starts.append(final_start)
    return starts


def filter_subedge_boxes(boxes: List[BoundingBox], sub_size: Tuple[int, int], 
                         margin_ratio: float, area_threshold: float = 0.1) -> List[BoundingBox]:
    """过滤切片边缘区域的检测框，减少因切片导致的虚假检测"""
    sub_w, sub_h = sub_size
    margin_w = sub_w * margin_ratio
    margin_h = sub_h * margin_ratio
    
    center_box = BoundingBox(
        xmin=margin_w, ymin=margin_h,
        xmax=sub_w - margin_w, ymax=sub_h - margin_h
    )
    
    filtered_boxes = []
    for box in boxes:
        intersection_area = box.get_intersection_area(center_box)
        area_ratio = intersection_area / max(box.get_area(), 1e-6)
        if area_ratio >= area_threshold:
            filtered_boxes.append(box)
    return filtered_boxes


def merge_boxes_with_nms(boxes: List[BoundingBox], nms_threshold: float = 0.5) -> List[BoundingBox]:
    """NMS合并高度重叠的边界框"""
    if not boxes:
        return []
    bboxes = torch.tensor([box.to_xyxy() for box in boxes], dtype=torch.float32)
    scores = torch.tensor([box.score for box in boxes], dtype=torch.float32)
    keep_indices = nms(bboxes, scores, nms_threshold).cpu().numpy()
    return [boxes[i] for i in keep_indices]


class YOLOPredictor:
    def __init__(self, model_path: Path):
        """
        Initialize the YOLO predictor.

        Args:
            model_path: Path to the YOLO model
        """
        self.model_path = model_path
        self.logger = logging.getLogger(__name__)

    def perform_standard_inference(self):        
        pass

    def perform_sliced_inference(self, 
                                 src: Union[str, Path, List[str], List[Path]], 
                                 conf: float = 0.25,
                                 slice_height: int = 1024,
                                 slice_width: int = 1024,
                                 save_txt: bool = True,
                                 save_conf: bool = False,
                                 overlap_height_ratio: float = 0.5,
                                 overlap_width_ratio: float = 0.5,
                                 edge_margin_ratio: float = 0.075,
                                 nms_threshold: float = 0.5,
                                 output_dir: Optional[Path] = None,
                                 ) -> None:
        """
        执行带边缘过滤的切片推理。

        Args:
            src: 图像路径或路径列表
            conf: 置信度阈值
            slice_height: 切片高度
            slice_width: 切片宽度
            save_txt: 是否保存 YOLO 格式预测结果
            save_conf: 是否在 txt 中保存置信度
            overlap_height_ratio: 高度方向重叠比例
            overlap_width_ratio: 宽度方向重叠比例
            edge_margin_ratio: 边缘过滤比例（过滤切片边缘区域）
            nms_threshold: NMS IoU 阈值
            output_dir: 输出目录
        """
        from sahi.utils.cv import read_image

        image_paths = resolve_image_paths(src)
        import os
        device = os.environ.get('CUDA_VISIBLE_DEVICES', 'cuda:0' if torch.cuda.is_available() else 'cpu')

        detection_model = AutoDetectionModel.from_pretrained(
            model_type='ultralytics',
            model_path=self.model_path,
            confidence_threshold=conf,
            device=device)
        
        for image_path in image_paths:
            im = read_image(str(image_path))
            H, W, _ = im.shape
            all_predictions = []

            # 如果图像尺寸小于切片大小，直接整图推理
            if H < slice_height or W < slice_width:
                result = get_sliced_prediction(
                    im, detection_model,
                    slice_height=H, slice_width=W,
                    overlap_height_ratio=0.0, overlap_width_ratio=0.0,
                )
                sub_predictions = result.object_prediction_list
                custom_boxes = [BoundingBox.from_sahi_prediction(pred) for pred in sub_predictions]
                all_predictions = custom_boxes
            else:
                # 切片推理
                x_starts = calculate_steps(W, slice_width, overlap_width_ratio)
                y_starts = calculate_steps(H, slice_height, overlap_height_ratio)

                for y in y_starts:
                    for x in x_starts:
                        sub_w = min(slice_width, W - x)
                        sub_h = min(slice_height, H - y)
                        sub_im = im[y:y+sub_h, x:x+sub_w]

                        result = get_sliced_prediction(
                            sub_im, detection_model,
                            slice_height=sub_h, slice_width=sub_w,
                            overlap_height_ratio=overlap_height_ratio,
                            overlap_width_ratio=overlap_width_ratio,
                        )
                        sub_predictions = result.object_prediction_list

                        # 转换为自定义边界框
                        custom_boxes = [BoundingBox.from_sahi_prediction(pred) for pred in sub_predictions]
                        
                        # 边缘过滤
                        filtered_boxes = filter_subedge_boxes(custom_boxes, (sub_w, sub_h), edge_margin_ratio)
                        
                        # 坐标偏移到原图坐标系
                        for box in filtered_boxes:
                            box.xmin += x
                            box.xmax += x
                            box.ymin += y
                            box.ymax += y
                        
                        all_predictions.extend(filtered_boxes)

            # NMS 合并
            merged_predictions = merge_boxes_with_nms(all_predictions, nms_threshold)

            # 保存可视化结果
            if output_dir:
                vis_path = output_dir / f"{Path(image_path).stem}.png"
                vis_img = im.copy()
                for pred in merged_predictions:
                    x1, y1, x2, y2 = map(int, pred.to_xyxy())
                    cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{pred.category_id}:{pred.score:.2f}"
                    cv2.putText(vis_img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.imwrite(str(vis_path), vis_img)

            # 保存 YOLO 格式预测结果
            if save_txt and output_dir:
                output_path = output_dir / f"{Path(image_path).stem}.txt"
                with open(output_path, 'w') as f:
                    for pred in merged_predictions:
                        # 转换为 YOLO 格式: class x_center y_center width height (归一化)
                        x_center = (pred.xmin + pred.xmax) / 2 / W
                        y_center = (pred.ymin + pred.ymax) / 2 / H
                        width = (pred.xmax - pred.xmin) / W
                        height = (pred.ymax - pred.ymin) / H
                        if save_conf:
                            f.write(f"{pred.category_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} {pred.score:.4f}\n")
                        else:
                            f.write(f"{pred.category_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

