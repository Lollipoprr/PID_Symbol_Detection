import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union
import logging
from ultralytics import YOLO
import os
from utils.helpers import get_files, get_image_files, get_text_files

class BBoxUtils:
    logger = logging.getLogger(__name__)
    def __init__(self):
        """Initialize the bounding box utilities."""
        pass
        
    
    @staticmethod
    def convert_pascalvoc_array_to_yolo(
        bboxes: List[List[Union[int, float]]],
        image_width: int,
        image_height: int
    ) -> List[List[float]]:
        """
        Converts Pascal VOC bounding boxes to YOLO format.

        Input format:
            - [xmin, ymin, xmax, ymax]
            - [class_id, xmin, ymin, xmax, ymax]
            - [class_id, conf, xmin, ymin, xmax, ymax]

        Output format:
            - [x_center, y_center, width, height]
            - [class_id, x_center, y_center, width, height]
            - [class_id, conf, x_center, y_center, width, height]

        Args:
            bboxes (List[List]): Pascal VOC-style boxes.
            image_width (int): Width of the image.
            image_height (int): Height of the image.

        Returns:
            List[List[float]]: YOLO-style normalized bounding boxes.
        """
        yolo_bboxes = []
        warned_no_class = False
        warned_conf = False

        for box in bboxes:
            try:
                if len(box) == 4:
                    xmin, ymin, xmax, ymax = box
                    if not warned_no_class:
                        BBoxUtils.logger.info("⚠️ No class information provided. Output will be [x_center, y_center, width, height].")
                        warned_no_class = True
                    output_format = "xywh"
                elif len(box) == 5:
                    class_id, xmin, ymin, xmax, ymax = box
                    output_format = "class_xywh"
                elif len(box) == 6:
                    class_id, conf, xmin, ymin, xmax, ymax = box
                    if not warned_conf:
                        BBoxUtils.logger.info("ℹ️ Confidence scores detected. Output will be [class_id, x_center, y_center, width, height, conf].")
                        warned_conf = True
                    output_format = "class_xywh_conf"
                else:
                    BBoxUtils.logger.warning(f"⚠️ Skipping invalid box format: {box}")
                    continue

                x_center = ((xmax + xmin) / 2) / image_width
                y_center = ((ymax + ymin) / 2) / image_height
                width = (xmax - xmin) / image_width
                height = (ymax - ymin) / image_height

                if output_format == "xywh":
                    yolo_bboxes.append([round(x_center, 6), round(y_center, 6), round(width, 6), round(height, 6)])
                elif output_format == "class_xywh":
                    yolo_bboxes.append([int(class_id), round(x_center, 6), round(y_center, 6), round(width, 6), round(height, 6)])
                elif output_format == "class_xywh_conf":
                    yolo_bboxes.append([int(class_id), round(x_center, 6), round(y_center, 6), round(width, 6), round(height, 6), round(conf, 4)])
            except Exception as e:
                BBoxUtils.logger.error(f"❌ Error converting box {box}: {e}")
                continue

        return yolo_bboxes
    
    @staticmethod
    def save_bboxes_array_to_file(bboxes: List[List[float]], output_filepath: Union[str, Path]) -> bool:
        """
        Saves a list of bounding boxes to a text file, with each box on a new line
        and values separated by spaces.

        Args:
            bboxes (List[List[float]]): A list of bounding boxes. Each inner list
                                         should contain float values representing
                                         the bounding box coordinates (and optionally
                                         a class ID as the first element).
            output_filepath (str): The full path to the text file where the boxes
                                   will be saved.

        Returns:
            bool: True if the boxes were successfully saved to the file, False otherwise
                  (e.g., if an error occurred during file writing).
        """
        try:
            os.makedirs(os.path.dirname(str(output_filepath)), exist_ok=True)
            with open(output_filepath, 'w') as outfile:
                for bbox in bboxes:
                    line = ' '.join(map(str, bbox)) + '\n'
                    outfile.write(line)
            return True
        except Exception as e:
            BBoxUtils.logger.error(f"Error saving bounding boxes to '{output_filepath}': {e}")
            return False
        
    @staticmethod
    def get_bboxes_array_from_file(filepath: Union[Path, str]) -> List[List[float]]:
        """
        Extracts bounding boxes from a text file where each line represents a box.
        Assumes the first value in each line is an integer class ID, and subsequent
        values are float coordinates.

        Args:
            filepath (Path): The path to the text file containing bounding box information.

        Returns:
            list: A list of lists, where each inner list represents a bounding box.
                  The first element of each inner list is the integer class ID,
                  and the remaining elements are float coordinates. Returns an empty
                  list if the file is not found or if lines cannot be parsed correctly.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            BBoxUtils.logger.error(f"❌ Error: File not found: {filepath}")
            return []

        boxes = []
        try:
            with filepath.open('r') as f:
                for line in f:
                    line = line.strip()
                    if line:  # Ignore empty lines
                        parts = line.split()
                        if len(parts) >= 2:  # Need at least class ID and one coordinate
                            try:
                                class_id = int(parts[0])
                                coordinates = [float(p) for p in parts[1:]]
                                boxes.append([class_id] + coordinates)
                            except ValueError:
                                BBoxUtils.logger.warning(f"⚠️ Warning: Could not parse line in '{filepath}': {line}. Skipping.")
                        else:
                            BBoxUtils.logger.error(f"⚠️ Warning: Insufficient data in line '{line}' in '{filepath}'. Skipping.")
        except Exception as e:
            BBoxUtils.logger.error(f"❌ Error reading file '{filepath}': {e}")
            return []

        return boxes
    
    @staticmethod
    def _convert_line_to_class_agnostic(bbox_line: str) -> Optional[str]:
        """
        Converts a single class-aware bounding box annotation line to a class-agnostic line.

        Args:
            bbox_line (str): A string representing a single bounding box annotation
                             where the first element is the class ID, followed by
                             other bounding box coordinates (e.g., 'class_id x y w h'
                             for YOLO or COCO, or 'class_id xmin ymin xmax ymax' for Pascal VOC).

        Returns:
            Optional[str]: A string representing a class-agnostic bounding box annotation
                           where the first element (class ID) is replaced by '0', and the
                           rest of the original elements are maintained.
                           Returns None if the input line is invalid (less than 2 parts).
        """
        parts = bbox_line.strip().split()
        if len(parts) < 2:
            BBoxUtils.logger.warning(f"Invalid bounding box annotation format: {bbox_line}")
            return None

        try:
            # Attempt to convert the first part to an integer (class ID)
            int(parts[0]) # This will raise ValueError if conversion fails
            agnostic_parts = ['0'] + parts[1:]
            return " ".join(agnostic_parts)
        except ValueError:
            BBoxUtils.logger.warning(f"Could not parse class ID in annotation: {bbox_line}")
            return None
        
    @staticmethod
    def convert_bboxes_file_to_class_agnostic(input_file_path: Union[str, os.PathLike], output_folder: str) -> bool:
        """
        Reads a text file containing class-aware bounding box labels, converts them
        to class-agnostic format, and saves the converted labels in a new text file
        within the specified output folder. The output file will have the same name
        as the original input file.

        Args:
            input_file_path (Union[str, os.PathLike]): Path to the input text file containing class-aware labels.
            output_folder (str): Name of the folder where the class-agnostic
                                   label files will be saved.

        Returns:
            bool: True if the conversion was successful, False otherwise (e.g., if
                  the input file doesn't exist or an error occurs).
        """
        input_file_path = Path(input_file_path)
        output_folder = Path(output_folder)

        if not input_file_path.exists():
            BBoxUtils.logger.error(f"❌ Error: Input file not found: {input_file_path}")
            return False

        try:
            output_folder.mkdir(parents=True, exist_ok=True)  # makes dir if not exist
            output_file_path = output_folder / input_file_path.name

            with input_file_path.open('r') as infile, output_file_path.open('w') as outfile:
                for line in infile:
                    agnostic_line = BBoxUtils()._convert_line_to_class_agnostic(line.strip())
                    if agnostic_line:
                        outfile.write(agnostic_line + '\n')
            BBoxUtils.logger.info(f"✅ saved class-agnostic labels for '{input_file_path.name}' at '{output_folder}' folder")
            return True
        except Exception as e:
            BBoxUtils.logger.error(f"An error occurred during conversion of '{input_file_path}': {e}")
            return False
        
    @staticmethod
    def convert_bboxes_array_to_class_agnostic(bboxes: List[List[float]]) -> List[List[float]]:
        """
        Converts a list of bounding boxes (where the first element is the class ID)
        to a class-agnostic format by setting the class ID to 0.

        Args:
            bboxes (List[List[float]]): A list of bounding boxes. Each inner list
                                         should have the class ID as the first element
                                         and the coordinates as subsequent elements.

        Returns:
            List[List[float]]: A new list of bounding boxes where the class ID
                               of each box has been set to 0.
        """
        agnostic_bboxes = []
        for bbox in bboxes:
            if bbox:
                # 提取 class ID（索引0），其余为坐标
                class_id = bbox[0]
                coords = bbox[1:]
                agnostic_bbox = [0] + [float(val) for val in coords]
                agnostic_bboxes.append(agnostic_bbox)
        return agnostic_bboxes

    @staticmethod
    def save_crops_by_class(
        image_path: Union[str, Path],
        label_path: Union[str, Path],
        output_dir: Union[str, Path])-> None:

        """
        Saves cropped regions from an image into folders by class, based on YOLO label file.

        Args:
            image_path (Union[str, Path]): Path to input image.
            label_path (Union[str, Path]): Path to YOLO-format label file.
            output_dir (Union[str, Path]): Root directory to save cropped images.

        Returns:
            None
        """
        image_path = Path(image_path)
        label_path = Path(label_path)
        output_dir = Path(output_dir)

        # Create the output directory if it doesn't exist
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        image = cv2.imread(str(image_path))
        if image is None:
            BBoxUtils.logger.error(f"❌ Failed to load image: {image_path}")
            return

        height, width, _ = image.shape

        try:
            bboxes = BBoxUtils.get_bboxes_array_from_file(label_path)
            if bboxes and len(bboxes[0])!=5:
                raise ValueError(f"Expected 5 coordinates, got {len(bboxes[0])}")
            for i, bbox in enumerate(bboxes):
                class_id, x_c, y_c, w, h = bbox
                # check all are normalized values and not greater than 1 and less than 0
                if not (0 <= x_c <= 1 and 0 <= y_c <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                    BBoxUtils.logger.error(f"⚠️ Invalid normalized coordinates in {label_path.name} at line {i+1}  (ONLY YOLO FORMAT SUPPORTED): {bbox}")
                    continue

                class_id = int(class_id)
                # Convert to pixel coordinates
                x1 = int((x_c - w / 2) * width)
                y1 = int((y_c - h / 2) * height)
                x2 = int((x_c + w / 2) * width)
                y2 = int((y_c + h / 2) * height)

                # Ensure coordinates are within image bounds
                x1 = max(0, min(x1, width - 1))
                y1 = max(0, min(y1, height - 1))
                x2 = max(0, min(x2, width - 1))
                y2 = max(0, min(y2, height - 1))

                # Crop the image using the coordinates
                crop = BBoxUtils.crop_image(image, x1, y1, x2, y2)
                if crop.size == 0:
                    BBoxUtils.logger.error(f"⚠️ Empty crop at line {i+1} in {label_path.name}")
                    continue
                # Create the class directory if it doesn't exist
                class_dir = output_dir / str(class_id)
                class_dir.mkdir(parents=True, exist_ok=True)

                crop_filename = f"{image_path.stem}_{class_id}_{i}.png"
                crop_path = class_dir / crop_filename
                cv2.imwrite(str(crop_path), crop)

            BBoxUtils.logger.info(f"✅ Crops saved for image: {image_path.name}")
        except Exception as e:
            BBoxUtils.logger.error(f"❌ Error processing {image_path.name} or {label_path.name}: {e}")

    @staticmethod
    def crop_image(image: np.ndarray, x1: float, y1: float, x2: float, y2: float) -> np.ndarray:
        """
        Crop the image using the specified coordinates.
        """
        return image[y1:y2, x1:x2]
    
    @staticmethod
    def extract_bbox_crops(image_dir: Path, label_dir: Path, 
                                  output_dir: Path) -> None:
        
        """
        Save bounding box crops by class.
        
        Args:
            image_dir: Directory containing images
            label_dir: Directory containing labels
            output_dir: Directory to save the crops
        """
        image_paths = get_image_files(image_dir)

        # create output_dir if it doesn't exist
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for image_path in image_paths:
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"❌ Failed to load image: {image_path}")
                return
            height, width, _ = image.shape

            try:
                label_path = label_dir / (image_path.stem + ".txt")
                if not label_path.exists():
                    print(f"❌ Label file not found for {image_path.name}: {label_path.name}")
                    continue
                bboxes = BBoxUtils.get_bboxes_array_from_file(label_path)
                if bboxes and len(bboxes[0])!=5:
                    raise ValueError(f"Expected 5 coordinates, got {len(bboxes[0])}")
                for i, bbox in enumerate(bboxes):
                    class_id, x_c, y_c, w, h = bbox
                    # check all are normalized values and not greater than 1 and less than 0
                    if not (0 <= x_c <= 1 and 0 <= y_c <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                        print(f"⚠️ Invalid normalized coordinates in {label_path.name} at line {i+1}  (ONLY YOLO FORMAT SUPPORTED): {bbox}")
                        continue
                    
                    class_id = int(class_id)
                    # Convert to pixel coordinates
                    x1 = int((x_c - w / 2) * width)
                    y1 = int((y_c - h / 2) * height)
                    x2 = int((x_c + w / 2) * width)
                    y2 = int((y_c + h / 2) * height)

                    # Ensure coordinates are within image bounds
                    x1 = max(0, min(x1, width - 1))
                    y1 = max(0, min(y1, height - 1))
                    x2 = max(0, min(x2, width - 1))
                    y2 = max(0, min(y2, height - 1))

                    # Crop the image using the coordinates
                    crop = BBoxUtils.crop_image(image, x1, y1, x2, y2)
                    if crop.size == 0:
                        print(f"⚠️ Empty crop at line {i+1} in {label_path.name}")
                        continue
                    # Create the class directory if it doesn't exist
                    class_dir = output_dir / str(class_id)
                    class_dir.mkdir(parents=True, exist_ok=True)

                    crop_filename = f"{image_path.stem}_{class_id}_{i}.png"
                    crop_path = class_dir / crop_filename
                    cv2.imwrite(str(crop_path), crop)

                print(f"✅ Crops saved for image: {image_path.name}")
            except Exception as e:
                print(f"❌ Error processing {image_path.name} or {label_path.name}: {e}")

    @staticmethod
    def convert_yolo_array_to_cxyxy(bboxes: List[List[float]], image_width: int, image_height: int) -> List[List[float]]:
        """
        Converts YOLO bounding boxes to (c, x1, y1, x2, y2) format.

        Args:
            bboxes (List[List[float]]): YOLO-style boxes.
            image_width (int): Width of the image.
            image_height (int): Height of the image.

        Returns:
            List[List[float]]: Converted bounding boxes in (c, x1, y1, x2, y2) format.
        """
        cxyxy_bboxes = []
        for box in bboxes:
            if len(box) == 5:
                class_id, x_center, y_center, width, height = box
            elif len(box) == 6:
                class_id, x_center, y_center, width, height, conf = box
            else:
                print(f"⚠️ Invalid YOLO box format: {box}. Expected 5 or 6 values.")
                continue
            x1 = int((x_center - width / 2) * image_width)
            y1 = int((y_center - height / 2) * image_height)
            x2 = int((x_center + width / 2) * image_width)
            y2 = int((y_center + height / 2) * image_height)
            cxyxy_bboxes.append([int(class_id), x1, y1, x2, y2])
            
        return cxyxy_bboxes
    
    @staticmethod
    def convert_yolo_line_to_cxyxy(bbox_line: str, image_width: int, image_height: int) -> Optional[List[float]]:
        """
        Converts a single YOLO bounding box annotation line to (c, x1, y1, x2, y2) format.

        Args:
            bbox_line (str): A string representing a single YOLO bounding box annotation
                             where the first element is the class ID, followed by
                             other bounding box coordinates (e.g., 'class_id x_center y_center width height').
            image_width (int): Width of the image.
            image_height (int): Height of the image.
            
        Returns:
            Optional[List[float]]: A list representing the bounding box in (c, x1, y1, x2, y2) format.
                                   Returns None if the input line is invalid (less than 5 parts).
        """
        parts = bbox_line.strip().split()
        if len(parts) < 5:
            BBoxUtils.logger.error(f"⚠️ Expect cxywh or cxywh_conf format. Invalid bounding box annotation format: {bbox_line}")
            return None

        try:
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])

            # Convert to pixel coordinates
            x1 = int((x_center - width / 2) * image_width)
            y1 = int((y_center - height / 2) * image_height)
            x2 = int((x_center + width / 2) * image_width)
            y2 = int((y_center + height / 2) * image_height)

            return [int(class_id), x1, y1, x2, y2]
        except ValueError:
            BBoxUtils.logger.error(f"⚠️ Could not parse bounding box line: {bbox_line}")
            return None
        
      
def coco_array_to_yolo_file(coco_annotations_list, sahi_predictions, output_dir, image_filename, save_conf=False):
    """
    Converts a list of COCO annotations for a single image into YOLO normalized format
    and saves them to a .txt file.

    Args:
        coco_annotations_list (list): A list of COCO annotation dictionaries for a single image.
                                       Each dictionary should contain 'bbox' and 'category_id'.
        image_width (int): The width of the image.
        image_height (int): The height of the image.
        output_dir (str): The directory where the YOLO .txt file will be saved.
        image_filename (str): The base filename of the image (without extension).
                              This will be used to name the output .txt file
                              (e.g., if image_filename is 'image1.png', the label file will be 'image1.txt').

    """
    output_path = Path(output_dir) / Path(image_filename).with_suffix('.txt').name
    yolo_lines = []

    for annotation in coco_annotations_list:
        bbox = annotation.get('bbox')
        category_id = annotation.get('category_id')
        score = annotation.get('score')  
        if bbox is None:
            print(f"Warning: No bounding box found for {image_filename}. Skipping annotation.")
            continue
        if category_id is None:
            print(f"Warning: No category ID found for {image_filename}. Skipping annotation.")
            continue
        if score is None:
            print(f"Warning: No score found for {image_filename}. Skipping annotation.")
            continue

        if bbox and category_id is not None:
            x_min, y_min, width, height = bbox

            # Calculate normalized center x, center y, width, and height
            x_center = (x_min + width / 2) / sahi_predictions.image_width
            y_center = (y_min + height / 2) / sahi_predictions.image_height
            normalized_width = width / sahi_predictions.image_width
            normalized_height = height / sahi_predictions.image_height
            yolo_class_id = category_id
            if save_conf:
                yolo_lines.append(f"{yolo_class_id} {x_center:.6f} {y_center:.6f} {normalized_width:.6f} {normalized_height:.6f} {score:.6f}")
            else:
                # Save without confidence score
                yolo_lines.append(f"{yolo_class_id} {x_center:.6f} {y_center:.6f} {normalized_width:.6f} {normalized_height:.6f}")

    with open(str(output_path), 'w') as f:
        for line in yolo_lines:
            f.write(line + '\n')

    print(f"YOLO annotations saved to: {output_path}")