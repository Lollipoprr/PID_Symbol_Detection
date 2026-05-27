import os, cv2
import numpy as np
from pathlib import Path
from typing import List, Optional, Union
from natsort import natsorted


def get_files(directory: Union[str, Path], extensions: List[str]) -> List[Path]:
    """
    Recursively retrieves all files in a directory (and subdirectories) 
    that match the given extensions.

    Args:
        directory (Union[str, Path]): The root directory to search.
        extensions (List[str]): A list of file extensions to match (e.g., [".jpg", ".txt"]).

    Returns:
        List[Path]: A naturally sorted list of matching file paths.

    Raises:
        ValueError: If the path is not a directory, or no files are found.
    """
    directory = Path(directory)

    if not directory.is_dir():
        raise ValueError(f"❌ The provided path is not a directory: {directory}")
    if not extensions:
        raise ValueError("❌ No extensions provided.")

    ext_files = [file for file in directory.rglob("*") if file.suffix in extensions]

    if not ext_files:
        raise ValueError(f"❌ No files found with extensions {extensions} in {directory}")

    return natsorted(ext_files)


def get_image_files(directory: Union[str, Path], extensions: Optional[List[str]] = None) -> List[Path]:
    """
    Retrieves all image files in a directory (and subdirectories) 
    that match the given extensions.

    Args:
        directory (Union[str, Path]): The root directory to search.
        extensions (Optional[List[str]]): A list of file extensions to match (e.g., [".jpg", ".png"]).

    Returns:
        List[Path]: A naturally sorted list of matching image file paths.

    Raises:
        ValueError: If the path is not a directory, or no image files are found.
    """
    if extensions is None:
        extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]

    return get_files(directory, extensions)

def get_text_files(directory: Union[str, Path], extensions: Optional[List[str]] = None) -> List[Path]:
    """
    Retrieves all text files in a directory (and subdirectories) 
    that match the given extensions.

    Args:
        directory (Union[str, Path]): The root directory to search.
        extensions (Optional[List[str]]): A list of file extensions to match (e.g., [".txt"]).

    Returns:
        List[Path]: A naturally sorted list of matching text file paths.

    Raises:
        ValueError: If the path is not a directory, or no text files are found.
    """
    if extensions is None:
        extensions = [".txt"]

    return get_files(directory, extensions)


def resolve_image_paths(src: Union[str, Path, List[str], List[Path]]) -> List[str]:
        # if src is a directory, get all the images in the directory
        if isinstance(src, Path) and Path(src).is_dir():
            image_paths = get_image_files(src)

        # if src is a single image or a list of images, convert to list 
        elif isinstance(src, (str, Path)) and not Path(src).is_dir():
            image_paths = [src]
        # if src is a list of paths, convert to list of strings 
        elif isinstance(src, list):
            if all(isinstance(item, (str, Path)) for item in src):
                image_paths = [str(item) for item in src]
            else:
                raise ValueError("All items in src must be strings or Paths")
        else:
            raise ValueError("Invalid src type. Either a directory or a list of image paths is expected.")
        
        return image_paths