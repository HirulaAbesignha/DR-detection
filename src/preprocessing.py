"""
Image preprocessing functions for retinal images.
"""

import cv2
import numpy as np
from PIL import Image

from .utils import CONFIG


def normalize_image(image):
    """Comprehensive image normalization for retinal images."""
    # Ensure uint8
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    
    # Ensure RGB
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)


    # Apply CLAHE for contrast enhancement
    img_lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    img_lab[:, :, 0] = clahe.apply(img_lab[:, :, 0])
    img_enhanced = cv2.cvtColor(img_lab, cv2.COLOR_LAB2RGB).astype(np.float32)
    
    # Normalize to [0, 1]
    img_norm = img_enhanced / 255.0
    
    # Apply ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_norm = (img_norm - mean) / std
    
    return img_norm

def preprocess_image(image, target_size=None):
    """Complete preprocessing pipeline for a single image."""
    if target_size is None:
        target_size = (CONFIG['IMG_SIZE'], CONFIG['IMG_SIZE'])
    
    # Load image if path
    if isinstance(image, str):
        image = cv2.imread(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif isinstance(image, Image.Image):
        image = np.array(image)
    
    # Ensure numpy array
    image = np.array(image)
    
    # Resize if needed
    if image.shape[:2] != target_size:
        image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    
    # Normalize
    image = normalize_image(image)
    
    return image