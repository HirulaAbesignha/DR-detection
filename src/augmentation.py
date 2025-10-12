"""
Data augmentation functions for retinal images.
"""

import cv2
import numpy as np


def apply_augmentation(image, label=None):
    """Apply data augmentation based on class."""
    # Class-specific augmentation probabilities
    aug_prob = {0: 0.2, 1: 0.4, 2: 0.6, 3: 0.8, 4: 0.8}
    
    probability = aug_prob.get(label, 0.3) if label is not None else 0.5
    
    if np.random.random() < probability:
        # Apply multiple augmentations
        image = random_horizontal_flip(image)
        image = random_rotation(image)
        image = random_brightness(image)
    
    return image


def random_horizontal_flip(image, probability=0.5):
    """Randomly flip image horizontally."""
    if np.random.random() < probability:
        return cv2.flip(image, 1)
    return image


def random_rotation(image, max_angle=15, probability=0.3):
    """Randomly rotate image."""
    if np.random.random() < probability:
        angle = np.random.uniform(-max_angle, max_angle)
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        
        return rotated
    return image


def random_brightness(image, range_factor=0.2, probability=0.3):
    """Randomly adjust brightness."""
    if np.random.random() < probability:
        factor = np.random.uniform(1 - range_factor, 1 + range_factor)
        adjusted = np.clip(image.astype(np.float32) * factor, 0, 255).astype(np.uint8)
        return adjusted
    return image