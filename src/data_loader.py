import os
import glob
import numpy as np
import cv2
from collections import Counter
import tensorflow as tf
from tensorflow import keras

from .utils import CONFIG
from .preprocessing import preprocess_image
from .augmentation import apply_augmentation


def load_dataset(data_path=None, sample_size=None):
    """
    Load diabetic retinopathy dataset.
    
    Args:
        data_path: Path to local dataset (optional)
        sample_size: Number of samples to load (optional)
        
    Returns:
        List of samples with 'image', 'label', 'id' keys
    """
    if sample_size is None:
        sample_size = CONFIG['SAMPLE_SIZE']
    
    # Check for local dataset first
    if data_path and os.path.exists(data_path):
        print(f"Loading from local path: {data_path}")
        return load_local_dataset(data_path, sample_size)
    
    # Try loading from HuggingFace
    try:
        print("Loading dataset from HuggingFace...")
        from datasets import load_dataset as hf_load_dataset
        
        ds = hf_load_dataset(
            "youssefedweqd/Diabetic_Retinopathy_Detection_preprocessed2",
            streaming=True
        )
        
        samples = []
        train_iter = iter(ds["train"])
        
        for i, sample in enumerate(train_iter):
            if i >= sample_size:
                break
            samples.append(sample)
            
            if (i + 1) % 100 == 0:
                print(f"Loaded {i+1}/{sample_size} samples")
        
        print(f"Dataset loaded: {len(samples)} samples")
        return samples
        
    except Exception as e:
        print(f"Could not load from HuggingFace: {e}")
        print("Creating simulated dataset...")
        return create_simulated_dataset(sample_size)


def load_local_dataset(dataset_path, sample_size=None):
    """
    Load dataset from local folder structure.
    
    Expected structure:
        dataset_path/
            class_0/
                image1.jpg
                image2.jpg
            class_1/
            ...
    
    Args:
        dataset_path: Path to dataset root
        sample_size: Maximum samples per class
        
    Returns:
        List of samples
    """
    samples = []
    samples_per_class = sample_size // 5 if sample_size else None
    
    for class_label in range(5):
        class_folder = os.path.join(dataset_path, f'class_{class_label}')
        
        if not os.path.exists(class_folder):
            print(f"Warning: {class_folder} not found, skipping...")
            continue
        
        # Get all image files
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            image_files.extend(glob.glob(os.path.join(class_folder, ext)))
        
        # Limit samples if specified
        if samples_per_class:
            image_files = image_files[:samples_per_class]
        
        print(f"Loading class {class_label} ({CONFIG['CLASS_NAMES'][class_label]}): {len(image_files)} images")
        
        for img_path in image_files:
            try:
                img = cv2.imread(img_path)
                if img is None:
                    continue
                    
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                samples.append({
                    'image': img,
                    'label': class_label,
                    'id': os.path.basename(img_path)
                })
                
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
                continue
    
    print(f"Loaded {len(samples)} images from local storage")
    return samples


def create_simulated_dataset(sample_size=2000):
    """
    Create simulated retinal images for testing.
    
    Args:
        sample_size: Total number of samples to create
        
    Returns:
        List of simulated samples
    """
    print("Creating simulated dataset...")
    np.random.seed(42)
    
    samples = []
    # Balanced distribution
    class_distribution = [sample_size // 5] * 5
    
    sample_id = 0
    for class_label, count in enumerate(class_distribution):
        for _ in range(count):
            # Create base image
            img = np.random.randint(20, 200, (CONFIG['IMG_SIZE'], CONFIG['IMG_SIZE'], 3), dtype=np.uint8)
            
            # Add patterns based on severity
            if class_label >= 3:  # Severe/Proliferative
                for _ in range(np.random.randint(10, 20)):
                    x, y = np.random.randint(20, CONFIG['IMG_SIZE']-20, 2)
                    cv2.circle(img, (x, y), np.random.randint(2, 8), (0, 0, 255), -1)
            elif class_label >= 2:  # Moderate
                for _ in range(np.random.randint(5, 12)):
                    x, y