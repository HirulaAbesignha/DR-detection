"""
Data loading and generator utilities.
"""

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
                    x, y = np.random.randint(20, CONFIG['IMG_SIZE']-20, 2)
                    cv2.circle(img, (x, y), np.random.randint(2, 5), (150, 0, 0), -1)
            elif class_label >= 1:  # Mild
                for _ in range(np.random.randint(2, 6)):
                    x, y = np.random.randint(20, CONFIG['IMG_SIZE']-20, 2)
                    cv2.circle(img, (x, y), np.random.randint(1, 3), (100, 0, 0), -1)
            
            samples.append({
                'image': img,
                'label': class_label,
                'id': f'sim_{sample_id}'
            })
            sample_id += 1
    
    np.random.shuffle(samples)
    print(f"Created {len(samples)} simulated samples")
    return samples


def analyze_dataset(samples):
    """
    Analyze dataset statistics.
    
    Args:
        samples: List of samples
        
    Returns:
        dict: Dataset statistics
    """
    labels = [s['label'] for s in samples]
    class_counts = Counter(labels)
    
    stats = {
        'total_samples': len(samples),
        'class_distribution': dict(class_counts),
        'class_names': CONFIG['CLASS_NAMES'],
        'imbalance_ratio': max(class_counts.values()) / min(class_counts.values()) if class_counts else 0
    }
    
    return stats


class DRDataGenerator(keras.utils.Sequence):
    """
    Memory-efficient data generator for DR dataset.
    """
    
    def __init__(self, samples, batch_size=None, shuffle=True, augment=False):
        """
        Initialize data generator.
        
        Args:
            samples: List of samples
            batch_size: Batch size (default: from CONFIG)
            shuffle: Whether to shuffle data
            augment: Whether to apply augmentation
        """
        self.samples = samples
        self.batch_size = batch_size or CONFIG['BATCH_SIZE']
        self.shuffle = shuffle
        self.augment = augment
        self.indices = np.arange(len(self.samples))
        self.on_epoch_end()
    
    def __len__(self):
        """Number of batches per epoch."""
        return int(np.ceil(len(self.samples) / self.batch_size))
    
    def __getitem__(self, idx):
        """
        Generate one batch of data.
        
        Args:
            idx: Batch index
            
        Returns:
            tuple: (X_batch, y_batch)
        """
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_samples = [self.samples[i] for i in batch_indices]
        
        X, y = self._generate_batch(batch_samples)
        return X, y
    
    def _generate_batch(self, batch_samples):
        """
        Generate batch data.
        
        Args:
            batch_samples: List of samples for this batch
            
        Returns:
            tuple: (X, y) as numpy arrays
        """
        X = np.zeros((len(batch_samples), CONFIG['IMG_SIZE'], CONFIG['IMG_SIZE'], 3), dtype=np.float32)
        y = np.zeros((len(batch_samples), 5), dtype=np.float32)
        
        for i, sample in enumerate(batch_samples):
            try:
                img = np.array(sample['image'])
                label = int(sample['label'])
                
                # Resize if needed
                if img.shape[:2] != (CONFIG['IMG_SIZE'], CONFIG['IMG_SIZE']):
                    img = cv2.resize(img, (CONFIG['IMG_SIZE'], CONFIG['IMG_SIZE']))
                
                # Augmentation
                if self.augment:
                    img = apply_augmentation(img, label)
                
                # Preprocessing
                img = preprocess_image(img)
                
                X[i] = img
                y[i, label] = 1.0
                
            except Exception as e:
                # Use zero image for failed samples
                print(f"Error processing sample: {e}")
                X[i] = np.zeros((CONFIG['IMG_SIZE'], CONFIG['IMG_SIZE'], 3))
                y[i, 0] = 1.0
        
        return X, y
    
    def on_epoch_end(self):
        """Shuffle indices after each epoch."""
        if self.shuffle:
            np.random.shuffle(self.indices)


def calculate_class_weights(samples):
    """
    Calculate class weights for imbalanced dataset.
    
    Args:
        samples: List of samples
        
    Returns:
        dict: Class weights
    """
    labels = [int(s['label']) for s in samples]
    class_counts = Counter(labels)
    
    total = len(labels)
    num_classes = 5
    
    class_weights = {}
    for cls in range(num_classes):
        count = class_counts.get(cls, 1)
        class_weights[cls] = total / (num_classes * count)
    
    return class_weights


if __name__ == "__main__":
    # Test data loading
    print("Testing data loader...")
    
    # Load dataset
    samples = load_dataset(sample_size=100)
    
    # Analyze
    stats = analyze_dataset(samples)
    print("\nDataset Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Test generator
    generator = DRDataGenerator(samples, batch_size=8, augment=True)
    print(f"\nGenerator created: {len(generator)} batches")
    
    # Test batch
    X_batch, y_batch = generator[0]
    print(f"Batch shape: X={X_batch.shape}, y={y_batch.shape}")
    
    # Test class weights
    weights = calculate_class_weights(samples)
    print(f"\nClass weights: {weights}")