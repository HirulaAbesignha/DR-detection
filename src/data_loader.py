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
    """Load diabetic retinopathy dataset."""
    if sample_size is None:
        sample_size = CONFIG['SAMPLE_SIZE']
    
    # Check for local dataset first
    if data_path and os.path.exists(data_path):
        print(f"Loading from local path: {data_path}")
        
        return load_local_dataset(data_path, sample_size)
    
    # Try loading from HuggingFace  
    print("Loading dataset from HuggingFace...")
    try:
        from datasets import load_dataset as hf_load_dataset
        
        # Load entire dataset first
        print("Downloading dataset (this may take a few minutes on first run)...")
        ds = hf_load_dataset(
            "youssefedweqd/Diabetic_Retinopathy_Detection_preprocessed2"
     )
        print(f"Dataset info: {ds}")
        
        # Try 'train' split
        if 'train' in ds:
            dataset = ds['train']
        else:
            # Use first available split
            dataset = ds[list(ds.keys())[0]]
        
        print(f"Using split with {len(dataset)} total samples")
        
        # Sample evenly from all classes
        samples_by_class = {0: [], 1: [], 2: [], 3: [], 4: []}
        
        # First pass: collect all samples by class
        print("Organizing samples by class...")
        for i, sample in enumerate(dataset):
            label = int(sample['label'])
            if label in samples_by_class:
                samples_by_class[label].append({
                    'image': np.array(sample['image']),
                    'label': label,
                    'id': f'hf_{i}'
                })
            
            if (i + 1) % 1000 == 0:
                print(f"Processed {i+1} samples...")
                counts = {k: len(v) for k, v in samples_by_class.items()}
                print(f"  Current distribution: {counts}")
        
        # Report distribution
        print(f"\nFull dataset distribution:")
        for cls in range(5):
            print(f"  Class {cls}: {len(samples_by_class[cls])} images")
        
        # Sample from each class proportionally
        samples = []
        samples_per_class = sample_size // 5
        
        for cls in range(5):
            available = len(samples_by_class[cls])
            take = min(available, samples_per_class)
            
            if take > 0:
                np.random.shuffle(samples_by_class[cls])
                samples.extend(samples_by_class[cls][:take])
                print(f"  Taking {take} samples from class {cls}")
        
        np.random.shuffle(samples)
        
        print(f"\n✓ Dataset loaded: {len(samples)} samples from HuggingFace")
        return samples
        
    except Exception as e:
        print(f"Could not load from HuggingFace: {e}")
        import traceback
        print(traceback.format_exc())
        raise ValueError("No dataset found! Please provide --data-path with real images")


def load_local_dataset(dataset_path, sample_size=None):
    """Load dataset from local folder structure."""
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
    
    print(f"✓ Loaded {len(samples)} images from local storage")
    return samples


def create_simulated_dataset(sample_size=2000):
    """Create EXTREMELY simple and distinct simulated images for easy learning."""
    print("Creating super simple, highly distinct dataset for easy learning...")
    np.random.seed(42)
    
    samples = []
    class_distribution = [sample_size // 5] * 5
    
    sample_id = 0
    for class_label, count in enumerate(class_distribution):
        for _ in range(count):
            # Create solid color background for each class (VERY DISTINCT)
            img = np.zeros((CONFIG['IMG_SIZE'], CONFIG['IMG_SIZE'], 3), dtype=np.uint8)
            
            if class_label == 0:  # No DR - Pure GREEN background
                img[:, :, 1] = 200  # Green channel
                # Add slight random texture
                img = img + np.random.randint(0, 20, img.shape, dtype=np.uint8)
                
            elif class_label == 1:  # Mild - GREEN + 1 large RED circle center
                img[:, :, 1] = 180  # Green background
                img = img + np.random.randint(0, 15, img.shape, dtype=np.uint8)
                # ONE large red circle in center
                center = CONFIG['IMG_SIZE'] // 2
                cv2.circle(img, (center, center), 40, (0, 0, 255), -1)
                
            elif class_label == 2:  # Moderate - GREEN + 4 RED circles (corners)
                img[:, :, 1] = 160  # Green background
                img = img + np.random.randint(0, 15, img.shape, dtype=np.uint8)
                # Four red circles in corners
                offset = 50
                positions = [
                    (offset, offset),
                    (CONFIG['IMG_SIZE']-offset, offset),
                    (offset, CONFIG['IMG_SIZE']-offset),
                    (CONFIG['IMG_SIZE']-offset, CONFIG['IMG_SIZE']-offset)
                ]
                for pos in positions:
                    cv2.circle(img, pos, 30, (0, 0, 255), -1)
                    
            elif class_label == 3:  # Severe - GREEN + RED grid pattern
                img[:, :, 1] = 140  # Green background
                img = img + np.random.randint(0, 15, img.shape, dtype=np.uint8)
                # Grid of 9 red circles (3x3)
                for i in range(3):
                    for j in range(3):
                        x = (i + 1) * CONFIG['IMG_SIZE'] // 4
                        y = (j + 1) * CONFIG['IMG_SIZE'] // 4
                        cv2.circle(img, (x, y), 25, (0, 0, 255), -1)
                        
            elif class_label == 4:  # Proliferative - GREEN + RED circles + WHITE lines
                img[:, :, 1] = 120  # Dark green background
                img = img + np.random.randint(0, 15, img.shape, dtype=np.uint8)
                # Many red circles (random)
                for _ in range(15):
                    x = np.random.randint(40, CONFIG['IMG_SIZE']-40)
                    y = np.random.randint(40, CONFIG['IMG_SIZE']-40)
                    cv2.circle(img, (x, y), 20, (0, 0, 255), -1)
                # White crossing lines (X pattern)
                cv2.line(img, (0, 0), (CONFIG['IMG_SIZE'], CONFIG['IMG_SIZE']), 
                        (255, 255, 255), 5)
                cv2.line(img, (CONFIG['IMG_SIZE'], 0), (0, CONFIG['IMG_SIZE']), 
                        (255, 255, 255), 5)
            
            samples.append({
                'image': img,
                'label': class_label,
                'id': f'sim_{sample_id}'
            })
            sample_id += 1
    
    np.random.shuffle(samples)
    print(f"✓ Created {len(samples)} SUPER SIMPLE samples (should reach 80-90% accuracy)")
    print("   Class 0: Green only")
    print("   Class 1: Green + 1 red circle center")
    print("   Class 2: Green + 4 red circles corners")
    print("   Class 3: Green + 3x3 red grid")
    print("   Class 4: Green + many red circles + white X")
    return samples
    
    np.random.shuffle(samples)
    print(f"✓ Created {len(samples)} samples with highly distinct visual patterns")
    return samples

def analyze_dataset(samples):
    """Analyze dataset statistics."""
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
    """Memory-efficient data generator for DR dataset."""
    
    def __init__(self, samples, batch_size=None, shuffle=True, augment=False):
        self.samples = samples
        self.batch_size = batch_size or CONFIG['BATCH_SIZE']
        self.shuffle = shuffle
        self.augment = augment
        self.indices = np.arange(len(self.samples))
        self.on_epoch_end()
    
    def __len__(self):
        return int(np.ceil(len(self.samples) / self.batch_size))
    
    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_samples = [self.samples[i] for i in batch_indices]
        
        X, y = self._generate_batch(batch_samples)
        return X, y
    
    def _generate_batch(self, batch_samples):
        X = np.zeros((len(batch_samples), CONFIG['IMG_SIZE'], CONFIG['IMG_SIZE'], 3), dtype=np.float32)
        y = np.zeros((len(batch_samples), 5), dtype=np.float32)
        
        for i, sample in enumerate(batch_samples):
            try:
                img = np.array(sample['image'])
                label = int(sample['label'])
                
                # Resize if needed
                if img.shape[:2] != (CONFIG['IMG_SIZE'], CONFIG['IMG_SIZE']):
                    img = cv2.resize(img, (CONFIG['IMG_SIZE'], CONFIG['IMG_SIZE']))
                
                # Preprocessing first
                img = preprocess_image(img)
                
                # Augmentation after preprocessing (more effective)
                if self.augment:
                    # Apply augmentation to preprocessed image
                    if np.random.random() < 0.5:
                        img = np.fliplr(img)  # Flip horizontally
                
                X[i] = img
                y[i, label] = 1.0
                
            except Exception as e:
                # Use zero image for failed samples
                X[i] = np.zeros((CONFIG['IMG_SIZE'], CONFIG['IMG_SIZE'], 3))
                y[i, 0] = 1.0
        
        return X, y
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


def calculate_class_weights(samples):
    """Calculate class weights for imbalanced dataset with stronger correction."""
    labels = [int(s['label']) for s in samples]
    class_counts = Counter(labels)
    
    total = len(labels)
    num_classes = 5
    
    # Calculate base weights
    class_weights = {}
    for cls in range(num_classes):
        count = class_counts.get(cls, 1)
        # Use square root to make weights more aggressive for minority classes
        class_weights[cls] = (total / (num_classes * count)) ** 1.5
    
    # Further boost minority classes (3 and 4)
    class_weights[3] = class_weights[3] * 3.0  # Severe
    class_weights[4] = class_weights[4] * 3.0  # Proliferative
    
    print(f"Adjusted class weights: {class_weights}")
    
    return class_weights

def balance_dataset(samples, max_samples_per_class=2000):
    """Balance dataset by limiting majority class samples."""
    from collections import defaultdict
    
    # Group samples by class
    class_samples = defaultdict(list)
    for sample in samples:
        class_samples[sample['label']].append(sample)
    
    # Print original distribution
    print("\nOriginal distribution:")
    for cls in range(5):
        print(f"  Class {cls} ({CONFIG['CLASS_NAMES'][cls]}): {len(class_samples[cls])} samples")
    
    # Balance by limiting each class
    balanced_samples = []
    for cls in range(5):
        samples_in_class = class_samples[cls]
        
        if len(samples_in_class) > max_samples_per_class:
            # Undersample majority class
            np.random.shuffle(samples_in_class)
            samples_in_class = samples_in_class[:max_samples_per_class]
        
        balanced_samples.extend(samples_in_class)
    
    # Shuffle all samples
    np.random.shuffle(balanced_samples)
    
    # Print balanced distribution
    class_counts = Counter([s['label'] for s in balanced_samples])
    print("\nBalanced distribution:")
    for cls in range(5):
        print(f"  Class {cls} ({CONFIG['CLASS_NAMES'][cls]}): {class_counts[cls]} samples")
    
    return balanced_samples