"""
Utility functions for the DR detection system.
"""

import os
import gc
import psutil
import tensorflow as tf
from tensorflow import keras

# Global configuration
CONFIG = {
    'IMG_SIZE': 224,
    'BATCH_SIZE': 8,
    'EPOCHS': 50,
    'LEARNING_RATE': 1e-4,
    'SAMPLE_SIZE': 2500,
    'PATIENCE': 10,
    'NUM_WORKERS': 4,
    'CLASS_NAMES': {
        0: "No DR",
        1: "Mild",
        2: "Moderate",
        3: "Severe",
        4: "Proliferative DR"
    },
    'MODEL_SAVE_PATH': './models/best_dr_model.h5',
    'AUGMENTATION_ENABLED': True
}


def setup_gpu(memory_limit_mb=7168):
    """Configure GPU settings for local machine (8GB GPU)."""
    print("Configuring GPU...")
    gpus = tf.config.list_physical_devices('GPU')
    
    if gpus:
        try:
            tf.config.set_logical_device_configuration(
                gpus[0],
                [tf.config.LogicalDeviceConfiguration(memory_limit=memory_limit_mb)]
            )
            print(f"✓ GPU configured with {memory_limit_mb}MB memory limit")
            print(f"✓ GPU Device: {gpus[0]}")
            return True
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")
            return False
    else:
        print("⚠ No GPU found, using CPU")
        return False


def memory_cleanup():
    """Clean up memory and TensorFlow session."""
    gc.collect()
    try:
        tf.keras.backend.clear_session()
    except:
        pass


def check_memory():
    """Check current memory usage."""
    return psutil.virtual_memory().percent


def create_directories():
    """Create necessary project directories."""
    directories = [
        'data/retinal_images',
        'models',
        'outputs/plots',
        'outputs/logs',
        'outputs/predictions',
        'configs',
        'cache'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print("✓ Project directories created")


def load_trained_model(model_path):
    """Load a trained model from disk."""
    print(f"Loading model from: {model_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    custom_objects = {
        'accuracy': keras.metrics.Accuracy,
        'auc': keras.metrics.AUC,
        'precision': keras.metrics.Precision,
        'recall': keras.metrics.Recall
    }
    
    try:
        model = keras.models.load_model(model_path, custom_objects=custom_objects)
        print("✓ Model loaded successfully")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        raise


def save_model(model, save_path):
    """Save model to disk."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    print(f"✓ Model saved to: {save_path}")


def print_system_info():
    """Print system information."""
    print("\n" + "="*70)
    print("SYSTEM INFORMATION")
    print("="*70)
    
    # RAM
    ram = psutil.virtual_memory()
    print(f"RAM Total: {ram.total / (1024**3):.1f} GB")
    print(f"RAM Available: {ram.available / (1024**3):.1f} GB")
    print(f"RAM Usage: {ram.percent:.1f}%")
    
    # GPU
    gpus = tf.config.list_physical_devices('GPU')
    print(f"\nGPU Available: {'Yes' if gpus else 'No'}")
    if gpus:
        for i, gpu in enumerate(gpus):
            print(f"GPU {i}: {gpu.name}")
    
    # Python & TensorFlow
    import sys
    print(f"\nPython Version: {sys.version.split()[0]}")
    print(f"TensorFlow Version: {tf.__version__}")
    
    print("="*70 + "\n")