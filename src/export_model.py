"""
Model export utilities for deployment.
"""

import os
import json
import argparse
import tensorflow as tf
from tensorflow import keras

from .utils import CONFIG, load_trained_model


def export_model(model, export_path='./dr_model_export'):
    """Export model in multiple formats."""
    
    os.makedirs(export_path, exist_ok=True)
    
    print("Exporting model...")
    
    # SavedModel format (TensorFlow)
    savedmodel_path = os.path.join(export_path, 'saved_model')
    model.save(savedmodel_path)
    print(f"✓ Exported TensorFlow SavedModel: {savedmodel_path}")
    
    # H5 format
    h5_path = os.path.join(export_path, 'model.h5')
    model.save(h5_path)
    print(f"✓ Exported H5 format: {h5_path}")
    
    # TFLite format (for mobile deployment)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    
    tflite_path = os.path.join(export_path, 'model.tflite')
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    print(f"✓ Exported TFLite format: {tflite_path}")
    
    # Save config
    config = {
        'img_size': CONFIG['IMG_SIZE'],
        'class_names': CONFIG['CLASS_NAMES'],
        'model_architecture': 'EfficientNetB3',
        'input_shape': [CONFIG['IMG_SIZE'], CONFIG['IMG_SIZE'], 3]
    }
    
    config_path = os.path.join(export_path, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"✓ Exported config: {config_path}")
    
    print(f"\n✓ Model export complete: {export_path}")


def main():
    """Main function for model export."""
    parser = argparse.ArgumentParser(description='Export trained model')
    
    parser.add_argument('--model', type=str, default='./models/best_dr_model.h5',
                        help='Path to trained model')
    parser.add_argument('--output', type=str, default='./dr_model_export',
                        help='Output directory for exported models')
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model from: {args.model}")
    model = load_trained_model(args.model)
    
    # Export
    export_model(model, export_path=args.output)
    
    print("\n✓ Export completed successfully!")


if __name__ == "__main__":
    main()