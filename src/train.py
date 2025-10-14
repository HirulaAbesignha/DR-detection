"""
Training script for DR detection model.
"""

import os
import argparse
import numpy as np
import tensorflow.keras.backend as K

def focal_loss(gamma=2.0, alpha=0.25):
    """
    Focal loss for handling class imbalance.
    Focuses learning on hard examples.
    """
    def focal_loss_fixed(y_true, y_pred):
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        
        cross_entropy = -y_true * K.log(y_pred)
        loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy
        
        return K.sum(loss, axis=-1)
    
    return focal_loss_fixed

from sklearn.model_selection import train_test_split
from tensorflow import keras

from .utils import (
    CONFIG, setup_gpu, memory_cleanup, create_directories,
    print_system_info, save_model
)
from .model import create_dr_model, compile_model, unfreeze_base_model
from .data_loader import load_dataset, DRDataGenerator, calculate_class_weights, analyze_dataset
from .evaluation import evaluate_model, plot_training_history
from .visualization import plot_confusion_matrix, plot_roc_curves


def get_callbacks(model_save_path, patience=None):
    """Get training callbacks."""
    if patience is None:
        patience = CONFIG['PATIENCE']
    
    callbacks_list = [
        keras.callbacks.ModelCheckpoint(
            model_save_path,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=patience // 2,
            min_lr=1e-7,
            verbose=1
        ),
        keras.callbacks.CSVLogger(
            'outputs/logs/training_log.csv',
            append=False
        ),
        keras.callbacks.TerminateOnNaN()
    ]
    
    return callbacks_list


def train_model(data_path=None, epochs=None, batch_size=None, learning_rate=None):
    """Complete training pipeline."""
    # Setup
    print("="*70)
    print("DIABETIC RETINOPATHY DETECTION - TRAINING PIPELINE")
    print("="*70)
    
    print_system_info()
    setup_gpu()
    create_directories()
    
    # Update config if provided
    if epochs:
        CONFIG['EPOCHS'] = epochs
    if batch_size:
        CONFIG['BATCH_SIZE'] = batch_size
    if learning_rate:
        CONFIG['LEARNING_RATE'] = learning_rate
    
    # Load dataset
    print("\n[1/8] Loading dataset...")
    samples = load_dataset(data_path=data_path, sample_size=CONFIG['SAMPLE_SIZE'])
    print(f"✓ Loaded {len(samples)} samples")
    
    # Analyze dataset
    stats = analyze_dataset(samples)
    print(f"✓ Original class distribution: {stats['class_distribution']}")
    print(f"✓ Original imbalance ratio: {stats['imbalance_ratio']:.2f}:1")
    
    # ALWAYS balance dataset for severely imbalanced data
    print("\n⚠ Balancing dataset to ensure all classes are represented...")
    from .data_loader import balance_dataset
    samples = balance_dataset(samples, max_samples_per_class=500)
    
    # Analyze balanced dataset
    stats = analyze_dataset(samples)
    print(f"✓ Balanced class distribution: {stats['class_distribution']}")
    print(f"✓ Balanced imbalance ratio: {stats['imbalance_ratio']:.2f}:1")
    
    # Split dataset AFTER balancing (this ensures all classes in test set)
    print("\n[2/8] Splitting dataset...")
    labels = [s['label'] for s in samples]
    
    # Verify all classes are present
    unique_labels = set(labels)
    print(f"✓ Classes present in dataset: {sorted(unique_labels)}")
    
    if len(unique_labels) < 5:
        print(f"⚠ WARNING: Only {len(unique_labels)} classes found! Expected 5.")
        print("   The model may not learn to distinguish all DR severities.")
    
    train_samples, test_samples = train_test_split(
        samples, test_size=0.2,
        stratify=labels,
        random_state=42
    )
    
    train_labels = [s['label'] for s in train_samples]
    train_samples, val_samples = train_test_split(
        train_samples, test_size=0.15,
        stratify=train_labels,
        random_state=42
    )
    
    print(f"✓ Train: {len(train_samples)} | Val: {len(val_samples)} | Test: {len(test_samples)}")
    
    # Create data generators
    print("\n[3/8] Creating data generators...")
    train_gen = DRDataGenerator(
        train_samples,
        batch_size=CONFIG['BATCH_SIZE'],
        shuffle=True,
        augment=True
    )
    
    val_gen = DRDataGenerator(
        val_samples,
        batch_size=CONFIG['BATCH_SIZE'],
        shuffle=False,
        augment=False
    )
    
    test_gen = DRDataGenerator(
        test_samples,
        batch_size=CONFIG['BATCH_SIZE'],
        shuffle=False,
        augment=False
    )
    
    print(f"✓ Generators created: Train={len(train_gen)} batches, Val={len(val_gen)} batches")
    
    # Calculate class weights
    class_weights = calculate_class_weights(train_samples)
    print(f"✓ Class weights: {class_weights}")
    
    # Create model
    print("\n[4/8] Building model...")
    model, base_model = create_dr_model()
    # Use focal loss for imbalanced data
    optimizer = keras.optimizers.Adam(learning_rate=CONFIG['LEARNING_RATE'])
    model.compile(
        optimizer=optimizer,
        loss=focal_loss(gamma=2.0, alpha=0.25),
        metrics=[
            'accuracy',
            keras.metrics.AUC(name='auc'),
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall')
        ]
    )
    
    print(f"✓ Model created: {model.count_params():,} parameters")
    print("\nModel architecture:")
    model.summary()
    
    # Get callbacks
    callbacks_list = get_callbacks(CONFIG['MODEL_SAVE_PATH'], patience=CONFIG['PATIENCE'])
    
    # Training Stage 1: Frozen base
    print("\n[5/8] Training Stage 1 - Frozen base model...")
    print(f"Training for {CONFIG['EPOCHS']} epochs (max)...")
    
    history1 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=CONFIG['EPOCHS'],
        callbacks=callbacks_list,
        class_weight=class_weights,
        verbose=1
    )
    
    memory_cleanup()
    
    # Training Stage 2: Fine-tuning
    print("\n[6/8] Training Stage 2 - Fine-tuning...")
    base_model = unfreeze_base_model(base_model, num_layers_to_freeze=100)
    
    # Recompile with lower learning rate
    model = compile_model(model, learning_rate=CONFIG['LEARNING_RATE'] / 10, use_focal_loss=False)
    
    history2 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=20,
        callbacks=callbacks_list,
        class_weight=class_weights,
        verbose=1
    )
    
    # Combine histories
    for key in history1.history:
        if key in history2.history:
            history1.history[key].extend(history2.history[key])
    
    memory_cleanup()
    
    # Plot training history
    print("\n[7/8] Generating visualizations...")
    plot_training_history(history1)
    
    # Evaluate on test set
    print("\n[8/8] Evaluating on test set...")
    metrics = evaluate_model(model, test_gen, test_samples)
    
    print(f"\n{'='*70}")
    print("FINAL TEST RESULTS:")
    print(f"{'='*70}")
    print(f"Test Accuracy:  {metrics['accuracy']*100:.2f}%")
    print(f"Test AUC:       {metrics['auc']:.4f}")
    print(f"Test Precision: {metrics['precision']:.4f}")
    print(f"Test Recall:    {metrics['recall']:.4f}")
    print(f"Test F1-Score:  {metrics['f1_score']:.4f}")
    
    # Generate confusion matrix and ROC curves
    plot_confusion_matrix(metrics['y_true'], metrics['y_pred'], CONFIG['CLASS_NAMES'])
    plot_roc_curves(metrics['y_true'], metrics['y_pred_proba'], CONFIG['CLASS_NAMES'])
    
    # Save final model
    save_model(model, CONFIG['MODEL_SAVE_PATH'])
    
    memory_cleanup()
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    
    return model, history1, metrics


def main():
    """Main function for command-line training."""
    parser = argparse.ArgumentParser(description='Train DR detection model')
    
    parser.add_argument('--data-path', type=str, default=None,
                        help='Path to dataset directory')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--sample-size', type=int, default=5000,
                        help='Number of samples to load')
    
    args = parser.parse_args()
    
    # Update config
    CONFIG['SAMPLE_SIZE'] = args.sample_size
    
    # Train model
    model, history, metrics = train_model(
        data_path=args.data_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )
    
    print("\n✓ Training completed successfully!")
    print(f"✓ Model saved to: {CONFIG['MODEL_SAVE_PATH']}")


if __name__ == "__main__":
    main()