"""
Model evaluation utilities.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score
)

from .utils import CONFIG


def evaluate_model(model, test_generator, test_samples):
    """Evaluate model on test set."""
    print("Evaluating model on test set...")
    
    # Get predictions
    y_true = []
    y_pred_proba = []
    
    for i in range(len(test_generator)):
        X_batch, y_batch = test_generator[i]
        y_true.extend(np.argmax(y_batch, axis=1))
        y_pred_proba.extend(model.predict(X_batch, verbose=0))
    
    y_true = np.array(y_true)
    y_pred_proba = np.array(y_pred_proba)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Calculate AUC if possible
    try:
        from sklearn.preprocessing import label_binarize
        y_true_bin = label_binarize(y_true, classes=list(range(5)))
        auc = roc_auc_score(y_true_bin, y_pred_proba, average='weighted', multi_class='ovr')
    except:
        auc = 0.0
    
    # Print classification report
    print("\nClassification Report:")
    try:
        print(classification_report(
            y_true, y_pred,
            labels=list(range(5)),
            target_names=list(CONFIG['CLASS_NAMES'].values()),
            zero_division=0
        ))
    except Exception as e:
        print(f"Could not generate full report: {e}")
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc': auc,
        'y_true': y_true,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }
    
    return metrics


def plot_training_history(history):
    """Plot training history."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Accuracy
    axes[0, 0].plot(history.history['accuracy'], label='Train', linewidth=2)
    axes[0, 0].plot(history.history['val_accuracy'], label='Validation', linewidth=2)
    axes[0, 0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Loss
    axes[0, 1].plot(history.history['loss'], label='Train', linewidth=2)
    axes[0, 1].plot(history.history['val_loss'], label='Validation', linewidth=2)
    axes[0, 1].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # AUC
    if 'auc' in history.history:
        axes[1, 0].plot(history.history['auc'], label='Train', linewidth=2)
        axes[1, 0].plot(history.history['val_auc'], label='Validation', linewidth=2)
        axes[1, 0].set_title('Model AUC', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('AUC')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Precision/Recall
    if 'precision' in history.history:
        axes[1, 1].plot(history.history['precision'], label='Precision', linewidth=2)
        axes[1, 1].plot(history.history['recall'], label='Recall', linewidth=2)
        axes[1, 1].set_title('Precision & Recall', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('outputs/plots/training_history.png', dpi=300, bbox_inches='tight')
    print("✓ Training history plot saved to: outputs/plots/training_history.png")
    plt.close()