"""
Visualization utilities for evaluation.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize


def plot_confusion_matrix(y_true, y_pred, class_names):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=list(class_names.values()),
        yticklabels=list(class_names.values()),
        cbar_kws={'label': 'Count'}
    )
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig('outputs/plots/confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("✓ Confusion matrix saved to: outputs/plots/confusion_matrix.png")
    plt.close()


def plot_roc_curves(y_true, y_pred_proba, class_names):
    """Plot ROC curves for all classes."""
    # Binarize labels
    y_true_bin = label_binarize(y_true, classes=list(range(5)))
    
    plt.figure(figsize=(12, 8))
    
    # Plot ROC curve for each class
    for i in range(5):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
        roc_auc = auc(fpr, tpr)
        
        plt.plot(
            fpr, tpr, linewidth=2,
            label=f'{class_names[i]} (AUC = {roc_auc:.3f})'
        )
    
    # Plot diagonal
    plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - Multi-Class Classification', fontsize=16, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('outputs/plots/roc_curves.png', dpi=300, bbox_inches='tight')
    print("✓ ROC curves saved to: outputs/plots/roc_curves.png")
    plt.close()