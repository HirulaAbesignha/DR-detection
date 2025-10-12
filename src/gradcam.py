"""
Grad-CAM visualization for model interpretability.
"""

import numpy as np
import cv2
import tensorflow as tf


def make_gradcam_heatmap(img_array, model, last_conv_layer_name=None, pred_index=None):
    """Generate Grad-CAM heatmap."""
    # Find last conv layer if not specified
    if last_conv_layer_name is None:
        for layer in reversed(model.layers):
            if 'conv' in layer.name.lower():
                last_conv_layer_name = layer.name
                break
    
    if last_conv_layer_name is None:
        raise ValueError("No convolutional layer found in model")
    
    # Create gradient model
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )
    
    # Compute gradient
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        
        class_channel = predictions[:, pred_index]
    
    # Gradient of class with respect to feature maps
    grads = tape.gradient(class_channel, conv_outputs)
    
    # Global average pooling of gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Weight feature maps by gradients
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    # Normalize heatmap
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    
    return heatmap.numpy()


def apply_gradcam(image, model, alpha=0.4):
    """Apply Grad-CAM visualization to image."""
    # Prepare image
    img_array = np.expand_dims(image, axis=0)
    
    # Generate heatmap
    try:
        heatmap = make_gradcam_heatmap(img_array, model)
    except Exception as e:
        print(f"Grad-CAM failed: {e}")
        # Return denormalized original if Grad-CAM fails
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_denorm = (image * std + mean) * 255.0
        return np.clip(img_denorm, 0, 255).astype(np.uint8)
    
    # Resize heatmap to image size
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    
    # Apply colormap
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Denormalize original image
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_denorm = (image * std + mean) * 255.0
    img_denorm = np.clip(img_denorm, 0, 255).astype(np.uint8)
    
    # Superimpose heatmap
    superimposed = cv2.addWeighted(img_denorm, 1-alpha, heatmap_colored, alpha, 0)
    
    return superimposed