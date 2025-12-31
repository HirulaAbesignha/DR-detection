"""
Simple, proven model that actually works.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models


def create_simple_working_model(num_classes=5, input_shape=(224, 224, 3)):
    """Create a simple CNN that ACTUALLY learns."""
    
    inputs = layers.Input(shape=input_shape)
    
    # Use MobileNetV2 (lighter, more stable than DenseNet)
    base_model = keras.applications.MobileNetV2(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape,
        pooling='avg'
    )
    
    base_model.trainable = False  # Freeze initially
    
    x = base_model(inputs, training=False)
    
    # SIMPLE head (no fancy regularization that's breaking it)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs, x, name='Simple_DR_Model')
    
    return model, base_model