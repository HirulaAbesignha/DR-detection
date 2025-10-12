"""
Model architecture definitions for DR detection.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB3

from .utils import CONFIG


def create_dr_model(num_classes=5, input_shape=None):
    """
    Create EfficientNetB3-based model for diabetic retinopathy detection.
    """
    if input_shape is None:
        input_shape = (CONFIG['IMG_SIZE'], CONFIG['IMG_SIZE'], 3)
    
    # Base model - EfficientNetB3
    base_model = EfficientNetB3(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape,
        pooling='avg'
    )
    
    # Freeze base model initially
    base_model.trainable = False
    
    # Build model
    inputs = layers.Input(shape=input_shape, name='input_image')
    x = base_model(inputs, training=False)
    
    # Custom classification head
    x = layers.Dense(512, activation='relu', name='dense_1')(x)
    x = layers.BatchNormalization(name='bn_1')(x)
    x = layers.Dropout(0.5, name='dropout_1')(x)
    
    x = layers.Dense(256, activation='relu', name='dense_2')(x)
    x = layers.BatchNormalization(name='bn_2')(x)
    x = layers.Dropout(0.4, name='dropout_2')(x)
    
    x = layers.Dense(128, activation='relu', name='dense_3')(x)
    x = layers.Dropout(0.3, name='dropout_3')(x)
    
    # Output layer
    outputs = layers.Dense(
        num_classes, 
        activation='softmax', 
        dtype='float32', 
        name='predictions'
    )(x)
    
    # Create model
    model = models.Model(inputs, outputs, name='DR_EfficientNetB3')
    
    return model, base_model


def compile_model(model, learning_rate=None):
    """Compile model with optimizer and metrics."""
    if learning_rate is None:
        learning_rate = CONFIG['LEARNING_RATE']
    
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=[
            'accuracy',
            keras.metrics.AUC(name='auc'),
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall')
        ]
    )
    
    return model


def unfreeze_base_model(base_model, num_layers_to_freeze=100):
    """Unfreeze base model for fine-tuning."""
    base_model.trainable = True
    
    # Freeze early layers
    for layer in base_model.layers[:num_layers_to_freeze]:
        layer.trainable = False
    
    trainable_count = sum([1 for layer in base_model.layers if layer.trainable])
    print(f"Base model unfrozen: {trainable_count} trainable layers")
    
    return base_model