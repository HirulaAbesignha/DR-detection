"""
Model architecture definitions for DR detection.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models

from .utils import CONFIG


def create_dr_model(num_classes=5, input_shape=None):
    """Create custom CNN model for diabetic retinopathy detection."""
    if input_shape is None:
        input_shape = (CONFIG['IMG_SIZE'], CONFIG['IMG_SIZE'], 3)
    
    # Simple but effective CNN
    inputs = layers.Input(shape=input_shape, name='input_image')
    
    # Block 1
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)
    
    # Block 2
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)
    
    # Block 3
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)
    
    # Block 4
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    
    # Dense layers with more capacity
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    
    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    
    # Output
    outputs = layers.Dense(num_classes, activation='softmax', dtype='float32')(x)
    
    model = models.Model(inputs, outputs, name='DR_CustomCNN')
    
    # Return model and None for base_model (to maintain compatibility)
    return model, None


def compile_model(model, learning_rate=None, use_focal_loss=False):
    """Compile model with optimizer and metrics."""
    if learning_rate is None:
        learning_rate = CONFIG['LEARNING_RATE']
    
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    
    # Choose loss function
    if use_focal_loss:
        import tensorflow.keras.backend as K
        
        def focal_loss(gamma=2.0, alpha=0.25):
            def focal_loss_fixed(y_true, y_pred):
                epsilon = K.epsilon()
                y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
                cross_entropy = -y_true * K.log(y_pred)
                loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy
                return K.sum(loss, axis=-1)
            return focal_loss_fixed
        
        loss_fn = focal_loss(gamma=2.0, alpha=0.25)
    else:
        loss_fn = 'categorical_crossentropy'
    
    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
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
    # For custom CNN, just return as-is
    if base_model is None:
        return None
    
    base_model.trainable = True
    
    # Freeze early layers
    for layer in base_model.layers[:num_layers_to_freeze]:
        layer.trainable = False
    
    trainable_count = sum([1 for layer in base_model.layers if layer.trainable])
    print(f"Base model unfrozen: {trainable_count} trainable layers")
    
    return base_model