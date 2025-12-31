"""
Model architecture definitions for DR detection.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models

from .utils import CONFIG


def create_dr_model(num_classes=5, input_shape=None):
    """Create DenseNet121 model with transfer learning for DR detection."""
    if input_shape is None:
        input_shape = (CONFIG['IMG_SIZE'], CONFIG['IMG_SIZE'], 3)
    
    # Use DenseNet121 pretrained on ImageNet
    print("Loading DenseNet121 with ImageNet weights...")
    base_model = keras.applications.DenseNet121(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape,
        pooling='avg'
    )
    
    # Freeze base model initially
    base_model.trainable = False
    print(f"Base model frozen: {len(base_model.layers)} layers")
    
    # Build full model
    inputs = layers.Input(shape=input_shape, name='input_image')
    
    # DenseNet base
    x = base_model(inputs, training=False)
    
    # Classification head - improved capacity
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)

    x = layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.2)(x)

    # Output layer
    outputs = layers.Dense(num_classes, activation='softmax', dtype='float32', name='predictions')(x)
    
    # Create model
    model = models.Model(inputs, outputs, name='DR_DenseNet121')
    
    print(f"✅ DenseNet121 model created: {model.count_params():,} parameters")
    print(f"   Base model: {base_model.count_params():,} parameters")
    print(f"   Trainable: {sum([keras.backend.count_params(w) for w in model.trainable_weights]):,} parameters")
    
    return model, base_model
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