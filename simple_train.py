"""
Simple training script that works.
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras
import sys
sys.path.append('src')

from src.simple_model import create_simple_working_model
from src.data_loader import load_dataset, DRDataGenerator

# Load data
print("Loading dataset...")
samples = load_dataset(sample_size=10000)

# Split
labels = [s['label'] for s in samples]
train_samples, test_samples = train_test_split(samples, test_size=0.2, stratify=labels, random_state=42)
train_labels = [s['label'] for s in train_samples]
train_samples, val_samples = train_test_split(train_samples, test_size=0.15, stratify=train_labels, random_state=42)

print(f"Train: {len(train_samples)}, Val: {len(val_samples)}, Test: {len(test_samples)}")

# Create generators
train_gen = DRDataGenerator(train_samples, batch_size=32, shuffle=True, augment=True)
val_gen = DRDataGenerator(val_samples, batch_size=32, shuffle=False, augment=False)
test_gen = DRDataGenerator(test_samples, batch_size=32, shuffle=False, augment=False)

# Calculate SIMPLE balanced weights
print("Calculating class weights...")
all_labels = np.array([s['label'] for s in train_samples])
class_weights_array = compute_class_weight('balanced', classes=np.unique(all_labels), y=all_labels)
class_weights = {i: weight for i, weight in enumerate(class_weights_array)}
print(f"Class weights: {class_weights}")

# Create model
print("Creating model...")
model, base_model = create_simple_working_model()
model.summary()

# Compile with SIMPLE settings
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),  # HIGHER learning rate
    loss='categorical_crossentropy',  # NO focal loss
    metrics=['accuracy']
)

# SIMPLE callbacks
callbacks = [
    keras.callbacks.ModelCheckpoint(
        './models/simple_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_accuracy',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        mode='max',
        verbose=1
    )
]

# Train
print("Training...")
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=30,
    callbacks=callbacks,
    class_weight=class_weights,
    verbose=1
)

# Evaluate
print("\nEvaluating...")
test_loss, test_acc = model.evaluate(test_gen)
print(f"\n✅ FINAL TEST ACCURACY: {test_acc*100:.2f}%")

# Save
model.save('./models/simple_final.h5')
print("Model saved!")