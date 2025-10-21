import keras
import numpy as np
from src.data_loader import load_dataset, DRDataGenerator
from src.utils import CONFIG

# Load model
print("Loading model...")
model = keras.models.load_model('./models/best_dr_model.h5', compile=False)

# Load test data (small sample)
print("Loading test data...")
test_samples = load_dataset(sample_size=1000)
test_gen = DRDataGenerator(test_samples, batch_size=16, shuffle=False, augment=False)

# Evaluate
print("Evaluating...")
results = model.evaluate(test_gen, verbose=1)

print(f"\n✅ Test Accuracy: {results[1]*100:.2f}%")
print(f"✅ Test Loss: {results[0]:.4f}")