from src.data_loader import load_dataset

samples = load_dataset(sample_size=10)
print(f"\nLoaded {len(samples)} samples")

for i, sample in enumerate(samples[:5]):
    img = sample['image']
    print(f"Sample {i}: shape={img.shape}, dtype={img.dtype}, label={sample['label']}")