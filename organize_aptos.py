import os
import pandas as pd
import shutil
from pathlib import Path
from collections import Counter

print("Reading APTOS labels...")
train_csv = pd.read_csv('data/aptos2019/train.csv')

print(f"\nFull APTOS Distribution:")
print(train_csv['diagnosis'].value_counts().sort_index())
print(f"Total images: {len(train_csv)}")

base_path = Path('data/aptos2019/organized')
for class_num in range(5):
    class_folder = base_path / f'class_{class_num}'
    class_folder.mkdir(parents=True, exist_ok=True)

print("\nCopying ALL images to class folders...")
for idx, row in train_csv.iterrows():
    image_id = row['id_code']
    diagnosis = row['diagnosis']
    
    src = f'data/aptos2019/train_images/{image_id}.png'
    dst = f'data/aptos2019/organized/class_{diagnosis}/{image_id}.png'
    
    if os.path.exists(src):
        shutil.copy(src, dst)
    
    if (idx + 1) % 500 == 0:
        print(f"  Copied {idx + 1}/{len(train_csv)}...")

print("\nOrganization complete!")

print("\nFinal structure:")
for class_num in range(5):
    folder = base_path / f'class_{class_num}'
    count = len(list(folder.glob('*.png')))
    print(f"  Class {class_num}: {count} images")

total = sum([len(list((base_path / f'class_{c}').glob('*.png'))) for c in range(5)])
print(f"\nTotal organized: {total} images")