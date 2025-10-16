import os
import shutil
import random
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm
from .config import Config, load_config

load_dotenv()
config = load_config()
SOURCE_PATHS = config.IMAGE_SOURCE_PATH

if not SOURCE_PATHS:
    raise ValueError("SOURCE_PATH not set")

# Split by comma and strip whitespace
source_paths = [Path(p.strip()) for p in SOURCE_PATHS.split(',')]

images_dir = Path(f'{config.current_path}/images')
dataset_dir = Path(f'{config.current_path}/dataset')

image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
label_exts = {'.txt'}

# Copy images from all source paths with tqdm progress
for source_path in source_paths:
    if not source_path.exists():
        print(f"Warning: source path {source_path} does not exist, skipping.")
        continue
    
    # Count total image files first for progress bar
    total_files = 0
    for root, dirs, files in os.walk(source_path):
        total_files += sum(1 for f in files if Path(f).suffix.lower() in image_exts)
    
    with tqdm(total=total_files, desc=f"Copying images from {source_path}", unit="img") as pbar:
        for root, dirs, files in os.walk(source_path):
            root_path = Path(root)
            if root_path == source_path:
                prefix = 'root'
            else:
                rel_path = root_path.relative_to(source_path)
                prefix = '_'.join(rel_path.parts)
            for file in files:
                if Path(file).suffix.lower() in image_exts:
                    src_file = root_path / file
                    dst_file = images_dir / f"{prefix}_{file}"
                    shutil.copy2(src_file, dst_file)
                    pbar.update(1)

# Delete old dataset if exists
if dataset_dir.exists():
    shutil.rmtree(dataset_dir)

# Create dataset folders for images and labels splits
for split in ['train', 'val', 'test']:
    (dataset_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
    (dataset_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)

# List all images in images_dir
all_images = [f for f in images_dir.iterdir() if f.suffix.lower() in image_exts]

# Shuffle and split (80% train, 10% val, 10% test)
random.seed(42)
random.shuffle(all_images)
n = len(all_images)
train_end = int(0.8 * n)
val_end = train_end + int(0.1 * n)

splits = {
    'train': all_images[:train_end],
    'val': all_images[train_end:val_end],
    'test': all_images[val_end:]
}

label_src_dir = Path(f'{config.current_path}/image_labels')

# Move/copy images and labels to their split folders with tqdm
for split, files in splits.items():
    print(f"Processing split '{split}' with {len(files)} images...")
    for img_path in tqdm(files, desc=f"Copying {split}", unit="img"):
        # Copy image
        dst_img_path = dataset_dir / 'images' / split / img_path.name
        shutil.copy2(img_path, dst_img_path)

        # Copy label if exists
        stem = img_path.stem
        for ext in label_exts:
            label_file = label_src_dir / f"{stem}{ext}"
            if label_file.exists():
                dst_label_path = dataset_dir / 'labels' / split / label_file.name
                shutil.copy2(label_file, dst_label_path)
                break

print("Done!")
