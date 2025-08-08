# train.py
from .yolo_manager import YOLOManager
from .utils import get_abs_path, backup_file
import os
from .config import Config
import yaml
import os
from pathlib import Path
import shutil

def create_filtered_dataset(original_dataset_path, output_filtered_dataset_path):
    """
    Create a filtered dataset with only images that have non-empty labels
    """
    shutil.rmtree(output_filtered_dataset_path, ignore_errors=True)
    original_path = Path(original_dataset_path)
    output_path = Path(output_filtered_dataset_path)
    
    # Create output directory structure
    output_images = output_path / "images"
    output_labels = output_path / "labels"
    
    for split in ['train', 'val', 'test']:
        (output_images / split).mkdir(parents=True, exist_ok=True)
        (output_labels / split).mkdir(parents=True, exist_ok=True)
    
    filtered_counts = {}
    
    for split in ['train', 'val', 'test']:
        original_images_dir = original_path / 'images' / split
        original_labels_dir = original_path / 'labels' / split
        
        output_images_dir = output_images / split
        output_labels_dir = output_labels / split
        
        if not original_images_dir.exists() or not original_labels_dir.exists():
            print(f"Skipping {split} - source directory not found")
            filtered_counts[split] = 0
            continue
        
        total_count = 0
        copied_count = 0
        
        # Process each image
        for img_file in original_images_dir.glob('*'):
            if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                total_count += 1
                label_file = original_labels_dir / f"{img_file.stem}.txt"
                
                # Check if label file exists and has content
                if label_file.exists():
                    with open(label_file, 'r') as f:
                        content = f.read().strip()
                        if content:  # Label file has content
                            # Copy image
                            shutil.copy2(img_file, output_images_dir / img_file.name)
                            # Copy label
                            shutil.copy2(label_file, output_labels_dir / label_file.name)
                            copied_count += 1
                        else:
                            print(f"Skipping {img_file.name} - empty label file")
                else:
                    print(f"Skipping {img_file.name} - no label file")
        
        filtered_counts[split] = copied_count
        print(f"{split.upper()} split: {copied_count}/{total_count} images copied")
    
    return filtered_counts

def create_filtered_yaml(output_filtered_dataset_path, filtered_counts):
    """
    Create the YAML file for the filtered dataset
    """
    output_path = Path(output_filtered_dataset_path)
    yaml_path = f'{Config.current_path}/filtered_comic.yaml'
    
    # Create YAML structure
    yaml_data = {
        'names': ['panel'],
        'nc': 1,
        'path': str(output_path),
        'train': str(output_path / 'images' / 'train'),
        'val': str(output_path / 'images' / 'val')
    }
    
    # Only add test if it has images
    if filtered_counts.get('test', 0) > 0:
        yaml_data['test'] = str(output_path / 'images' / 'test')
    
    # Write YAML file
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)
    
    print(f"\n✅ Created filtered dataset YAML: {yaml_path}")
    return yaml_path

def main():
    """Main training function."""
    try:
        # Initialize YOLO manager
        yolo_manager = YOLOManager()
        
        # Configuration
        data_yaml_path = f'{Config.current_path}/filtered_comic.yaml'
        
        if not os.path.isfile(data_yaml_path):
            raise FileNotFoundError(f"❌ Dataset YAML not found: {data_yaml_path}")
        
        print(f"🎯 Training model: {Config.YOLO_MODEL_NAME}")
        
        # Train model
        model = yolo_manager.train(
            data_yaml_path=data_yaml_path,
            run_name=Config.YOLO_MODEL_NAME
        )
        
        # Validate model
        metrics = yolo_manager.validate()
        
        # Backup best weights
        weights_path = yolo_manager.get_best_weights_path()
        backup_path = f'{Config.YOLO_MODEL_NAME}.pt'
        backup_file(weights_path, backup_path)
        
        print("🎉 Training completed successfully!")
        
    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        raise

if __name__ == "__main__":# Configuration
    # Configuration
    original_dataset_path = "/home/jebineinstein/git/comic-panel-extractor/comic_panel_extractor/dataset"
    output_filtered_dataset_path = "/home/jebineinstein/git/comic-panel-extractor/comic_panel_extractor/filtered_dataset"
    
    print("🔍 Starting dataset filtering...")
    print(f"📂 Source: {original_dataset_path}")
    print(f"📁 Output: {output_filtered_dataset_path}")
    
    # Create filtered dataset
    filtered_counts = create_filtered_dataset(original_dataset_path, output_filtered_dataset_path)
    
    # Create YAML file
    yaml_path = create_filtered_yaml(output_filtered_dataset_path, filtered_counts)
    
    # Summary
    total_filtered = sum(filtered_counts.values())
    print(f"\n📊 Filtering Summary:")
    for split, count in filtered_counts.items():
        if count > 0:
            print(f"   {split.upper()}: {count} images")
    print(f"   TOTAL: {total_filtered} images with labels")
    
    print(f"\n🎯 Use this YAML for training: {yaml_path}")
    
    # Display the created YAML content
    with open(yaml_path, 'r') as f:
        yaml_content = f.read()
    print(f"\n📄 Generated YAML content:")
    print("─" * 50)
    print(yaml_content)
    print("─" * 50)
    main()