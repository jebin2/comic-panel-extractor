# train.py
from yolo_manager import YOLOManager
from utils import Config, get_abs_path, backup_file
import os

def main():
    """Main training function."""
    try:
        # Initialize YOLO manager
        yolo_manager = YOLOManager()
        
        # Configuration
        data_yaml_path = get_abs_path('./comic.yaml')
        
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

if __name__ == "__main__":
    main()