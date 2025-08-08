# inference.py
from yolo_manager import YOLOManager
from utils import Config, get_abs_path, get_image_paths
import os

def run_inference(weights_path: str, images_dirs, output_dir: str = 'temp_dir') -> None:
    """
    Run inference on images using trained model.
    
    Args:
        weights_path: Path to model weights
        images_dirs: Directory or list of directories containing images
        output_dir: Directory to save annotated results
    """
    try:
        # Validate weights file
        weights_path = get_abs_path(weights_path)
        if not os.path.isfile(weights_path):
            raise FileNotFoundError(f"❌ Weights file not found: {weights_path}")
        
        # Get image paths
        image_paths = get_image_paths(images_dirs)
        if not image_paths:
            raise ValueError("❌ No images found in the provided directories.")
        
        print(f"🔍 Found {len(image_paths)} images for inference")
        
        # Initialize YOLO manager and load model
        yolo_manager = YOLOManager()
        yolo_manager.load_model(weights_path)
        
        # Run inference
        yolo_manager.annotate_images(image_paths, output_dir)
        
        print("🎉 Inference completed successfully!")
        
    except Exception as e:
        print(f"❌ Inference failed: {str(e)}")
        raise

def main():
    """Main inference function."""
    weights_path = f'{Config.YOLO_MODEL_NAME}.pt'
    images_dirs = [
        './dataset/images/train',
        './dataset/images/val', 
        './dataset/images/test'
    ]
    
    run_inference(weights_path, images_dirs, './temp_dir')

if __name__ == "__main__":
    main()