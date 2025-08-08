# utils.py
import os
import shutil
from glob import glob
from typing import List, Union
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Configuration class to manage environment variables and paths."""
    YOLO_MODEL_NAME = os.getenv('YOLO_MODEL_NAME', 'default_model')
    DEFAULT_IMAGE_SIZE = 640
    SUPPORTED_EXTENSIONS = ['jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'PNG']

def get_abs_path(relative_path: str) -> str:
    """Convert relative path to absolute path."""
    return os.path.abspath(relative_path)

def get_image_paths(directories: Union[str, List[str]]) -> List[str]:
    """
    Get all image paths from given directories.
    
    Args:
        directories: Single directory path or list of directory paths
        
    Returns:
        List of image file paths
    """
    if isinstance(directories, str):
        directories = [directories]
    
    all_images = []
    for directory in directories:
        abs_dir = get_abs_path(directory)
        if not os.path.isdir(abs_dir):
            print(f"⚠️ Warning: Skipping non-directory {abs_dir}")
            continue
            
        # Support multiple image extensions
        for ext in Config.SUPPORTED_EXTENSIONS:
            pattern = os.path.join(abs_dir, f'*.{ext}')
            images = sorted(glob(pattern))
            all_images.extend(images)
    
    return list(set(all_images))  # Remove duplicates

def clean_directory(directory: str, create_if_not_exists: bool = True) -> None:
    """Clean directory contents or create if it doesn't exist."""
    if os.path.exists(directory):
        shutil.rmtree(directory)
    
    if create_if_not_exists:
        os.makedirs(directory, exist_ok=True)

def backup_file(source_path: str, backup_path: str) -> str:
    """Backup a file to specified location."""
    backup_path = get_abs_path(backup_path)
    os.makedirs(os.path.dirname(backup_path), exist_ok=True)
    shutil.copy(source_path, backup_path)
    print(f"✅ File backed up to: {backup_path}")
    return backup_path

# yolo_manager.py
import os
import cv2
from ultralytics import YOLO
from typing import List, Optional, Dict, Any
from utils import Config, get_abs_path, clean_directory

class YOLOManager:
    """Manages YOLO model training and inference operations."""
    
    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or Config.YOLO_MODEL_NAME
        self.model = None
    
    def load_model(self, weights_path: Optional[str] = None) -> YOLO:
        """Load YOLO model from weights or pretrained model."""
        if weights_path and os.path.isfile(weights_path):
            print(f"📦 Loading model from: {weights_path}")
            self.model = YOLO(weights_path)
        else:
            print("✨ Loading pretrained model 'yolo11s.pt'")
            self.model = YOLO("yolo11s.pt")
        return self.model
    
    def train(self, 
              data_yaml_path: str,
              run_name: Optional[str] = None,
              device: int = 0,
              resume: bool = True,
              **kwargs) -> YOLO:
        """
        Train YOLO model with given parameters.
        
        Args:
            data_yaml_path: Path to dataset YAML file
            run_name: Name for the training run
            device: Device to use for training
            resume: Whether to resume from checkpoint if available
            **kwargs: Additional training parameters
        """
        run_name = run_name or self.model_name
        checkpoint_path = f"runs/detect/{run_name}/weights/last.pt"
        
        # Check for existing checkpoint
        if resume and os.path.isfile(checkpoint_path):
            print(f"🔄 Resuming training from checkpoint: {checkpoint_path}")
            self.model = YOLO(checkpoint_path)
            resume_flag = True
        else:
            self.load_model()
            resume_flag = False
        
        # Default training parameters
        train_params = {
            'data': data_yaml_path,
            'imgsz': Config.DEFAULT_IMAGE_SIZE,
            'epochs': 200,
            'batch': 10,
            'name': run_name,
            'device': device,
            'cache': True,
            'project': 'runs/detect',
            'exist_ok': True,
            'resume': resume_flag
        }
        
        # Update with custom parameters
        train_params.update(kwargs)
        
        print(f"🚀 Starting training with parameters: {train_params}")
        self.model.train(**train_params)
        return self.model
    
    def validate(self) -> Dict[str, Any]:
        """Validate the model and return metrics."""
        if not self.model:
            raise ValueError("❌ No model loaded. Please train or load a model first.")
        
        metrics = self.model.val()
        print("📊 Validation Metrics:", metrics)
        return metrics
    
    def get_best_weights_path(self, run_name: Optional[str] = None) -> str:
        """Get path to best trained weights."""
        run_name = run_name or self.model_name
        weights_path = os.path.join('runs', 'detect', run_name, 'weights', 'best.pt')
        
        if not os.path.isfile(weights_path):
            raise FileNotFoundError(f"❌ Trained weights not found at: {weights_path}")
        
        return weights_path
    
    def annotate_images(self, image_paths: List[str], output_dir: str = 'temp_dir', image_size: int = None) -> None:
        """
        Annotate images with model predictions.
        
        Args:
            image_paths: List of image file paths
            output_dir: Directory to save annotated images
            image_size: Size for inference
        """
        if not self.model:
            raise ValueError("❌ No model loaded. Please load a model first.")
        
        if not image_paths:
            raise ValueError("❌ No images provided for annotation.")
        
        image_size = image_size or Config.DEFAULT_IMAGE_SIZE
        clean_directory(output_dir)
        
        print(f"🎨 Annotating {len(image_paths)} images...")
        
        for idx, image_path in enumerate(image_paths):
            if not os.path.isfile(image_path):
                print(f"⚠️ Warning: Skipping non-existent file {image_path}")
                continue
                
            print(f'🔍 Processing ({idx+1}/{len(image_paths)}): {os.path.basename(image_path)}')
            
            try:
                results = self.model(image_path, imgsz=image_size)
                annotated_frame = results[0].plot()
                
                # Use original filename with prefix
                original_name = os.path.basename(image_path)
                name, ext = os.path.splitext(original_name)
                save_path = os.path.join(output_dir, f'annotated_{name}{ext}')
                
                cv2.imwrite(save_path, annotated_frame)
                print(f'✅ Saved: {save_path}')
                
            except Exception as e:
                print(f"❌ Error processing {image_path}: {str(e)}")
        
        print(f"🎉 Annotation complete! Results saved to: {output_dir}")