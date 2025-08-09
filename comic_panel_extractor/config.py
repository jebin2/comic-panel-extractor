from dataclasses import dataclass
import os

from dotenv import load_dotenv
load_dotenv()

@dataclass
class Config:
	"""Configuration settings for the comic-to-video pipeline."""
	org_input_path: str = ""
	input_path: str = ""
	current_path = os.path.abspath(os.path.join(os.path.dirname(__file__)))
	YOLO_BASE_MODEL_NAME = os.getenv('YOLO_BASE_MODEL_NAME', 'yolo11s-seg')
	yolo_base_model_path: str = f'{current_path}/{YOLO_BASE_MODEL_NAME}.pt'
	YOLO_MODEL_NAME = f'{os.getenv('YOLO_MODEL_NAME', 'comic_panel')}_{YOLO_BASE_MODEL_NAME}'
	yolo_trained_model_path: str = f'{current_path}/{YOLO_MODEL_NAME}.pt'
	black_overlay_input_path: str = ""
	output_folder: str = "temp_dir"
	distance_threshold: int = 70
	vertical_threshold: int = 30
	text_cood_file_name: str = "detect_and_group_text.json"
	min_text_length: int = 2
	min_area_ratio: float = 0.05
	min_width_ratio: float = 0.15
	min_height_ratio: float = 0.15
	
	# Additional parameters for BorderPanelExtractor
	panel_filename_pattern: str = r"panel_\d+_\((\d+), (\d+), (\d+), (\d+)\)\.jpg"

	"""Configuration class to manage environment variables and paths."""
	DEFAULT_IMAGE_SIZE = 640
	SUPPORTED_EXTENSIONS = ['jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'PNG']

def get_text_cood_file_path(config: Config):
	return f'{config.output_folder}/{config.text_cood_file_name}'