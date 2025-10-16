from dataclasses import dataclass
import os
import toml

from dotenv import load_dotenv
load_dotenv()

CURRENT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__)))
CONFIG_FILE = f"{CURRENT_PATH}/config.toml"

# Load TOML config
if os.path.exists(CONFIG_FILE):
	config_data = toml.load(CONFIG_FILE)
else:
	raise FileNotFoundError(f"Config file not found: {CONFIG_FILE}")

@dataclass
class Config:
	"""Configuration settings for the comic-to-video pipeline."""
	current_path: str = CURRENT_PATH

	# Read from TOML config
	EPOCH: int = int(config_data.get("EPOCH", 200))
	YOLO_BASE_MODEL_NAME: str = config_data.get("YOLO_BASE_MODEL_NAME", "yolo11s-seg")
	YOLO_MODEL_NAME: str = config_data.get("YOLO_MODEL_NAME", f"comic_panel_{YOLO_BASE_MODEL_NAME}")
	IMAGE_SOURCE_PATH: str = config_data.get("IMAGE_SOURCE_PATH", "")

	# Derived paths
	yolo_base_model_path: str = f"{current_path}/{YOLO_BASE_MODEL_NAME}.pt"
	yolo_trained_model_path: str = f"{current_path}/{YOLO_MODEL_NAME}.pt"

	# Other parameters
	org_input_path: str = ""
	input_path: str = ""
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

	# Static constants
	DEFAULT_IMAGE_SIZE: int = 640
	SUPPORTED_EXTENSIONS: list = ('jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'PNG')

def get_text_cood_file_path(config: Config):
	"""Return full path to text coordinate file."""
	return f"{config.output_folder}/{config.text_cood_file_name}"