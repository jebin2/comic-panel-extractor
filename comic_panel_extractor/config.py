from dataclasses import dataclass
from pathlib import Path

# Path to this script's directory
CURRENT_DIR = Path(__file__).parent.resolve()

@dataclass
class Config:
    """Configuration settings for the comic-to-video pipeline."""
    org_input_path: str = ""
    input_path: str = ""
    yolo_model_path: str = (CURRENT_DIR / "best.pt").resolve()
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

def get_text_cood_file_path(config: Config):
    return f'{config.output_folder}/{config.text_cood_file_name}'