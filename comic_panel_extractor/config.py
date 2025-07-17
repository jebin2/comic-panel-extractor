from dataclasses import dataclass

@dataclass
class Config:
    """Configuration settings for the comic-to-video pipeline."""
    input_path: str = ""
    output_folder: str = "temp_dir"
    distance_threshold: int = 70
    vertical_threshold: int = 30
    text_cood_path: str = f"{output_folder}/detect_and_group_text.json"
    min_text_length: int = 2