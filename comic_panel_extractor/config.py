from dataclasses import dataclass

@dataclass
class Config:
    """Configuration settings for the comic-to-video pipeline."""
    input_path: str = ""
    output_folder: str = "temp_dir"
    distance_threshold: int = 70
    vertical_threshold: int = 30
    text_cood_file_name: str = "detect_and_group_text.json"
    min_text_length: int = 2

def get_text_cood_file_path(config: Config):
    return f'{config.output_folder}/{config.text_cood_file_name}'